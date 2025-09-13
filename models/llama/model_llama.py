import math
import torch
from torch import nn
import torch.nn.functional as F

from layers import RMSNorm
from utils import ACT2FN, BaseModel

from .config_llama import LlamaConfig

# Taken from the llama repo
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, attention_mask, freqs_cis):
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        xq = self.q_proj(x).view(hidden_shape)
        xk = self.k_proj(x).view(hidden_shape)
        xv = self.v_proj(x).view(hidden_shape)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            # flash attention
            attn_output = F.scaled_dot_product_attention(
                xq, xk, xv,
                scale=self.scaling,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                enable_gqa=self.num_key_value_groups == 1
            )
        else:
            xk = repeat_kv(xk, self.num_key_value_groups)
            xv = repeat_kv(xv, self.num_key_value_groups)

            attn_weights = torch.matmul(xq, xk.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, xv)
            attn_output = attn_output.transpose(1, 2).contiguous()
    
        # re-assemble all head outputs side by side
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn    = ACT2FN[config.hidden_act]()
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))


class LlamaDecoder(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention = LlamaAttention(config)
        self.feed_forward = LlamaMLP(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask, freqs_cis):
        h = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(BaseModel):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoder(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings * 2,
            config.rope_theta,
        )

        # weight sharing scheme
        if config.tie_word_embeddings:
            self.tok_embeddings.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

         # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, labels=None, ignore_index=-100):
        T = input_ids.size(1)

        h = self.tok_embeddings(input_ids)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:T]

        mask = None
        if T > 1: # Causal Attention Mask
            mask = torch.full((T, T), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.type_as(h)

        for decoder_layer in self.layers:
            h = decoder_layer(
                h,
                mask,
                freqs_cis,
            )

        h = self.norm(h)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(logits.device)

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
