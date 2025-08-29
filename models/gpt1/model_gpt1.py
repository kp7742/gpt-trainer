import math
import torch
from torch import nn
import torch.nn.functional as F

from layers import Conv1D
from utils import ACT2FN, BaseModel

from .config_gpt1 import GPT1Config

class GPT1Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.attn_pdrop = config.attn_pdrop
        self.head_dim = self.n_embd // self.n_head # 768/12 = 64 For each head
        self.scaling = (1.0 / math.sqrt(self.head_dim))

        # query, key and value in single layer
        self.c_attn = Conv1D(3 * config.n_embd, config.n_embd)
        # output projection
        self.c_proj = Conv1D(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,
            )

    def forward(self, x):
        B, T, C = x.size() # batch_size, num_tokens, n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
    
        # causal self-attention; Self-attend:(B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        if self.flash:
            attn_weights = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_pdrop if self.training else 0.0, 
                is_causal=True
            ) # flash attention
        else:
            # manual implementation of attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling # Dot product for each head
            attn_weights = attn_weights.masked_fill(self.mask[:,:,:T,:T] == 0, -torch.inf) # Apply mask to lower left triangle
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_weights = torch.matmul(attn_weights, v) # (B, n_head, T, T) x (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
    
        # re-assemble all head outputs side by side
        attn_output = attn_weights.transpose(1, 2).contiguous().view(B, T, C)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPT1MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Conv1D(config.n_inner, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_inner)
        self.act = ACT2FN[config.activation_function]()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class GPT1Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GPT1Attention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT1MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class GPT1Model(BaseModel):
    def __init__(self, config: GPT1Config):
        super().__init__()
        self.config = config
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT1Block(config) for _ in range(config.n_layer)])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer("position_ids", torch.arange(config.n_positions), persistent=False)

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, labels=None, ignore_index=-100):
        # idx is of shape (B, T)
        T = input_ids.size(1)
        assert T <= self.config.n_positions, f"Cannot forward sequence of length {T}, block size is only {self.config.n_positions}"
        position_ids = self.position_ids[None, : T]

        inputs_embeds = self.tokens_embed(input_ids) # token embeddings of shape (B, T, n_embd)
        position_embeds = self.positions_embed(position_ids) # position embeddings of shape (T, n_embd)

        x = inputs_embeds + position_embeds
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        logits = self.lm_head(x)

        if labels is not None:
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
            loss = None

        return logits, loss
