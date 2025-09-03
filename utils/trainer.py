import math
import time
import tqdm
import torch
import inspect
import torch.nn.functional as F

from torch import nn
from datasets import Dataset
from contextlib import nullcontext
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerBase
from transformers import DataCollatorForLanguageModeling

from .preconfig import PreConfig
from .time_utils import timeSince, timePassed, tokenSpeed

class TrainerConfig(PreConfig):
    def __init__(
        self,
        seed=1337,
        batch_size=1,
        context_size=1024,
        n_epochs=1,
        micro_batch=1,
        learning_rate=1e-2,
        min_learning_rate=0,
        lr_decay_iters=0,
        max_steps=1,
        warmup_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        grad_clip=1.0,
        weight_decay=0.1,
        eval_iters=1000,
        log_interval=500,
        eval_interval=2000,
        sample_interval=1000,
        sample_size=1024,
        sample_top_k=None,
        sample_temperature=1.0,
        sample_prompt="Lily and Tom were playing in the yard",
        compile=False,
        mix_prec=False,
        dtype=torch.float32,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.context_size = context_size
        self.n_epochs = n_epochs
        self.micro_batch = micro_batch
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.eval_iters = eval_iters
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.sample_interval = sample_interval
        self.sample_size = sample_size
        self.sample_top_k = sample_top_k
        self.sample_temperature = sample_temperature
        self.sample_prompt = sample_prompt
        self.compile = compile
        self.mix_prec = mix_prec
        self.dtype = dtype


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            config: TrainerConfig,
            tokenizer: PreTrainedTokenizerBase,
            train_dataset: Dataset,
            eval_dataset: Dataset,
        ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.effective_steps = self.config.max_steps // self.config.micro_batch

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.auto_ctx = nullcontext() if self.device == 'cpu' or self.device == 'mps' else torch.amp.autocast(device_type=self.device, dtype=self.config.dtype)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # Set mlm=False for causal language modeling (GPT-like models)
        )

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.data_collator)
        self.test_dataloader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.data_collator)

        ## training pre_setup
        torch.manual_seed(self.config.seed)
        # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision('high') # May able to use mix precision

    def get_batch(self, iter):
        batch = next(iter)
        input, target = batch['input_ids'], batch['labels']
        input, target = input.to(self.device), target.to(self.device)
        return input, target

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_steps:
            return self.config.learning_rate * (it + 1) / (self.config.warmup_steps + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_steps) / (self.config.lr_decay_iters - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_learning_rate + coeff * (self.config.learning_rate - self.config.min_learning_rate)

    def configure_optimizers(self):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.adam_beta1, self.config.adam_beta2), eps=self.config.adam_eps, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def cal_ppl(self, input_ids, target_ids):
        max_length = self.config.context_size
        stride = max_length
        seq_len = input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        # prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            # trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

            input_ids = input_ids[:, begin_loc:end_loc] # .to(device)
            target_ids = target_ids[:, begin_loc:end_loc]
            # target_ids = input_ids.clone()
            # target_ids[:, :-trg_len] = -100

            with self.auto_ctx:
                _, loss = self.model(input_ids, target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            # prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        return torch.exp(avg_nll)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        # Few Training Samples
        losses = torch.zeros(self.config.eval_iters)
        train_batch = iter(self.train_dataloader)
        for s in tqdm.tqdm(range(self.config.eval_iters), desc="Training Eval"):
            input, target = self.get_batch(train_batch)

            with self.auto_ctx:
                _, loss = self.model(input, target)

            if "cuda" in self.device:
                torch.cuda.synchronize() # wait for the GPU to finish work

            losses[s] = loss.item()
        out['train'] = losses.mean()

        # All Test Samples
        test_iter = self.config.eval_iters
        losses = torch.zeros(test_iter)
        test_batch = iter(self.test_dataloader)
        for s in tqdm.tqdm(range(test_iter), desc="Testing Eval"):
            input, target = self.get_batch(test_batch)

            with self.auto_ctx:
                _, loss = self.model(input, target)

            if "cuda" in self.device:
                torch.cuda.synchronize() # wait for the GPU to finish work

            losses[s] = loss.item()
        out['test'] = losses.mean()

        # PPL of model
        ppls = torch.zeros(test_iter)
        test_batch = iter(self.test_dataloader)

        for s in tqdm.tqdm(range(test_iter), desc="Testing PPL"):
            input_ids, target_ids = self.get_batch(test_batch)

            ppls[s] = self.cal_ppl(input_ids, target_ids)
        out['ppl'] = ppls.mean()

        # Enable training
        self.model.train()
        return out

    def train(self):
        scaler = None
        if self.config.mix_prec:
            scaler = torch.amp.GradScaler(
                self.device, enabled=(self.dtype == torch.float16))

        optimizer = self.configure_optimizers()

        if self.config.compile:
            self.model = torch.compile(self.model)

        self.model.to(dtype=self.config.dtype, device=self.device)

        start = time.time()

        self.model.train()

        for epoch in range(1, self.config.n_epochs + 1):
            epoch_loss = 0
            curr_epoch = time.time()

            train_batch = iter(self.train_dataloader)

            input, target = self.get_batch(train_batch)

            for step in tqdm.tqdm(range(self.effective_steps), desc="GPT Training"):
                curr_step = time.time()

                lr = self.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                if step % self.config.eval_interval == 0:
                    losses = self.estimate_loss()
                    print("Eval Step: {} |".format(step),
                        "Train Loss: {:.4f} |".format(losses['train']),
                        "Test Loss: {:.4f} |".format(losses['test']),
                        "Test PPL: {:.4f}".format(losses['ppl']),'\n')

                for _ in range(self.config.micro_batch):
                    with self.auto_ctx:
                        _, loss = self.model(input, target)

                        # scale the loss to account for gradient accumulation
                        loss = loss / self.config.micro_batch

                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    input, target = self.get_batch(train_batch)

                    # backward pass
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # clip the gradient
                if self.config.grad_clip != 0.0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                if "cuda" in self.device:
                    torch.cuda.synchronize() # wait for the GPU to finish work

                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.config.micro_batch
                epoch_loss += lossf

                if step % self.config.log_interval == 0:
                    print("Step: {} {:.2f} |".format(timeSince(curr_epoch, step / self.effective_steps), step / self.effective_steps * 100),
                        "Loss: {:.4f} |".format(lossf),
                        "PPL: {:.4f} |".format(self.cal_ppl(input, target)),
                        "DT: {} |".format(timePassed(curr_step)),
                        "Tok/s: {}".format(tokenSpeed(curr_step, self.config.batch_size * self.config.context_size)), '\n')

                if step % self.config.sample_interval == 0:
                    print('Sample')
                    print('----\n{}\n----\n'.format(
                        self.model.generate(
                            self.device, self.tokenizer, context_size=self.config.context_size,
                            prompt=self.config.sample_prompt, size=self.config.sample_size,
                            temperature=self.config.sample_temperature, top_k=self.config.sample_top_k)))
                    self.model.train()

            epoch_loss = epoch_loss / self.effective_steps

            print("Epoch: {} {:.2f} |".format(timeSince(start, epoch / self.config.n_epochs), epoch / self.config.n_epochs * 100),
                "Loss: {:.4f}".format(epoch_loss), '\n')

        # Final stats and sample
        final_losses = self.estimate_loss()
        print("Train Loss: {:.4f} |".format(final_losses['train']),
            "Test Loss: {:.4f} |".format(final_losses['test']),
            "Test PPL: {:.4f}".format(final_losses['ppl']),'\n')
        
        print('Sample')
        print('----\n{}\n----\n'.format(
            self.model.generate(
                self.device, self.tokenizer, context_size=self.config.context_size,
                prompt=self.config.sample_prompt, size=self.config.sample_size,
                temperature=self.config.sample_temperature, top_k=self.config.sample_top_k)))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
