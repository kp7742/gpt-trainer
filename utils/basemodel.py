import torch
from torch import nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, path, override_config):
        model = cls(override_config)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

    """
    Taken from the nanoGPT repo.
    """
    @torch.no_grad()
    def generate(self, device, tokenizer, context_size, prompt="Hello, I'm a language model,", size=100, temperature=1.0, top_k=None):
        self.eval()

        enc_prompt = tokenizer.encode(prompt)

        input_ids = torch.tensor(enc_prompt, dtype=torch.long)

        input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device)

        for _ in range(size):
            # if the sequence context is growing too long we must crop it at block_size
            curr_ids = input_ids if input_ids.size(1) <= context_size else input_ids[:, -context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(curr_ids) # (B, T, vocab_size)
            if "cuda" in device:
                torch.cuda.synchronize() # wait for the GPU to finish work
            # pluck the logits at the last step and scale by desired temperature
            logits = logits[:, -1, :]
            if temperature is not None:
                logits = logits / temperature # (B, vocab_size)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            # end of text found
            if idx_next[0] == tokenizer.eos_token_id:
                break

        tokens = input_ids[0, :size].tolist()
        try:
            eos_idx = tokens.index(tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        except ValueError:
            pass
        decoded = tokenizer.decode(tokens)
        # decoded = tokenizer.decode(input_ids[0].tolist())
        return decoded
