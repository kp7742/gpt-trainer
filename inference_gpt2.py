import torch

from transformers import AutoTokenizer
from models.gpt2 import GPT2Config, GPT2Model

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    print()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16 if 'cuda' in device else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1024
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(tokenizer)
    print()

    model_config = GPT2Config()
    print(model_config)
    print()

    model = GPT2Model.from_pretrained("gpt2_model.pt", model_config)
    print(model)
    print()
    
    model.to(device)

    model_size = sum(t.numel() for n, t in model.named_parameters() if t.requires_grad and "wte" not in n)
    total_model_size = sum(t.numel() for t in model.parameters() if t.requires_grad)
    print(f"Model dtype: {dtype}")
    print(f"Model size: {total_model_size}, {model_size} | {total_model_size/1000**2:.1f}M Total | {model_size/1000**2:.1f}M w/o Embd\n")

    print(model.generate(
        device, tokenizer, context_size=model_config.n_positions, 
        prompt="Lily and Tom were playing in the yard"))

if __name__ == "__main__":
    main()