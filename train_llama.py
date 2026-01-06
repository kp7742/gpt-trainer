import torch

from torchinfo import summary
from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from utils import Trainer, TrainerConfig
from models.llama import LlamaConfig, LlamaModel

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

    batch_size = 16
    context_size = 1024
    device_count = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16 if 'cuda' in device else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("PleIAs/Monad")

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = context_size
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(tokenizer)
    print()
    
    tinystories_ds = load_from_disk("data/pretraining_ds")
    print(tinystories_ds)
    print()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Set mlm=False for causal language modeling (GPT-like models)
    )

    train_dataloader = DataLoader(tinystories_ds["train"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    print(f"Training Steps: {len(train_dataloader)}")
    print()

    test_dataloader = DataLoader(tinystories_ds["test"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    print(f"Validation Steps: {len(test_dataloader)}")
    print()

    # Training Config
    n_epochs = 1
    learning_rate = 8e-3
    min_learning_rate = learning_rate * 0.1

    final_batch = 512 # 512 with 2048 context, ~1.05M
    micro_batch = int(final_batch / batch_size)

    total_rows = 512000 * 2 # ~2B tokens # len(tinystories_ds["train"])
    total_tokens = total_rows * context_size

    total_steps = int(total_rows // (batch_size * micro_batch))
    warmup_steps = int(total_steps * 0.1) # 2000
    lr_decay_iters = total_steps - warmup_steps

    print(f"Effective Steps: {total_steps}")
    print(f"Estimated Tokens: {total_tokens/10**12:.2f}T | {total_tokens/10**9:.2f}B | {total_tokens/10**6:.2f}M")
    # print(f"Estimated Total Steps: {total_rows // (batch_size * micro_batch * device_count)}")
    print(f"Effective Batch Size: {(context_size * batch_size * micro_batch * device_count)/1000**2:.2f}M")
    print()

    model_config = LlamaConfig()

    model_config.max_position_embeddings = context_size
    model_config.vocab_size = tokenizer.vocab_size
    model_config.hidden_size = 256
    model_config.intermediate_size = 2 * model_config.hidden_size
    model_config.num_hidden_layers = 8
    model_config.num_attention_heads = 2
    model_config.num_key_value_heads = 1

    print(model_config)
    print()

    model = LlamaModel(model_config)
    print(model)
    print()

    model.to(dtype=dtype)

    model_size1 = sum(t.numel() for _, t in model.state_dict().items())
    model_size2 = sum(t.numel() for n, t in model.state_dict().items() if "tok_embeddings" not in n)
    model_size3 = sum(t.numel() for n, t in model.named_parameters() if t.requires_grad and "tok_embeddings" not in n)
    print(f"Model dtype: {dtype}")
    print(f"Model size: {model_size1}, {model_size2}, {model_size3} | {model_size1/1000**2:.1f}M Total | {model_size2/1000**2:.1f}M w/o Embd | {model_size3/1000**2:.1f}M w/o Embd(Tie)")
    print()

    ids = torch.randint(0, context_size, size=(batch_size, context_size), dtype=torch.int64)
    print(f"Test Data: {ids.size()}\n")

    summary(model, input_data=(ids,))
    print()

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        context_size=context_size,
        n_epochs=n_epochs,
        micro_batch=micro_batch,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        lr_decay_iters=lr_decay_iters,
        max_steps=total_steps,
        warmup_steps=warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        weight_decay=1e-2,
        log_interval=10,
        eval_interval=500,
        sample_interval=100,
        dtype=dtype
    )

    print(trainer_config)
    print()

    trainer = Trainer(
        model=model,
        config=trainer_config,
        tokenizer=tokenizer,
        train_dataset=tinystories_ds["train"],
        eval_dataset=tinystories_ds["test"]
    )

    trainer.train()

    # trainer.save_model("llama_model.pt")

if __name__ == "__main__":
    main()
