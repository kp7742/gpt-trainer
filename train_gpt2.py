import torch

from torchinfo import summary
from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from utils import Trainer, TrainerConfig
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

    batch_size = 1
    context_size = 1024
    device_count = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16 if 'cuda' in device else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1024
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
    micro_batch = 32
    learning_rate = 6e-4
    max_steps = 32000 # len(train_dataloader)
    total_rows = len(tinystories_ds["train"])
    effective_steps = max_steps // micro_batch

    warmup_steps = int(effective_steps * 0.1) # 2000
    lr_decay_iters = effective_steps - warmup_steps
    min_learning_rate = learning_rate * 0.1

    print(f"Effective Steps: {effective_steps}")
    print(f"Estimated Total Steps: {total_rows // (batch_size * micro_batch * device_count)}")
    print(f"Effective Batch Size: {(context_size * batch_size * micro_batch * device_count)}\n")

    model_config = GPT2Config()
    print(model_config)
    print()

    model = GPT2Model(model_config)
    print(model)
    print()

    model.to(dtype=dtype)

    model_size = sum(t.numel() for n, t in model.named_parameters() if t.requires_grad and "wte" not in n)
    total_model_size = sum(t.numel() for t in model.parameters() if t.requires_grad)
    print(f"Model dtype: {dtype}")
    print(f"Model size: {total_model_size}, {model_size} | {total_model_size/1000**2:.1f}M Total | {model_size/1000**2:.1f}M w/o Embd\n")

    ids = torch.randint(0, context_size, size=(1, context_size), dtype=torch.int64)
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
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-08,
        weight_decay=1e-1,
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

    # trainer.save_model("gpt2_model.pt")

if __name__ == "__main__":
    main()