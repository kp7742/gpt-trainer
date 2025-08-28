import multiprocessing

from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

cpu_count = multiprocessing.cpu_count()
num_workers = int(cpu_count) // 2
context_size = 1024
print(f"Using {cpu_count} CPU cores | {num_workers} Workers | {context_size} Context Size\n")

tinystories_ds = load_dataset("noanabeshima/TinyStoriesV2", split="train")
print(tinystories_ds, '\n')

tinystories_ds = tinystories_ds.train_test_split(test_size=0.02, shuffle=True)
print(tinystories_ds, '\n')

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokenizer.padding_side = "right"
tokenizer.model_max_length = 1024

print(tokenizer, '\n')

def tokenize_pretrain(examples):
    # max_length=context_size, padding=False, truncation=True,
    text = [f"{s.strip()}{tokenizer.eos_token}" for s in examples['text']]
    return tokenizer(text, max_length=context_size, padding=False, truncation=True, return_token_type_ids=False, return_overflowing_tokens=False, return_length=False)

tok_pretrain_ds = tinystories_ds.map(tokenize_pretrain, batched=True, num_proc=num_workers, remove_columns=['text'])
print(tok_pretrain_ds, '\n')

print("Sample:", tokenizer.decode(tok_pretrain_ds["train"][3]['input_ids']), '\n')

# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= context_size:
#         total_length = (total_length // context_size) * context_size
#     # Split by chunks of block_size.
#     result = {
#         k: [t[i : i + context_size] for i in range(0, total_length, context_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# tok_pretrain_ds = tok_pretrain_ds.map(group_texts, batched=True, num_proc=num_workers)
# print(tok_pretrain_ds, '\n')

print("Saving Dataset:\n")
tok_pretrain_ds.save_to_disk("data/pretraining_ds")