"""Instructor reference solution for fine_tune_assignment (exported to .py)."""

# %%
# Colab: uncomment if imports fail
# !pip install -q transformers datasets accelerate wandb

import os
import random
import textwrap
import time

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("Tip: enable Colab GPU before Part 3 for faster training.")

MODEL_NAME = "distilgpt2"

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
base_model.to(device)
base_model.eval()

print(f"Loaded {MODEL_NAME}")
print(f"Parameters: {base_model.num_parameters() / 1e6:.1f} million")

# %%
def generate_text(model, prompt, max_new_tokens=80, temperature=0.8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


BASELINE_PROMPTS = [
    "Once upon a time",
    "Question: What is the capital of France?\nAnswer:",
    "Write a haiku about rain.",
    "Translate to Yoda-speak: The Force is strong in you.\n",
]

print("=== Base model outputs - SAVE for Part 4 ===\n")
for p in BASELINE_PROMPTS:
    result = generate_text(base_model, p)
    print("PROMPT:", repr(p))
    print(textwrap.fill(result, width=100))
    print("-" * 60)

# %%
part1_reflection = """The base model mostly continues the prompt instead of following instructions. distilgpt2 only
learned next-token prediction on raw web text, so given 'Write a haiku about rain.' it predicts more
prompt-like text rather than a haiku. It was never shown instruction/response examples; that comes from
fine-tuning (and RLHF), which this model skipped."""

# %%
TASK_NAME = "yoda"
DATASET_NAME = "dvgodoy/yoda_sentences"
INPUT_COLUMN = "sentence"
OUTPUT_COLUMN = "translation_extra"

raw_dataset = load_dataset(DATASET_NAME, split="train")
splits = raw_dataset.train_test_split(test_size=0.1, seed=SEED)
train_data, eval_data = splits["train"], splits["test"]

print(f"Train: {len(train_data)} examples | Held out: {len(eval_data)} examples")
for ex in train_data.select(range(3)):
    print(ex[INPUT_COLUMN], "->", ex[OUTPUT_COLUMN])

# %%
PROMPT_TEMPLATE = "### Input:\n{input}\n\n### Output:\n{output}"


def format_training_text(input_text, output_text):
    return PROMPT_TEMPLATE.format(input=input_text, output=output_text)


def format_inference_prompt(input_text):
    return f"### Input:\n{input_text}\n\n### Output:\n"


def build_dataset(examples):
    texts = [format_training_text(ex[INPUT_COLUMN], ex[OUTPUT_COLUMN]) for ex in examples]
    return Dataset.from_dict({"text": texts})


print(build_dataset(train_data)[0]["text"])

# %%
LEARNING_RATE_RUN1 = 5e-5
LEARNING_RATE_RUN2 = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 128

# %%
def tokenize_batch(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def train_finetune_run(train_data, learning_rate, output_dir, run_name):
    if learning_rate is None or NUM_EPOCHS is None or BATCH_SIZE is None:
        raise ValueError("Complete TODO 1 first.")

    dataset = build_dataset(train_data).map(tokenize_batch, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {output_dir} in {elapsed/60:.1f} min - check W&B for loss curve")
    return model, elapsed

# %%
import wandb
wandb.login()

# %%
finetuned_model_1, time_1 = train_finetune_run(
    train_data, LEARNING_RATE_RUN1, "./finetuned-run1", f"{TASK_NAME}-lr{LEARNING_RATE_RUN1}")
finetuned_model_2, time_2 = train_finetune_run(
    train_data, LEARNING_RATE_RUN2, "./finetuned-run2", f"{TASK_NAME}-lr{LEARNING_RATE_RUN2}")

# %%
FINETUNED_DIR = "./finetuned-run1"  # switch if run 2 was better

try:
    finetuned_model = AutoModelForCausalLM.from_pretrained(FINETUNED_DIR).to(device)
    finetuned_model.eval()
    print("Loaded", FINETUNED_DIR)
except Exception as e:
    print("Train in Part 3 first.", e)
    finetuned_model = None

# %%
EVAL_EXAMPLES = [(ex[INPUT_COLUMN], ex[OUTPUT_COLUMN]) for ex in eval_data.select(range(3))] + [
    ("The Force is strong in you!", None),
    ("Write a poem about the ocean.", None),
]


def first_response(text):
    # keep only what follows "### Output:" up to the first line break
    return text.split("### Output:\n")[-1].split("\n")[0].strip()


for prompt, expected in EVAL_EXAMPLES:
    print("INPUT:", prompt)
    if expected:
        print("EXPECTED:", expected)
    print("\nBASE:")
    print(textwrap.fill(generate_text(base_model, prompt, max_new_tokens=40), width=100))
    if finetuned_model:
        print("\nFINE-TUNED:")
        print(first_response(generate_text(finetuned_model, format_inference_prompt(prompt), max_new_tokens=40)))
    print("=" * 72)

# %%
part4_reflection = """Yes, fine-tuning helped. On held-out sentences the base model repeats or rambles, while the
fine-tuned model follows the Input/Output format and reorders the sentence into Yoda-speak, often ending with
'Yes, hrrrm.' like the training outputs. In W&B both runs' loss fell steadily over the 3 epochs. The lower
learning rate (5e-5) gave a smoother curve; the higher one (2e-4) dropped faster but was noisier. Fine-tuning
took a few minutes, versus weeks for pretraining, because it only nudges weights the base model already had."""
