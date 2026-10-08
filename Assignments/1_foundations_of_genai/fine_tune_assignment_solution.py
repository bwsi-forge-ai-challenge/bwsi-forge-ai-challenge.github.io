"""Instructor reference solution for fine_tune_assignment (exported to .py)."""

# %%
# Colab: uncomment if imports fail
# !pip install -q transformers datasets accelerate wandb

import os
import random
import textwrap
import time

import torch
from datasets import Dataset
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
WANDB_PROJECT = "forge-llm-finetune"  # TODO: add your name

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
    "Translate to pirate: Hello friend\n",
]

print("=== Base model outputs - SAVE for Part 4 ===\n")
for p in BASELINE_PROMPTS:
    result = generate_text(base_model, p)
    print("PROMPT:", repr(p))
    print(textwrap.fill(result, width=100))
    print("-" * 60)

# %%
part1_reflection = """The base model mostly continues the prompt rather than following instructions. distilgpt2 only
ever learned next-token prediction on raw web text (Lesson 1: How Do LLMs Actually Learn) - it was never shown
examples of 'here is an instruction, here is the correct response.' So when it sees 'Write a haiku about rain.',
the most probable next tokens (statistically, from its training data) are more prompt-like text, not an actual
haiku. ChatGPT-style assistants add a fine-tuning stage on instruction/response pairs plus RLHF on top of a base
model like this one; distilgpt2 stops after pretraining, so it behaves like a well-read autocomplete engine
instead of an assistant."""

# %%
TASK_NAME = 'casual_to_formal'
TRAINING_PAIRS = [
    ("hey can u send the report", "Could you please send the report at your earliest convenience."),
    ("sry im late", "I apologize for my tardiness."),
    ("thx for the help", "Thank you for your assistance."),
    ("gonna be out tmrw", "I will be out of the office tomorrow."),
    ("cant make the meeting", "I am unable to attend the meeting."),
    ("pls review asap", "Please review this document at your earliest convenience."),
    ("need this by eod", "I would appreciate receiving this by the end of the day."),
    ("whats the status", "Could you please provide a status update?"),
    ("lets sync later", "Let us schedule time to connect later."),
    ("idk what to do", "I am uncertain about the appropriate next steps."),
    ("fyi the client called", "For your information, the client contacted us."),
    ("sorry for the confusion", "I apologize for any confusion this may have caused."),
    ("can we push the deadline", "Would it be possible to extend the deadline?"),
    ("got it thanks", "Understood. Thank you for the clarification."),
    ("hey team quick update", "Hello team, I would like to share a brief update."),
    ("this looks good to me", "This appears satisfactory from my perspective."),
    ("any updates?", "Do you have any updates you can share?"),
    ("running 10 min late", "I anticipate arriving approximately ten minutes late."),
    ("can u hop on a call", "Would you be available for a brief call?"),
    ("we should talk about this", "I believe we should discuss this matter further."),
    ("not sure i agree", "I am not certain that I agree with that assessment."),
    ("sounds good", "That proposal works for me."),
    ("will do", "I will take care of that."),
    ("my bad", "I take responsibility for that oversight."),
    ("keep me posted", "Please keep me informed of any developments."),
    ("lmk when ur free", "Please let me know when you are available."),
    ("can we reschedule?", "Would it be possible to reschedule our meeting?"),
    ("just following up", "I am following up on my previous message."),
    ("per my last email", "As I noted in my previous email,"),
    ("heads up", "Please be advised that"),
    ("no worries", "There is no issue."),
    ("gotcha", "I understand."),
    ("ping me", "Please contact me when convenient."),
    ("looping in sarah", "I am copying Sarah for visibility."),
    ("out of pocket today", "I will be unavailable today."),
    ("touch base next week", "Let us connect next week."),
    ("draft looks fine", "The draft meets my expectations."),
    ("need more info", "I require additional information."),
    ("works for me", "That time is acceptable to me."),
    ("see u there", "I look forward to seeing you there."),
    ("thanks in advance", "Thank you in advance for your help."),
    ("as discussed", "As we discussed previously,"),
    ("please advise", "Please advise on the recommended course of action."),
    ("for the record", "For the record,"),
    ("please confirm receipt", "Please confirm that you have received this message."),
    ("i have a conflict", "I have a scheduling conflict at that time."),
    ("can you clarify", "Could you please clarify your request?"),
    ("please prioritize this", "Please prioritize this item."),
    ("when you get a chance", "When you have a moment, please"),
    ("appreciate your patience", "Thank you for your patience."),
]
print(len(TRAINING_PAIRS))

# %%
PROMPT_TEMPLATE = "### Input:\n{input}\n\n### Output:\n{output}"


def format_training_text(input_text, output_text):
    return PROMPT_TEMPLATE.format(input=input_text, output=output_text)


def format_inference_prompt(input_text):
    return f"### Input:\n{input_text}\n\n### Output:\n"


def build_dataset(pairs):
    return Dataset.from_dict({"text": [format_training_text(i, o) for i, o in pairs]})


print(build_dataset(TRAINING_PAIRS)[0]["text"])

# %%
LEARNING_RATE_RUN1 = 5e-5
LEARNING_RATE_RUN2 = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 128

# %%
def tokenize_batch(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def train_finetune_run(pairs, learning_rate, output_dir, run_name):
    if learning_rate is None or NUM_EPOCHS is None or BATCH_SIZE is None:
        raise ValueError("Complete TODO 2 first.")
    if len(pairs) < 50:
        raise ValueError(f"Need >= 50 pairs; have {len(pairs)}")

    dataset = build_dataset(pairs).map(tokenize_batch, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=5,
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
import os
if os.environ.get('WANDB_MODE') != 'disabled':
    try:
        import wandb
        wandb.login(reinit=True)
    except Exception as exc:
        print('W&B skipped', exc)
        os.environ['WANDB_MODE'] = 'disabled'

finetuned_model_1, time_1 = train_finetune_run(
    TRAINING_PAIRS, LEARNING_RATE_RUN1, './finetuned-run1', f'{TASK_NAME}-lr{LEARNING_RATE_RUN1}')
finetuned_model_2, time_2 = train_finetune_run(
    TRAINING_PAIRS, LEARNING_RATE_RUN2, './finetuned-run2', f'{TASK_NAME}-lr{LEARNING_RATE_RUN2}')
print(f'Run1 {time_1/60:.1f}m Run2 {time_2/60:.1f}m')

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
EVAL_PROMPTS = [
    "hey can we move the deadline to friday",
    "pls send the slides when u can",
    "sorry i missed ur call",
    "Question: What is photosynthesis?\nAnswer:",
    "Write a poem about the ocean.",
]

for prompt in EVAL_PROMPTS:
    print("INPUT:", prompt)
    print("\nBASE:")
    print(textwrap.fill(generate_text(base_model, prompt), width=100))
    if finetuned_model:
        print("\nFINE-TUNED:")
        print(textwrap.fill(generate_text(finetuned_model, format_inference_prompt(prompt)), width=100))
    print("=" * 72)

# %%
part4_reflection = """
1. Did fine-tuning help? How can you tell?
   Yes - on held-out casual inputs, the fine-tuned model produces text that follows the
   "### Input: / ### Output:" format and shifts toward formal business phrasing, instead of
   rambling or repeating the prompt like the base model does. The clearest evidence is the
   side-by-side comparisons above: the base model continues text unpredictably, while the
   fine-tuned model reliably attempts a formal rewrite.

2. What did the W&B loss curves look like?
   Both runs should show loss decreasing over the ~3 epochs. The lower learning rate (5e-5)
   typically decreases smoothly; the higher one (2e-4) may drop faster but can look noisier or
   spike, since bigger update steps are more likely to overshoot on this small dataset.

3. Fine-tune time vs pretraining (Lesson 1)?
   Fine-tuning distilgpt2 on ~50 examples for 3 epochs takes a few minutes on a Colab GPU.
   Pretraining a model like this from scratch takes weeks to months on large GPU/TPU clusters
   and trillions of tokens - fine-tuning is a small nudge to already-learned weights, not
   training from zero.

4. Tradeoffs: distilgpt2 vs GPT-4-class models?
   distilgpt2 (~82M parameters) is fast, free, and easy to fine-tune locally, but it has far
   less world knowledge, weaker reasoning, and no instruction-tuning or RLHF - it only learns
   the narrow pattern in its training data. A GPT-4-class model already follows instructions
   well out of the box and needs little or no fine-tuning for most tasks, but it costs more
   per call, is slower, and you cannot download or fully customize its weights.
"""
print("Fill in the reflection before submitting.")
