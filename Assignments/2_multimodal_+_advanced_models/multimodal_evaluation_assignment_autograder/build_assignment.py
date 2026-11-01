"""Builds the multimodal evaluation assignment from one source of truth.

Usage (from anywhere): python3 build_assignment.py

Writes into Assignments/2_multimodal_+_advanced_models/:
  multimodal_evaluation_assignment.ipynb            student handout
  multimodal_evaluation_assignment_solution.ipynb   solution
  multimodal_evaluation_assignment_autograder/provided.py            the PROVIDED code, verbatim
  multimodal_evaluation_assignment_autograder/reference_solution.py  reference graded functions

Code cells given as code(student, solution) differ between the two notebooks. Snippets that
the autograder also needs are defined once below and reused, so the grader always runs exactly
the code students were given.
"""
import json
import os

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG_DIR = os.path.join(OUT, "multimodal_evaluation_assignment_autograder")
cells = []


def md(src):
    cells.append(("markdown", src.strip("\n"), None))


def code(src, solution=None):
    cells.append(("code", src.strip("\n"), None if solution is None else solution.strip("\n")))


# ===========================================================================
# Shared snippets (notebook + autograder)
# ===========================================================================
IMPORTS_SRC = '''
import json
import os
import random
import string
import time

import numpy as np
from PIL import Image

SEED = 0
'''

LOAD_SRC = '''
# Loading images (PROVIDED)
IMAGE_DIR = "multimodal_eval_images"


def load_image(name):
    """Open a photo by file name: one of the test photos in IMAGE_DIR, or any other image file
    that exists, such as a photo you uploaded to Colab."""
    path = name if os.path.exists(name) else os.path.join(IMAGE_DIR, name)
    return Image.open(path).convert("RGB")


def blank_image():
    """The 'blind' input: a plain gray image with nothing in it."""
    return Image.new("RGB", (384, 384), (128, 128, 128))


PROMPT_SUFFIX = " Answer with one word or number."
'''

TESTSET_SRC = '''
# The class test set (PROVIDED). Do not edit: everyone tests on the same 30 photo pairs.
TEST_SET = [
    {'id': 'count-0', 'category': 'count', 'question': 'How many apples are in the image: two or eight?',
     'answer': '2', 'choices': ['2', '8'], 'image': 'count-0.jpg', 'cf_image': 'count-0_cf.jpg', 'cf_answer': '8'},
    {'id': 'count-1', 'category': 'count', 'question': 'How many eggs are in the image: one or two?',
     'answer': '1', 'choices': ['1', '2'], 'image': 'count-1.jpg', 'cf_image': 'count-1_cf.jpg', 'cf_answer': '2'},
    {'id': 'count-2', 'category': 'count', 'question': 'How many cats are in the image: one or two?',
     'answer': '2', 'choices': ['1', '2'], 'image': 'count-2.jpg', 'cf_image': 'count-2_cf.jpg', 'cf_answer': '1'},
    {'id': 'count-3', 'category': 'count', 'question': 'How many cats are in the image: one or two?',
     'answer': '1', 'choices': ['1', '2'], 'image': 'count-3.jpg', 'cf_image': 'count-3_cf.jpg', 'cf_answer': '2'},
    {'id': 'count-4', 'category': 'count', 'question': 'How many traffic cones are in the image: one or two?',
     'answer': '1', 'choices': ['1', '2'], 'image': 'count-4.jpg', 'cf_image': 'count-4_cf.jpg', 'cf_answer': '2'},
    {'id': 'count-5', 'category': 'count', 'question': 'How many horses are in the image: two or eight?',
     'answer': '2', 'choices': ['2', '8'], 'image': 'count-5.jpg', 'cf_image': 'count-5_cf.jpg', 'cf_answer': '8'},
    {'id': 'color-0', 'category': 'color', 'question': 'What color is the car: red or orange?',
     'answer': 'red', 'choices': ['red', 'orange'], 'image': 'color-0.jpg', 'cf_image': 'color-0_cf.jpg', 'cf_answer': 'orange'},
    {'id': 'color-1', 'category': 'color', 'question': 'What color is the truck: white or black?',
     'answer': 'white', 'choices': ['white', 'black'], 'image': 'color-1.jpg', 'cf_image': 'color-1_cf.jpg', 'cf_answer': 'black'},
    {'id': 'color-2', 'category': 'color', 'question': 'What color is the bike: black or purple?',
     'answer': 'black', 'choices': ['black', 'purple'], 'image': 'color-2.jpg', 'cf_image': 'color-2_cf.jpg', 'cf_answer': 'purple'},
    {'id': 'color-3', 'category': 'color', 'question': 'What color is the sign: red or green?',
     'answer': 'red', 'choices': ['red', 'green'], 'image': 'color-3.jpg', 'cf_image': 'color-3_cf.jpg', 'cf_answer': 'green'},
    {'id': 'color-4', 'category': 'color', 'question': 'What color is the table: orange or red?',
     'answer': 'orange', 'choices': ['orange', 'red'], 'image': 'color-4.jpg', 'cf_image': 'color-4_cf.jpg', 'cf_answer': 'red'},
    {'id': 'color-5', 'category': 'color', 'question': "What color are the church's front doors: white or black?",
     'answer': 'white', 'choices': ['white', 'black'], 'image': 'color-5.jpg', 'cf_image': 'color-5_cf.jpg', 'cf_answer': 'black'},
    {'id': 'direction-0', 'category': 'direction', 'question': 'Is the heron facing left or right?',
     'answer': 'right', 'choices': ['left', 'right'], 'image': 'direction-0.jpg', 'cf_image': 'direction-0_cf.jpg', 'cf_answer': 'left'},
    {'id': 'direction-1', 'category': 'direction', 'question': 'Is the airplane facing left or right?',
     'answer': 'left', 'choices': ['left', 'right'], 'image': 'direction-1.jpg', 'cf_image': 'direction-1_cf.jpg', 'cf_answer': 'right'},
    {'id': 'direction-2', 'category': 'direction', 'question': 'Is the car facing left or right?',
     'answer': 'right', 'choices': ['left', 'right'], 'image': 'direction-2.jpg', 'cf_image': 'direction-2_cf.jpg', 'cf_answer': 'left'},
    {'id': 'direction-3', 'category': 'direction', 'question': 'Is the lizard facing left or right?',
     'answer': 'left', 'choices': ['left', 'right'], 'image': 'direction-3.jpg', 'cf_image': 'direction-3_cf.jpg', 'cf_answer': 'right'},
    {'id': 'direction-4', 'category': 'direction', 'question': 'Is the grasshopper facing left or right?',
     'answer': 'left', 'choices': ['left', 'right'], 'image': 'direction-4.jpg', 'cf_image': 'direction-4_cf.jpg', 'cf_answer': 'right'},
    {'id': 'direction-5', 'category': 'direction', 'question': 'Is the van facing to the left?',
     'answer': 'yes', 'choices': ['yes', 'no'], 'image': 'direction-5.jpg', 'cf_image': 'direction-5_cf.jpg', 'cf_answer': 'no'},
    {'id': 'state-0', 'category': 'state', 'question': 'Is the tiger standing or sitting?',
     'answer': 'standing', 'choices': ['standing', 'sitting'], 'image': 'state-0.jpg', 'cf_image': 'state-0_cf.jpg', 'cf_answer': 'sitting'},
    {'id': 'state-1', 'category': 'state', 'question': 'Is the dog sitting or lying down?',
     'answer': 'lying', 'choices': ['sitting', 'lying'], 'image': 'state-1.jpg', 'cf_image': 'state-1_cf.jpg', 'cf_answer': 'sitting'},
    {'id': 'state-2', 'category': 'state', 'question': 'Is the deer standing or lying down?',
     'answer': 'standing', 'choices': ['standing', 'lying'], 'image': 'state-2.jpg', 'cf_image': 'state-2_cf.jpg', 'cf_answer': 'lying'},
    {'id': 'state-3', 'category': 'state', 'question': 'Is the light turned on?',
     'answer': 'no', 'choices': ['yes', 'no'], 'image': 'state-3.jpg', 'cf_image': 'state-3_cf.jpg', 'cf_answer': 'yes'},
    {'id': 'state-4', 'category': 'state', 'question': 'Was this photo taken indoors or outdoors?',
     'answer': 'indoors', 'choices': ['indoors', 'outdoors'], 'image': 'state-4.jpg', 'cf_image': 'state-4_cf.jpg', 'cf_answer': 'outdoors'},
    {'id': 'state-5', 'category': 'state', 'question': "Are the butterfly's wings open or closed?",
     'answer': 'open', 'choices': ['open', 'closed'], 'image': 'state-5.jpg', 'cf_image': 'state-5_cf.jpg', 'cf_answer': 'closed'},
    {'id': 'text-0', 'category': 'text', 'question': 'Does the wall say "NO DUMPING"?',
     'answer': 'yes', 'choices': ['yes', 'no'], 'image': 'text-0.jpg', 'cf_image': 'text-0_cf.jpg', 'cf_answer': 'no'},
    {'id': 'text-1', 'category': 'text', 'question': 'Is there a sign that says "EXIT"?',
     'answer': 'no', 'choices': ['yes', 'no'], 'image': 'text-1.jpg', 'cf_image': 'text-1_cf.jpg', 'cf_answer': 'yes'},
    {'id': 'text-2', 'category': 'text', 'question': 'Is the word "WATER" shown in the image?',
     'answer': 'yes', 'choices': ['yes', 'no'], 'image': 'text-2.jpg', 'cf_image': 'text-2_cf.jpg', 'cf_answer': 'no'},
    {'id': 'text-3', 'category': 'text', 'question': 'Which word is on the sign: restore or winter?',
     'answer': 'restore', 'choices': ['restore', 'winter'], 'image': 'text-3.jpg', 'cf_image': 'text-3_cf.jpg', 'cf_answer': 'winter'},
    {'id': 'text-4', 'category': 'text', 'question': 'Is there a red "EXIT" sign in the image?',
     'answer': 'yes', 'choices': ['yes', 'no'], 'image': 'text-4.jpg', 'cf_image': 'text-4_cf.jpg', 'cf_answer': 'no'},
    {'id': 'text-5', 'category': 'text', 'question': 'Does the manhole cover say "WATER"?',
     'answer': 'no', 'choices': ['yes', 'no'], 'image': 'text-5.jpg', 'cf_image': 'text-5_cf.jpg', 'cf_answer': 'yes'},
]


def add_mismatches(items):
    """Give every item a 'mismatch_image': the photo of another item in the same
    category whose answer is different."""
    for k, item in enumerate(items):
        partner = next(o for o in items[k + 1:] + items[:k]
                       if o["category"] == item["category"] and o["answer"] != item["answer"])
        item["mismatch_image"] = partner["image"]


CATEGORIES = sorted({item["category"] for item in TEST_SET})
'''

NUMWORDS_SRC = '''
NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
'''

IS_CORRECT_SRC = '''
def is_correct(prediction, answer, choices):
    return extract_answer(prediction, choices) == answer
'''

ACC_SRC = '''
def accuracy(results):
    return sum(r["correct"] for r in results) / len(results)


def category_accuracy(results):
    return {c: accuracy([r for r in results if r["category"] == c])
            for c in sorted({r["category"] for r in results})}
'''

PRACTICE_SRC = '''
# Practice models (PROVIDED). Two fake "models" with known behavior, for testing your code.
def _fingerprint(image):
    return hash(image.tobytes())


def make_perfect_model(items):
    """A fake model that always looks: it knows the right answer for every original and
    counterfactual image in `items`, phrases it like a chatty model, and says it cannot
    tell for any image it does not recognize (such as the blank image)."""
    table = {}
    for item in items:
        for name, answer in [(item["image"], item["answer"]), (item["cf_image"], item["cf_answer"])]:
            table[(_fingerprint(load_image(name)), item["question"])] = answer

    def perfect_ask(image, question):
        answer = table.get((_fingerprint(image), question.replace(PROMPT_SUFFIX, "")))
        return f"The answer is {answer.capitalize()}." if answer else "I can't tell."
    return perfect_ask


def guessing_ask(image, question):
    """A fake model that never looks at the image and answers from the question alone."""
    q = question.lower()
    for keyword, guess in [("how many", "Two."), ("color", "Red."), ("left or right", "Left."),
                           ("standing", "Standing."), ("sitting", "Sitting."), ("which word", "Restore.")]:
        if keyword in q:
            return guess
    return "Yes."
'''

# Solutions of the graded functions (also written to reference_solution.py)
SOL_SCORING = '''
def normalize(text):
    """Lowercase, replace punctuation with spaces, map number words to digits, and split into words."""
    text = text.lower()
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    return [NUMBER_WORDS.get(word, word) for word in text.split()]


def extract_answer(prediction, choices):
    """Return the single choice named in the prediction, or None if zero or several are named."""
    words = set(normalize(prediction))
    found = [choice for choice in choices if choice in words]
    return found[0] if len(found) == 1 else None
'''

SOL_RUN = '''
def run_condition(ask_fn, items, condition):
    """Ask every question under one condition; return one result dict per item."""
    image_for = {
        "full": lambda item: load_image(item["image"]),
        "blind": lambda item: blank_image(),
        "mismatched": lambda item: load_image(item["mismatch_image"]),
        "counterfactual": lambda item: load_image(item["cf_image"]),
    }[condition]
    results = []
    for item in items:
        image = image_for(item)
        start = time.perf_counter()
        prediction = ask_fn(image, item["question"] + PROMPT_SUFFIX)
        seconds = time.perf_counter() - start
        target = item["cf_answer"] if condition == "counterfactual" else item["answer"]
        results.append({
            "id": item["id"], "category": item["category"], "prediction": prediction,
            "extracted": extract_answer(prediction, item["choices"]),
            "correct": is_correct(prediction, target, item["choices"]), "seconds": seconds,
        })
    return results
'''

SOL_METRICS = '''
def reliance_gap(full, blind):
    return accuracy(full) - accuracy(blind)


def change_rate(full, mismatched):
    changed = [f["extracted"] != m["extracted"] for f, m in zip(full, mismatched)]
    return sum(changed) / len(changed)


def flip_rate(counterfactual):
    return accuracy(counterfactual)
'''

SOL_EVALUATE = '''
def evaluate_model(ask_fn, items):
    """Run the full modality-reliance protocol on `items` and return a report dictionary."""
    add_mismatches(items)
    results = {c: run_condition(ask_fn, items, c) for c in ["full", "blind", "mismatched", "counterfactual"]}
    return {
        "full_accuracy": accuracy(results["full"]),
        "blind_accuracy": accuracy(results["blind"]),
        "reliance_gap": reliance_gap(results["full"], results["blind"]),
        "change_rate": change_rate(results["full"], results["mismatched"]),
        "flip_rate": flip_rate(results["counterfactual"]),
        "median_latency_s": float(np.median([r["seconds"] for r in results["full"]])),
        "category_accuracy": category_accuracy(results["full"]),
        "results": results,
    }
'''

# ===========================================================================
# Notebook
# ===========================================================================
md(r"""
# Assignment: Evaluate a Multimodal Model

**Unit:** Synthesis & Deployment | **Companion reading:** Evaluation, Benchmarks, and Safety
**Suggested time:** 2 to 2.5 hours

---

## What is this assignment trying to do?

Every model card on Hugging Face lists benchmark scores. In the lesson you saw that a high score does not prove a model actually *looks* at its input. In this assignment you will pick a real, open vision-language model yourself, build a small evaluation harness, test the model with it, and report what you found, the way an ML team would before deciding whether to ship a model.

1. **Part 1:** Find a vision-language model on Hugging Face, record what its model card claims, and write a wrapper so you can ask it questions about images.
2. **Part 2:** Score free-form answers fairly against a provided test set of 30 pairs of real photos.
3. **Part 3:** Run the model under one test condition and measure **accuracy** and **latency**.
4. **Part 4:** Implement the **modality-reliance protocol** from the lesson (full, blind, mismatched, and counterfactual inputs), compute the **reliance gap**, **change rate**, and **flip rate**, and wrap everything into one `evaluate_model` function.
5. **Part 5:** Design your own test category and evaluate your model on it.
6. **Part 6:** Fill in an **evaluation card** that reports your results.

### Where the test photos come from

The test set uses real photos from **NaturalBench**, a research benchmark built to catch models that answer without looking. Every question comes with **two similar photos that have different answers**, such as a red car and an orange car. The second photo is the *counterfactual*: if the model really looks, its answer should change between the two. Real photos can't be edited as cleanly as a drawing, so the two photos in a pair also differ in other ways (lighting, background, angle). Keep that in mind when you write your report.

### How this assignment is graded

Your evaluation functions are **autograded**: the autograder imports your functions and runs them on its **own** test models, so it never needs your model or a GPU. That means:

- Keep every function **name and signature** exactly as given.
- Your evaluation functions must use the model only through the `ask_fn` argument they receive, never through a global model.
- Code outside the graded functions (loading your model, plots, printing) is not run by the autograder, so it does not need to be anything special.

The written reflections and the evaluation card are also checked, and may be read by staff.

### How to work

- Fill in every cell marked `# TODO`. Most parts end with a **check** cell that tells you whether your code is right.
- Write reflections and report fields as Python strings in the cells provided.
- Run every cell top to bottom before submitting.
""")

md(r"""
---
## Part 0: Setup

**Use a GPU.** In Colab: **Runtime → Change runtime type → T4 GPU**. Then run the next cell. Colab already has PyTorch, Pillow, and Matplotlib; uncomment the `pip` line if your model needs a newer `transformers`.
""")

code(r"""
# Colab: uncomment and run once if your model needs a newer transformers, then restart the session
# !pip install -q -U transformers accelerate
""" + IMPORTS_SRC + r"""
import matplotlib.pyplot as plt
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
if DEVICE == "cpu":
    print("No GPU found. Small models still run on a CPU, just slowly; switch to a T4 if you can.")
""")

md(r"""
### Getting the test photos (PROVIDED)

The test photos come in a zip file, `multimodal_eval_images.zip` (about 5 MB). The next cell downloads it and unzips it into a folder called `multimodal_eval_images`. If the download fails, download the zip from the assignment page on the course website, upload it with the **Files** panel on the left side of Colab, and run the cell again. (Colab deletes files when the session ends, so the cell may need to download them again in a new session.)

`load_image(name)` opens a photo by its file name, and `blank_image()` returns a plain gray image. The last cell defines the test set, `TEST_SET`, which Part 2 explains. Just run these cells.
""")

code(LOAD_SRC)

code(r'''
# Download and unzip the test photos (PROVIDED)
import urllib.request
import zipfile

IMAGES_URL = ("https://raw.githubusercontent.com/bwsi-forge-ai-challenge/bwsi-forge-ai-challenge.github.io/"
              "main/Assignments/2_multimodal_%2B_advanced_models/multimodal_eval_images.zip")
if not os.path.isdir(IMAGE_DIR):
    if not os.path.exists("multimodal_eval_images.zip"):
        try:
            urllib.request.urlretrieve(IMAGES_URL, "multimodal_eval_images.zip")
        except Exception as exc:
            raise RuntimeError("Could not download the test photos. Download multimodal_eval_images.zip from the "
                               "assignment page, upload it in Colab's Files panel, and run this cell again.") from exc
    with zipfile.ZipFile("multimodal_eval_images.zip") as z:
        z.extractall(".")
print(len(os.listdir(IMAGE_DIR)), "files in", IMAGE_DIR)
''', r'''
# Download and unzip the test photos (PROVIDED)
import urllib.request
import zipfile

IMAGES_URL = ("https://raw.githubusercontent.com/bwsi-forge-ai-challenge/bwsi-forge-ai-challenge.github.io/"
              "main/Assignments/2_multimodal_%2B_advanced_models/multimodal_eval_images.zip")
if not os.path.isdir(IMAGE_DIR):
    if not os.path.exists("multimodal_eval_images.zip"):
        try:
            urllib.request.urlretrieve(IMAGES_URL, "multimodal_eval_images.zip")
        except Exception as exc:
            raise RuntimeError("Could not download the test photos. Download multimodal_eval_images.zip from the "
                               "assignment page, upload it in Colab's Files panel, and run this cell again.") from exc
    with zipfile.ZipFile("multimodal_eval_images.zip") as z:
        z.extractall(".")
print(len(os.listdir(IMAGE_DIR)), "files in", IMAGE_DIR)

# Solution only: the extra photos used by the sample MY_ITEMS in Part 5
SOLUTION_IMAGES_URL = IMAGES_URL.replace("multimodal_eval_images.zip", "multimodal_eval_solution_images.zip")
if not os.path.exists(os.path.join(IMAGE_DIR, "material-0.jpg")):
    if not os.path.exists("multimodal_eval_solution_images.zip"):
        urllib.request.urlretrieve(SOLUTION_IMAGES_URL, "multimodal_eval_solution_images.zip")
    with zipfile.ZipFile("multimodal_eval_solution_images.zip") as z:
        z.extractall(".")
''')

code(TESTSET_SRC + r"""
missing = [n for i in TEST_SET for n in (i["image"], i["cf_image"]) if not os.path.exists(os.path.join(IMAGE_DIR, n))]
assert not missing, f"missing photos {missing[:3]}: rerun the download cell"
print(len(TEST_SET), "items:", {c: sum(i["category"] == c for i in TEST_SET) for c in CATEGORIES})
""")

# ---------------------------------------------------------------------------
md(r"""
---
## Part 1: Find a Model

Go to the [Hugging Face model hub](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending) and find a model that **takes an image and a text question and answers in text**. On the hub these are tagged **Image-Text-to-Text** or **Visual Question Answering**.

Your model must:

- be open (you can download the weights without paying), and
- fit on a free Colab T4 GPU (16 GB). As a rule of thumb, that means **about 4 billion parameters or fewer** in 16-bit precision.

Some models that fit, if you want a starting point: `HuggingFaceTB/SmolVLM-256M-Instruct`, `HuggingFaceTB/SmolVLM-500M-Instruct`, `HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, `vikhyatk/moondream2`, and `Salesforce/blip-vqa-base`. You may pick any other model that meets the rules. Smaller models run faster but are often less accurate, which can make for an interesting report.

Read your model's **model card** (the page you land on) before you write any code. It tells you how to load the model and usually reports benchmark scores.

### TODO 1: Record what the model card says

Fill in `MODEL_INFO`. If the card does not report a benchmark score, write `"none reported"`.
""")

code(r"""
# TODO 1
MODEL_INFO = {
    "model_id": "",              # e.g. "HuggingFaceTB/SmolVLM-256M-Instruct"
    "url": "",                   # link to the model card
    "parameters": "",            # e.g. "256M" (from the card or the config)
    "developer": "",             # who released it
    "license": "",               # from the card
    "why_this_model": "",        # one or two sentences: why you picked it
    "claimed_benchmark": "",     # one score the card reports, e.g. "VQAv2: 80.2%" or "none reported"
}
""", r"""
MODEL_INFO = {
    "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "url": "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
    "parameters": "256M",
    "developer": "Hugging Face",
    "license": "Apache 2.0",
    "why_this_model": "It is one of the smallest instruction-tuned vision-language models, so it shows how far a tiny model gets and runs quickly on a T4.",
    "claimed_benchmark": "See the benchmark table on the model card (e.g. DocVQA, MMMU).",
}
""")

code(r"""
# Check TODO 1 (PROVIDED)
for key, value in MODEL_INFO.items():
    assert isinstance(value, str) and value.strip(), f"fill in MODEL_INFO['{key}']"
assert "/" in MODEL_INFO["model_id"], "model_id should look like 'organization/model-name'"
print("TODO 1 passed:", MODEL_INFO["model_id"])
""")

md(r"""
### TODO 2: Load the model and write `ask(image, question)`

Every model family is loaded a little differently, so copy the loading and generation code from your model card's **usage** section and adapt it into one function:

```python
ask(image, question) -> str
```

`image` is a PIL image and `question` is a string. The function returns **only the model's answer** as a string, without the prompt or chat template echoed back. This is the function you will pass as `ask_fn` to your evaluation code. (It is not autograded, since the autograder uses its own models.)

Tips:

- On a T4, load weights in half precision: `torch_dtype=torch.float16`.
- Keep answers short with `max_new_tokens=20` (or similar) and turn sampling off (`do_sample=False`) so your results are repeatable.
- Chat-style models generate the prompt tokens plus the answer. Decode only the new tokens, e.g. `output_ids[:, inputs["input_ids"].shape[1]:]`.
- Many recent models also work with `pipeline("image-text-to-text", model=MODEL_INFO["model_id"])`, which hides some of these details. Either approach is fine.

For example, the code for BLIP-VQA (a non-chat model) looks like this. Your code will differ.

```python
from transformers import BlipProcessor, BlipForQuestionAnswering
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=torch.float16).to(DEVICE)

def ask(image, question):
    inputs = processor(image, question, return_tensors="pt").to(DEVICE, torch.float16)
    out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)
```
""")

code(r'''
# TODO 2: load your model here, then complete ask()


def ask(image, question):
    """Ask your model one question about one PIL image. Return only the answer text."""
    raise NotImplementedError("TODO 2: ask")
''', r'''
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_INFO["model_id"])
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_INFO["model_id"],
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()


def ask(image, question):
    """Ask your model one question about one PIL image. Return only the answer text."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
''')

code(r"""
# Check TODO 2 (PROVIDED)
_demo = load_image(TEST_SET[0]["image"])
_demo_question = TEST_SET[0]["question"]
display(_demo.resize((240, 240 * _demo.height // _demo.width)))
_t0 = time.perf_counter()
_answer = ask(_demo, _demo_question + PROMPT_SUFFIX)
print("Question:", _demo_question, " Correct answer:", TEST_SET[0]["answer"])
print(f"Model says: {_answer!r}  ({time.perf_counter() - _t0:.2f} s)")
assert isinstance(_answer, str) and _answer.strip(), "ask() must return a non-empty string"
assert "<" not in _answer and "assistant" not in _answer.lower(), "ask() should return only the answer, not the chat template"
print("TODO 2 passed.")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## Part 2: The Test Set and Fair Scoring

The provided test set, `TEST_SET`, has **30 questions in 5 categories** (6 each): `count`, `color`, `direction` (which way something faces), `state` (standing or sitting, light on or off, and so on), and `text` (reading words in the photo). Every question has exactly two possible answers, and most questions name them, as in "What color is the car: red or orange?". Every item is a dictionary with:

- `id`, `category`, `question`, `answer`, and `choices` (the two valid answers),
- `image`: the file name of the original photo, and
- `cf_image` and `cf_answer`: the **counterfactual** photo, where the answer is different, and its correct answer.

`add_mismatches(items)` gives every item a `mismatch_image`: the photo of a different item in the same category whose answer is different. You will use it in Part 4.

Everyone in the class tests on the same 30 photo pairs. Run the next cell to look at one pair from each category.
""")

code(r"""
# Look at one original and its counterfactual from each category (PROVIDED)
def show_items(items, title=""):
    fig, axes = plt.subplots(2, len(items), figsize=(2.8 * len(items), 5.6))
    axes = np.array(axes).reshape(2, len(items))
    for col, item in enumerate(items):
        for row, (name, ans) in enumerate([(item["image"], item["answer"]), (item["cf_image"], item["cf_answer"])]):
            ax = axes[row, col]
            ax.imshow(load_image(name)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{item['id']}{' (cf)' if row else ''}\nanswer: {ans}", fontsize=9)
    fig.suptitle(title); plt.tight_layout(); plt.show()


show_items([next(i for i in TEST_SET if i["category"] == c) for c in CATEGORIES],
           "Top: original photos. Bottom: counterfactual photos.")
""")

md(r"""
### TODO 3: Score free-form answers

Your model answers in free text. It might say `"Red."`, `"The car is red"`, `"two"`, or `"I think it is red or orange"`. Comparing strings exactly would mark most of these wrong, which is the "many answers can be right" problem from the lesson. Instead, write two functions:

1. `normalize(text)`: lowercase the text, replace every punctuation character with a space, turn the number words `zero` through `ten` into digits (use `NUMBER_WORDS`), and return the list of words. Hint: `string.punctuation`.
2. `extract_answer(prediction, choices)`: normalize the prediction and find which of the item's `choices` appear among its words. If **exactly one** choice appears, return it. If none or more than one appear, return `None`. A hedge like "red or orange" names two choices, so it earns no credit.

`is_correct` is provided and uses your `extract_answer`.
""")

code(NUMWORDS_SRC + r'''

def normalize(text):
    """Lowercase, replace punctuation with spaces, map number words to digits, and split into words."""
    # TODO 3a
    raise NotImplementedError("TODO 3a: normalize")


def extract_answer(prediction, choices):
    """Return the single choice named in the prediction, or None if zero or several are named."""
    # TODO 3b
    raise NotImplementedError("TODO 3b: extract_answer")

''' + IS_CORRECT_SRC, NUMWORDS_SRC + SOL_SCORING + IS_CORRECT_SRC)

code(r"""
# Check TODO 3 (PROVIDED)
assert normalize("The circle is RED.") == ["the", "circle", "is", "red"]
assert normalize("Three squares!") == ["3", "squares"]
assert normalize("yes,it's there") == ["yes", "it", "s", "there"]
_colors = ["red", "blue", "green"]
assert extract_answer("Red.", _colors) == "red"
assert extract_answer("The circle is blue", _colors) == "blue"
assert extract_answer("red or blue", _colors) is None
assert extract_answer("I can't tell", _colors) is None
assert extract_answer("There are three.", [str(k) for k in range(10)]) == "3"
assert extract_answer("redder", _colors) is None, "match whole words only"
assert is_correct("Yes, there is.", "yes", ["yes", "no"])
print("TODO 3 passed.")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## Part 3: Running a Test Condition

### Practice models (PROVIDED)

Before you spend GPU time on your real model, you can test your evaluation code on two fake models whose behavior is known exactly:

- `make_perfect_model(items)` returns a model that **always looks**: it answers every original and counterfactual image in `items` correctly (in chatty phrasing like `"The answer is Red."`) and says `"I can't tell."` for any image it does not recognize, such as the blank image.
- `guessing_ask` **never looks**: it answers from the question alone, for example always `"Red."` for color questions.

They have the same signature as your `ask`, so you can pass either one as `ask_fn`. The autograder uses models like these to test your functions.
""")

code(PRACTICE_SRC)

md(r"""
### TODO 4: `run_condition(ask_fn, items, condition)`

Write a function that asks `ask_fn` every question in `items` under one **condition** and returns a list of result dictionaries, one per item, in the same order as `items`. The condition decides which image the model sees:

| `condition` | Image to use |
|---|---|
| `"full"` | `load_image(item["image"])` |
| `"blind"` | `blank_image()` |
| `"mismatched"` | `load_image(item["mismatch_image"])` |
| `"counterfactual"` | `load_image(item["cf_image"])` |

For each item, call `ask_fn(image, item["question"] + PROMPT_SUFFIX)`, and time **only that call** with `time.perf_counter()`. Each result must have these keys:

```python
{"id": ..., "category": ..., "prediction": <the raw answer>,
 "extracted": extract_answer(prediction, item["choices"]),
 "correct": <bool>, "seconds": <float>}
```

`correct` compares against `item["cf_answer"]` for the counterfactual condition and against `item["answer"]` for every other condition. `accuracy` and `category_accuracy` are provided below your function.
""")

code(r'''
def run_condition(ask_fn, items, condition):
    """Ask every question under one condition; return one result dict per item."""
    # TODO 4
    raise NotImplementedError("TODO 4: run_condition")

''' + ACC_SRC, SOL_RUN + "\n" + ACC_SRC)

code(r"""
# Check TODO 4 on the practice models, then run your model (PROVIDED)
_perfect = make_perfect_model(TEST_SET)
_res = run_condition(_perfect, TEST_SET, "full")
assert len(_res) == len(TEST_SET) and [r["id"] for r in _res] == [i["id"] for i in TEST_SET]
assert all({"id", "category", "prediction", "extracted", "correct", "seconds"} <= set(r) for r in _res)
assert all(r["seconds"] >= 0 for r in _res)
assert accuracy(_res) == 1.0, "the perfect model should get every full question right"
assert accuracy(run_condition(_perfect, TEST_SET, "counterfactual")) == 1.0, "counterfactual should score against cf_answer"
assert accuracy(run_condition(_perfect, TEST_SET, "blind")) == 0.0, "blind should use blank_image()"
assert accuracy(run_condition(guessing_ask, TEST_SET, "full")) == 0.5, "the guesser should get exactly half right"
print("TODO 4 passed. Note that the guesser, which never looks, still gets 50%: every question has two answers.\n")

if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
full = run_condition(ask, TEST_SET, "full")
PEAK_GPU_GB = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else None
print(f"Your model, full accuracy: {accuracy(full):.1%}")
print(f"Median latency per question: {np.median([r['seconds'] for r in full]):.2f} s")
print("Peak GPU memory:", f"{PEAK_GPU_GB:.2f} GB" if PEAK_GPU_GB else "n/a (CPU)")
for c, acc in category_accuracy(full).items():
    print(f"  {c:10s} {acc:.0%}")
print("\nWrong answers:")
for r in full:
    if not r["correct"]:
        item = next(i for i in TEST_SET if i["id"] == r["id"])
        print(f"  {r['id']:12s} expected {item['answer']!r:8s} model said {r['prediction']!r}")
""")

md(r"""
### Part 3 reflection

Which category was your model best at, and which was it worst at? Pick **one wrong answer** from the list above, look at its photo (for example `display(load_image(TEST_SET[0]["image"]))`), and explain what you think went wrong. Was it a perception error, or did the model give an answer the scorer could not read?
""")

code(r'''
part3_reflection = """
TODO: your answer here
"""
''')

# ---------------------------------------------------------------------------
md(r"""
---
## Part 4: Does the Model Actually Look?

Full accuracy alone cannot tell grounded answers from good guesses. The lesson's **modality-reliance protocol** adds three more conditions:

- **Blind:** a plain gray image. A grounded model should do much worse.
- **Mismatched:** the image from a *different* item whose answer is different. A grounded model should change its answer.
- **Counterfactual:** the minimally edited image. A grounded model should switch to the new correct answer.

### TODO 5: The three reliance numbers

Using the definitions from the lesson, write:

1. `reliance_gap(full, blind)`: accuracy on full minus accuracy on blind, as a fraction (e.g. `0.37`).
2. `change_rate(full, mismatched)`: the fraction of items whose **extracted** answer differs between the full and mismatched runs. Compare the `extracted` fields, not the raw text, so `"Red."` and `"red"` count as the same answer.
3. `flip_rate(counterfactual)`: the fraction of counterfactual items the model got right, meaning it switched to the new correct answer.

Each argument is a list of result dictionaries from `run_condition`. The lists are in the same order, so item *i* of one list matches item *i* of another.
""")

code(r'''
def reliance_gap(full, blind):
    # TODO 5a
    raise NotImplementedError("TODO 5a: reliance_gap")


def change_rate(full, mismatched):
    # TODO 5b
    raise NotImplementedError("TODO 5b: change_rate")


def flip_rate(counterfactual):
    # TODO 5c
    raise NotImplementedError("TODO 5c: flip_rate")
''', SOL_METRICS)

code(r"""
# Check TODO 5 (PROVIDED)
_mk = lambda ext, ok: {"extracted": ext, "correct": ok, "category": "x"}
_f = [_mk("red", True), _mk("blue", True), _mk(None, False), _mk("3", True)]
_b = [_mk(None, False), _mk("blue", True), _mk(None, False), _mk(None, False)]
_m = [_mk("green", False), _mk("blue", True), _mk(None, False), _mk("4", False)]
assert abs(reliance_gap(_f, _b) - 0.5) < 1e-9
assert abs(change_rate(_f, _m) - 0.5) < 1e-9
assert abs(flip_rate(_m) - 0.25) < 1e-9
print("TODO 5 passed.")
""")

md(r"""
### TODO 6: `evaluate_model(ask_fn, items)`

Put it all together. `evaluate_model` runs the whole protocol on any model and any list of items:

1. Call `add_mismatches(items)` so every item has a `mismatch_image`.
2. Run all four conditions with `run_condition`.
3. Return a dictionary with exactly these keys:

| Key | Value |
|---|---|
| `"full_accuracy"` | accuracy of the full run |
| `"blind_accuracy"` | accuracy of the blind run |
| `"reliance_gap"`, `"change_rate"`, `"flip_rate"` | your three functions from TODO 5 |
| `"median_latency_s"` | the median `seconds` of the **full** run, as a float (`np.median`) |
| `"category_accuracy"` | `category_accuracy` of the full run |
| `"results"` | a dictionary mapping each condition name (`"full"`, `"blind"`, `"mismatched"`, `"counterfactual"`) to its list of results |
""")

code(r'''
def evaluate_model(ask_fn, items):
    """Run the full modality-reliance protocol on `items` and return a report dictionary."""
    # TODO 6
    raise NotImplementedError("TODO 6: evaluate_model")
''', SOL_EVALUATE)

code(r"""
# Check TODO 6 on the practice models (PROVIDED)
_rep = evaluate_model(make_perfect_model(TEST_SET), TEST_SET)
_keys = {"full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate",
         "median_latency_s", "category_accuracy", "results"}
assert set(_rep) == _keys, f"return exactly the keys {sorted(_keys)}"
assert set(_rep["results"]) == {"full", "blind", "mismatched", "counterfactual"}
assert (_rep["full_accuracy"], _rep["blind_accuracy"], _rep["change_rate"], _rep["flip_rate"]) == (1.0, 0.0, 1.0, 1.0)
_rep = evaluate_model(guessing_ask, TEST_SET)
assert _rep["change_rate"] == 0.0 and _rep["reliance_gap"] == 0.0, "a model that never looks should never change its answer"
print("TODO 6 passed. The perfect model is fully grounded; the guesser has a reliance gap and change rate of 0.\n")

REPORT = evaluate_model(ask, TEST_SET)
for k in ["full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate"]:
    print(f"{k:15s} {REPORT[k]:.1%}")
print(f"median latency  {REPORT['median_latency_s']:.2f} s")

x = np.arange(len(CATEGORIES))
plt.figure(figsize=(7, 3.2))
for k, name in enumerate(["full", "blind", "counterfactual"]):
    accs = category_accuracy(REPORT["results"][name])
    plt.bar(x + (k - 1) * 0.27, [accs[c] for c in CATEGORIES], width=0.27, label=name)
plt.xticks(x, CATEGORIES); plt.ylabel("accuracy"); plt.ylim(0, 1.05); plt.legend(fontsize=8)
plt.title(MODEL_INFO["model_id"]); plt.tight_layout(); plt.show()
""")

md(r"""
### Part 4 reflection

Compare your numbers with **Model A** and **Model B** in the lesson's worked example. Is your model closer to A (scores well but mostly guesses) or B (clearly uses the image)? Point to the specific numbers that convince you. Then look at the **blind** predictions (`[r["prediction"] for r in REPORT["results"]["blind"]]`): what does your model say when there is nothing to see? Does it admit it cannot tell, or does it answer anyway?
""")

code(r'''
part4_reflection = """
TODO: your answer here
"""
''')

# ---------------------------------------------------------------------------
md(r"""
---
## Part 5: Design Your Own Test

The lesson gave three rules for a trustworthy test: questions whose answers **depend on the input**, **small** counterfactual edits that change only the tested detail, and questions and scoring fixed **before** you look at results. Now you will build a small test of your own, with your own photos.

### TODO 7: Add a new category with your own photos

Create a list `MY_ITEMS` with **at least 4 items** in a category that is **not** one of the five above. Each item needs **two photos** that show the same scene with **one thing changed**, so that the correct answer changes. Take them with your phone so the only difference is the detail you are testing: same spot, same angle, same lighting. Some ideas:

- **open or closed:** a laptop, a door, a book, a jar
- **full or empty:** a glass, a mug, a bowl
- **material:** "Is the spoon metal or wood?"
- **shape:** "Is the plate round or square?"
- **on or under:** "Is the pen on the book or under it?"

(Photos you find online also work if their license allows reuse, but a pair you shoot yourself makes a much cleaner counterfactual.) Avoid people's faces and anything personal. Your photos are **not** submitted: only the file names in `MY_ITEMS` are.

**Uploading:** open the **Files** panel on the left side of Colab and drag your photos in. They land in the current folder, so you can refer to each one by its file name, such as `"mug_full.jpg"`. Colab deletes uploaded files when the session ends, so keep a copy on your computer.

Each item needs these keys: `id`, `category`, `question`, `answer`, `choices`, `image`, `cf_image`, and `cf_answer`. Every `id` must be unique, `image` and `cf_image` must be different photos with different answers, and at least two different answers must appear across your items. Like the class test set, name both choices in the question ("...: full or empty?") and keep each choice to **one word**, because `extract_answer` matches single words.

Write `MY_ITEMS` as **one list of dictionaries typed out directly** (no loops), because the autograder reads it straight from your file. Write the items **before** running them on your model.
""")

code(r"""
# TODO 7
MY_ITEMS = []
""", r"""
# Sample answer: four photo pairs from NaturalBench, downloaded with the solution images in Part 0.
# A student would use their own photos here.
MY_ITEMS = [
    {'id': 'material-0', 'category': 'material', 'question': 'What is the wall made of: brick or wood?',
     'answer': 'brick', 'choices': ['brick', 'wood'], 'image': 'material-0.jpg', 'cf_image': 'material-0_cf.jpg', 'cf_answer': 'wood'},
    {'id': 'material-1', 'category': 'material', 'question': 'What are the bench legs made of: metal or wood?',
     'answer': 'wood', 'choices': ['metal', 'wood'], 'image': 'material-1.jpg', 'cf_image': 'material-1_cf.jpg', 'cf_answer': 'metal'},
    {'id': 'material-2', 'category': 'material', 'question': 'What is the star-shaped object made of: metal or stone?',
     'answer': 'metal', 'choices': ['metal', 'stone'], 'image': 'material-2.jpg', 'cf_image': 'material-2_cf.jpg', 'cf_answer': 'stone'},
    {'id': 'material-3', 'category': 'material', 'question': 'Is the wall made of brick or stucco?',
     'answer': 'brick', 'choices': ['brick', 'stucco'], 'image': 'material-3.jpg', 'cf_image': 'material-3_cf.jpg', 'cf_answer': 'stucco'},
]
""")

code(r"""
# Check TODO 7, then evaluate your model on your items (PROVIDED)
_keys = {"id", "category", "question", "answer", "choices", "image", "cf_image", "cf_answer"}
assert len(MY_ITEMS) >= 4, "write at least 4 items"
assert all(_keys <= set(i) for i in MY_ITEMS), f"every item needs the keys {sorted(_keys)}"
assert len({i["id"] for i in MY_ITEMS}) == len(MY_ITEMS), "every id must be unique"
assert len({i["category"] for i in MY_ITEMS}) == 1, "use one category name for all your items"
assert MY_ITEMS[0]["category"] not in CATEGORIES, "pick a new category"
assert all(i["answer"] in i["choices"] and i["cf_answer"] in i["choices"] for i in MY_ITEMS)
assert all(i["answer"] != i["cf_answer"] for i in MY_ITEMS), "each counterfactual must change the answer"
for i in MY_ITEMS:
    for name in (i["image"], i["cf_image"]):
        load_image(name)   # fails here if a photo is missing: upload it in the Files panel
assert all(load_image(i["image"]).tobytes() != load_image(i["cf_image"]).tobytes() for i in MY_ITEMS), \
    "image and cf_image must be different photos"
assert len({i["answer"] for i in MY_ITEMS}) >= 2, "use at least two different answers across your items"
assert evaluate_model(make_perfect_model(MY_ITEMS), MY_ITEMS)["flip_rate"] == 1.0
print("TODO 7 passed.\n")

show_items(MY_ITEMS[:4], f"Your category: {MY_ITEMS[0]['category']}")
MY_REPORT = evaluate_model(ask, MY_ITEMS)
for k in ["full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate"]:
    print(f"{k:15s} {MY_REPORT[k]:.1%}")
""")

# ---------------------------------------------------------------------------
md(r"""
---
## Part 6: Report Your Evaluation

Teams that publish a model write an **evaluation card**: what was tested, the results, and what the results do and do not show. Fill in the written fields of `EVAL_REPORT` as plain strings. The numbers are filled in for you from your runs. Keep each field to two or three sentences.

### TODO 8: Evaluation card
""")

code(r"""
# TODO 8: replace every "TODO" with your own writing (plain strings)
_NUMBERS = ["full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate", "median_latency_s"]
EVAL_REPORT = {
    # Filled in automatically
    "model": MODEL_INFO["model_id"],
    "parameters": MODEL_INFO["parameters"],
    "results": {k: round(REPORT[k], 3) for k in _NUMBERS},
    "category_accuracy": {k: round(v, 3) for k, v in REPORT["category_accuracy"].items()},
    "my_category": {MY_ITEMS[0]["category"]: {k: round(MY_REPORT[k], 3) for k in _NUMBERS}},
    "peak_gpu_gb": None if PEAK_GPU_GB is None else round(PEAK_GPU_GB, 2),

    # Written by you
    "summary": "TODO",                    # What is the model good and bad at, in one or two sentences?
    "grounding_verdict": "TODO",          # Does it actually use the image? Cite the reliance numbers.
    "claimed_vs_measured": "TODO",        # How does the model card's claim compare with what you measured, and why might they differ?
    "limitations_of_this_test": "TODO",   # What can this test NOT tell you? (Think: 30 photo pairs, two answer choices, one prompt wording.)
    "deployment_recommendation": "TODO",  # Would you use this model in an app that describes scenes to blind users? Why or why not?
}
""")

code(r"""
# Check TODO 8 and print your evaluation card (PROVIDED)
_written = ["summary", "grounding_verdict", "claimed_vs_measured", "limitations_of_this_test", "deployment_recommendation"]
for key in _written:
    assert isinstance(EVAL_REPORT[key], str) and EVAL_REPORT[key].strip() not in ("", "TODO"), f"fill in EVAL_REPORT['{key}']"
print("TODO 8 passed.\n")

r = EVAL_REPORT["results"]
print(f"EVALUATION CARD: {EVAL_REPORT['model']} ({EVAL_REPORT['parameters']} parameters)")
print("=" * 72)
print(f"Full accuracy {r['full_accuracy']:.0%} | blind {r['blind_accuracy']:.0%} | reliance gap {r['reliance_gap']:+.0%}"
      f" | change rate {r['change_rate']:.0%} | flip rate {r['flip_rate']:.0%}")
print("By category:", ", ".join(f"{c} {a:.0%}" for c, a in EVAL_REPORT["category_accuracy"].items()))
(cat, mr), = EVAL_REPORT["my_category"].items()
print(f"My category ({cat}): full {mr['full_accuracy']:.0%}, flip rate {mr['flip_rate']:.0%}")
print(f"Median latency {r['median_latency_s']} s per question; peak GPU memory {EVAL_REPORT['peak_gpu_gb']} GB")
print("Model card claims:", MODEL_INFO["claimed_benchmark"])
for key in _written:
    print(f"\n{key.replace('_', ' ').upper()}\n  {EVAL_REPORT[key]}")
""")

md(r"""
---
## Submitting

1. Make sure every `TODO` is complete, every check cell prints a passing message, and both reflections (`part3_reflection`, `part4_reflection`) are written.
2. Run all cells from top to bottom (**Runtime → Run all** in Colab).
3. Download the notebook as a Python file (**File → Download → Download .py**) and rename it `firstname_lastname_model_eval.py`.
4. Submit that `.py` file on Gradescope (see the assignment page on the course website for the link).
""")

# ===========================================================================
SOLUTION_TEXT = {
    'part3_reflection = """\nTODO: your answer here\n"""':
        'part3_reflection = """\nSample answer (your numbers will differ): the model was best at color and text and '
        'worst at direction. It answered "right" for the airplane in direction-1, which faces left. The plane is small '
        'and seen from below, and small vision-language models are known to be weak at left/right orientation, so this '
        'is a perception error, not a scoring problem: the scorer read its one-word answer correctly. A few state '
        'answers, such as "laying" instead of "lying", were scoring problems instead.\n"""',
    'part4_reflection = """\nTODO: your answer here\n"""':
        'part4_reflection = """\nSample answer: the model is closer to Model B. Its blind accuracy is near 50 percent, '
        'which is what guessing between two choices gives, so the reliance gap is large, and its change and flip rates '
        'are well above zero, which means its answers really follow the photo. On the blank image, however, it never '
        'says it cannot tell: it still answers "red" or "yes", which is the guessing behavior of Model A showing up '
        'exactly when there is no evidence.\n"""',
    '"summary": "TODO",':
        '"summary": "Sample: reliable on color and on reading large words in photos, but weak at telling which way an animal or vehicle is facing and at counting small objects.",',
    '"grounding_verdict": "TODO",':
        '"grounding_verdict": "Sample: it uses the image. The reliance gap and the flip rate are both large, although it never admits that a blank image is empty.",',
    '"claimed_vs_measured": "TODO",':
        '"claimed_vs_measured": "Sample: the card reports scores on large public benchmarks such as DocVQA. Our 30 photo pairs were chosen so that guessing fails, and each question has only two answers, so the two numbers measure different things and are not directly comparable.",',
    '"limitations_of_this_test": "TODO",':
        '"limitations_of_this_test": "Sample: only 30 photo pairs with one prompt wording and one run each, so one question is worth more than 3 points of accuracy. The two photos in a pair also differ in lighting and background, so a flip may not come from the detail we meant to test.",',
    '"deployment_recommendation": "TODO",':
        '"deployment_recommendation": "Sample: not on its own. It answers confidently when there is nothing to see, which is dangerous for blind users, so it would need a way to abstain and a much larger test on the kinds of photos those users take.",',
    '# TODO 8: replace every "TODO" with your own writing (plain strings)': '# TODO 8 (sample answers)',
}


def to_nb(solution):
    nb_cells = []
    for kind, src, sol in cells:
        text = sol if (solution and sol is not None) else src
        if solution:
            for old, new in SOLUTION_TEXT.items():
                text = text.replace(old, new)
        lines = text.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {"cells": nb_cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5}


for solution, name in [(False, "multimodal_evaluation_assignment.ipynb"),
                       (True, "multimodal_evaluation_assignment_solution.ipynb")]:
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(to_nb(solution), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", name)

GENERATED = "# GENERATED by build_assignment.py from the assignment notebook. Do not edit by hand.\n"
os.makedirs(AG_DIR, exist_ok=True)
with open(os.path.join(AG_DIR, "provided.py"), "w") as f:
    f.write(GENERATED + '"""The PROVIDED code from the handout, verbatim."""\n' + IMPORTS_SRC + LOAD_SRC + TESTSET_SRC
            + NUMWORDS_SRC + IS_CORRECT_SRC + ACC_SRC + PRACTICE_SRC)
with open(os.path.join(AG_DIR, "reference_solution.py"), "w") as f:
    f.write(GENERATED + '"""Reference versions of the graded functions. Exec\'d on top of provided.py."""\n'
            + SOL_SCORING + SOL_RUN + SOL_METRICS + SOL_EVALUATE)
print("wrote provided.py and reference_solution.py")
