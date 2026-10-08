# Autograder: `fine_tune_assignment`

Grades student uploads of **`firstname_lastname_fine_tune.py`** (exported from the notebook, see Part 5 of the assignment).

The grader is **static**: it parses the `.py` with `ast` and evaluates only safe pieces
(constants, `TRAINING_PAIRS`, the two prompt-format functions, the reflections). It never
downloads `distilgpt2`, logs in to W&B, or trains. Commented-out code is invisible to the
AST, so "uncomment the training cell" is verified. No third-party dependencies.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 1_foundations_of_genai/fine_tune_assignment_autograder
```

## Point breakdown (100)

| Part | Points |
| --- | ---: |
| dataset (>= 50 pairs, >= 5 new, no dupes, well-formed) + prompt format functions | 30 |
| hyperparameters set (two different learning rates, epochs, batch size) | 15 |
| `train_finetune_run` intact, W&B login active, two distinct training runs | 30 |
| Part 4 — fine-tuned model prompted with `format_inference_prompt` | 5 |
| Reflections — `part1_reflection` / `part4_reflection` length + key content | 20 |

Reflections get only a keyword/length sanity check; a manual read is still recommended.

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Gradescope image setup (only needs `python3`) |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Static checks → `results.json` |
| `starter_pairs.json` | The 45 provided pairs, used to count *new* student examples |
| `requirements.txt` | Empty (stdlib only) |

If the starter pairs in the handout change, regenerate `starter_pairs.json`.

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 1_foundations_of_genai/fine_tune_assignment_autograder --solution   # 100/100
./run_local_autograder.sh 1_foundations_of_genai/fine_tune_assignment_autograder /path/to/student.py
```

An untouched handout scores ~28/100 (credit for the provided scaffolding).
