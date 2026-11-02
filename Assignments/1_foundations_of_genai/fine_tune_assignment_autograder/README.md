# Autograder: `fine_tune_assignment`

Grades student uploads of **`firstname_lastname_fine_tune.py`** (exported from the notebook, see Part 5 of the assignment) plus one W&B plot image.

The grader is **static**: it parses the `.py` with `ast` and evaluates only safe pieces
(constants, the two prompt-format functions, the reflections). It never downloads `distilgpt2`
or the Yoda dataset, logs in to W&B, or trains. Commented-out code is invisible to the
AST, so "uncomment the training cell" and "implement the training function" are verified.
No third-party dependencies.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 1_foundations_of_genai/fine_tune_assignment_autograder
```

## Point breakdown (20)

Dataset: `dvgodoy/yoda_sentences` (720 rows, `sentence` -> `translation_extra`), loaded in the handout.

| Section | Points |
| --- | ---: |
| Dataset: `load_dataset("dvgodoy/yoda_sentences")` present, both prompt-format functions correct | 3 |
| Training function (TODO 2): `TrainingArguments` (output_dir, epochs, batch size, learning rate) 2; `report_to="wandb"` 1; `Trainer` with model/args/train_dataset/data_collator 1; `trainer.train()` and `save_model` 1 | 5 |
| Hyperparameters (TODO 1): two different learning rates, epochs and batch size set | 5 |
| Training runs (TODO 3): two active `train_finetune_run` calls 1 (given in the handout, so this guards against deleting one); different learning rate and output dir 1; a plot image (.png/.jpg/.jpeg/.gif/.webp, at least 1 KB) is uploaded next to the `.py` 1 | 3 |
| Reflections (`part1_reflection` >= 20 words, `part4_reflection` >= 40 words, keyword checks) | 4 |

Reflections get only a keyword/length sanity check; a manual read is still recommended.
The plot check is existence only; it does not look at the image, so spot-check a few by hand. The training runs themselves and the Part 4 evaluation are not graded (the unused check remains in `run_tests.py`).

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Gradescope image setup (only needs `python3`) |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Static checks → `results.json` |
| `requirements.txt` | Empty (stdlib only) |

If the dataset id changes, update `DATASET_ID` in `run_tests.py`.

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 1_foundations_of_genai/fine_tune_assignment_autograder --solution   # 20/20
./run_local_autograder.sh 1_foundations_of_genai/fine_tune_assignment_autograder /path/to/student.py
```

An untouched handout scores 4/20: the dataset section is provided, and the two `train_finetune_run` calls are already active in the handout.
