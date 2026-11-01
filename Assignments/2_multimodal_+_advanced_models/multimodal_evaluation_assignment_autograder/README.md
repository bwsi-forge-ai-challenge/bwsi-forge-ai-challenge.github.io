# Autograder: `multimodal_evaluation_assignment`

Grades student uploads of **`firstname_lastname_model_eval.py`** (exported from the notebook, see the Submitting section of the assignment).

The grader parses the `.py` with `ast` and never runs the whole file, so it never downloads the
student's vision-language model or needs a GPU. Instead it execs each graded evaluation function
on its own, on top of the PROVIDED handout code and reference versions of the other graded
functions, and runs it on the grader's **own local models**:

| Model | Behavior |
| --- | --- |
| perfect | Always looks: answers every original and counterfactual image correctly, says "I can't tell." on unknown images |
| guessing | Never looks: answers from the question alone |
| partial | Looks only for color and word questions, guesses otherwise |

Students test on 30 pairs of real photos (`../multimodal_eval_images.zip`), but the grader never
needs them: `run_tests.py` draws two hidden test sets of simple shapes in the same item format,
saves them as image files, and points the provided `load_image()` at that folder. The student's
outputs are compared with the reference implementation. A bug in one function only costs points
in its own section. `MY_ITEMS` refers to the student's own photos, which are not submitted, so
only the item definitions are checked.

Dependencies: `numpy` and `pillow` (to draw the grader's test images).

## The test photos

`make_test_images.py` builds `multimodal_eval_images.zip` (the class test set, with a
`CREDITS.txt`) and `multimodal_eval_solution_images.zip` (four extra pairs for the solution's
`MY_ITEMS`) from NaturalBench (Apache 2.0; photos from DOCCI, CC BY 4.0). Every pair it uses was
checked by eye. It needs the three NaturalBench parquet shards (about 1.1 GB):

```bash
python3 make_test_images.py PATH/TO/NaturalBench/data
```

The notebook downloads the zip from raw.githubusercontent.com on the `main` branch, so the zip
must be pushed before students start.

## Regenerating `provided.py` and `reference_solution.py`

Both files are generated, together with the handout and solution notebooks, by
`build_assignment.py` in this folder, so the grader always runs exactly the code students were
given. Do not edit the notebooks or the generated files by hand: edit `build_assignment.py` and
regenerate all four files, then re-run the smoke test below.

```bash
python3 build_assignment.py
```

## Point breakdown (20)

| Section | Points |
| --- | ---: |
| Answer scoring: `normalize`, `extract_answer` | 4 |
| `run_condition`: all four conditions, prompt suffix, timing only the model call | 4 |
| Reliance metrics: `reliance_gap`, `change_rate`, `flip_rate` | 3 |
| `evaluate_model`: report structure, reliance numbers, category accuracy, latency | 3 |
| Own test category: `MY_ITEMS` well-formed, valid counterfactuals | 2 |
| `MODEL_INFO` and the written fields of `EVAL_REPORT` | 2 |
| Reflections: `part3_reflection`, `part4_reflection` (length + key content) | 2 |

The written fields and reflections get only a length/keyword sanity check; a manual read is
still recommended. Students see section scores only; set `AUTOGRADE_DEBUG=1` locally to see
which sub-checks failed.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 2_multimodal_+_advanced_models/multimodal_evaluation_assignment_autograder
```

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Install deps when Gradescope builds the image |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Checks → `results.json` |
| `provided.py` | Generated: the handout's PROVIDED code (photo loading, test set, practice models) |
| `reference_solution.py` | Generated: reference graded functions |
| `requirements.txt` | `numpy`, `pillow` |
| `build_assignment.py` | Generates the two notebooks, `provided.py` and `reference_solution.py` (not used at grading time) |
| `make_test_images.py` | Builds the photo zips from NaturalBench (not used at grading time) |

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 2_multimodal_+_advanced_models/multimodal_evaluation_assignment_autograder --solution   # 20/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/multimodal_evaluation_assignment_autograder --student    # 0/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/multimodal_evaluation_assignment_autograder /path/to/student.py
```
