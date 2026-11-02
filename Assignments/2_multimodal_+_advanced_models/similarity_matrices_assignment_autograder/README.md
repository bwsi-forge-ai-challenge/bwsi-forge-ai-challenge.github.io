# Autograder: `similarity_matrices_assignment`

Grades student uploads of **`firstname_lastname_similarity.py`** (exported from the notebook, see the Submitting section of the assignment).

The whole assignment runs on CPU in seconds, so the grader works in two ways:

* **Graded functions** (`cosine_similarity`, `l2_normalize`, `similarity_matrix`,
  `retrieval_accuracy`, `info_nce_loss`) are exec'd one at a time on top of the reference
  versions of the others (`reference_solution.py`) and compared with the reference on the
  grader's own random inputs, plus a few property checks (length does not matter, rectangular
  matrices, normalization, the loss stays differentiable).
* **Cell-level TODOs** (random projection heads, training, the temperature sweep) are checked by
  re-running the submission one top-level statement at a time, with the graded functions swapped
  for the reference versions. A statement that raises is skipped, like a notebook cell that
  errors. The variables left behind (`W_img`, `W_txt`, `S_random`, `S_trained`, `results`) are
  compared with values recomputed in the same run.

Either way, a bug in one function only costs points in its own section.

Dependencies: `numpy`, `matplotlib` (the notebook's plotting cells run headless), and the
CPU-only build of `torch`, which `setup.sh` installs from the PyTorch index.

If you change the solution notebook's graded functions, copy them into `reference_solution.py`.

## Point breakdown (20)

| Section | Points |
| --- | ---: |
| Part 1: `cosine_similarity` | 2 |
| Part 2: `l2_normalize`, `similarity_matrix` | 3 |
| Part 2: `retrieval_accuracy` | 2 |
| Part 3: random projection heads (`W_img`, `W_txt`, `S_random`) | 2 |
| Part 4: `info_nce_loss` | 4 |
| Part 4: training the heads (`S_trained`) | 2 |
| Part 5: temperature sweep (`results`) | 2 |
| Reflections: `part2_reflection`, `part4_reflection`, `part5_reflection` (length + key content) | 3 |

The reflections get only a length/keyword sanity check; a manual read is still recommended.
Students see section scores only; set `AUTOGRADE_DEBUG=1` locally to see which sub-checks failed.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 2_multimodal_+_advanced_models/similarity_matrices_assignment_autograder
```

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Install deps when Gradescope builds the image |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Checks → `results.json` |
| `reference_solution.py` | Reference graded functions (copied from the solution notebook) |
| `requirements.txt` | `numpy`, `matplotlib` (torch is installed separately by `setup.sh`) |

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 2_multimodal_+_advanced_models/similarity_matrices_assignment_autograder --solution   # 20/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/similarity_matrices_assignment_autograder --student    # 0/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/similarity_matrices_assignment_autograder /path/to/student.py
```
