# Autograder: `rnn_assignment`

Grades student uploads of `rnn_assignment.ipynb`.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 0_prerequisites/rnn_assignment_autograder
```

## Point breakdown (100)

| Part | Points |
| --- | ---: |
| TODO 1 — `encode` & `CharDataset` | 30 |
| TODO 2 — `CharRNN` | 35 |
| TODO 3 — `train_one_epoch` & `evaluate` | 30 |
| Reflection markdown present (length check) | 5 |

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Install deps when Gradescope builds the image |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Notebook harness + TODO tests → `results.json` |
| `requirements.txt` | `torch`, `nbformat`, `numpy` |
| `data/tinyshakespeare.txt` | Offline corpus (no network at grade time) |

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 0_prerequisites/rnn_assignment_autograder --solution
./run_local_autograder.sh 0_prerequisites/rnn_assignment_autograder --student
```

Solution should score ~95/100 if reflections are blank; blank handout should score low.
