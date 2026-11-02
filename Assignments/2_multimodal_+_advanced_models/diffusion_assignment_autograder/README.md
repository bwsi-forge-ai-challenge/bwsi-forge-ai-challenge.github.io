# Autograder: `diffusion_assignment`

Grades student uploads of **`firstname_lastname_diffusion.py`** (exported from the notebook, see the Submitting section of the assignment).

The grader parses the `.py` with `ast` and never runs the whole file, so it never trains the
student's model, downloads SD-Turbo, or needs a GPU. Instead it execs each graded function on
its own, on top of the PROVIDED handout code and reference versions of the other graded
functions, and runs it on the grader's **own tiny models** on a CPU:

| Model | Used for |
| --- | --- |
| perfect | An "oracle" that knows the clean pictures and returns the exact noise. A correct loss gives ~0 for it, and a correct sampler recovers the target from any number of steps |
| tanh | An imperfect model that depends on the picture and the step; results are compared with the reference sampler |
| blurred | An oracle for a blurred target; results are compared with the reference sampler |
| conv | A one-layer network, to check the loss against the reference and that gradients flow |

A bug in one function only costs points in its own section (the other graded functions run
the reference versions). Student prompts, the favorite picture's settings, and the written
reflections are checked for form, length, and a few key words only; a manual read is still
recommended.

Dependencies: `numpy`, `pillow`, and the CPU-only build of `torch` (installed by `setup.sh`).

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
| `make_alpha_bar`, `add_noise` | 2 |
| `predict_clean` | 1 |
| `noise_prediction_loss`: values (including a perfect model scoring ~0), shape and gradients | 1.5 |
| `ddim_sample`: recovers a target, matches the reference and uses the right step numbers, clamps | 2.5 |
| `slerp` | 1 |
| `MY_PROMPTS` and `FAVORITE` well-formed | 2 |
| Reflections: `part1_reflection` to `part7_reflection` (length + key words) | 10 |

Students see section scores only; set `AUTOGRADE_DEBUG=1` locally to see which sub-checks
failed.

Zip from the `Assignments/` folder:

```bash
./make_autograder_zip.sh 2_multimodal_+_advanced_models/diffusion_assignment_autograder
```

## Package contents

| File | Role |
| --- | --- |
| `setup.sh` | Install deps when Gradescope builds the image (CPU-only PyTorch) |
| `run_autograder` | Gradescope entrypoint |
| `run_tests.py` | Checks → `results.json` |
| `provided.py` | Generated: the handout's PROVIDED code the grader needs (imports, toy-picture helpers, `TinyUNet`) |
| `reference_solution.py` | Generated: reference graded functions |
| `requirements.txt` | `numpy`, `pillow` |
| `build_assignment.py` | Generates the two notebooks, `provided.py` and `reference_solution.py` (not used at grading time) |

## Local smoke test

```bash
# from Assignments/
./run_local_autograder.sh 2_multimodal_+_advanced_models/diffusion_assignment_autograder --solution   # 20/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/diffusion_assignment_autograder --student    # 0/20
./run_local_autograder.sh 2_multimodal_+_advanced_models/diffusion_assignment_autograder /path/to/student.py
```

## Not tested by the grader

The notebook's GPU parts (training the tiny model, SD-Turbo) are not graded. Before the
assignment goes live, run the solution notebook once on a free Colab T4 and check that
training finishes in a few minutes, that the samples look like clothing, and that `generate`
returns a 512 x 512 picture.
