#!/usr/bin/env python3
"""
Gradescope autograder for similarity_matrices_assignment (submitted as a .py file)

Package lives at:
  Assignments/2_multimodal_+_advanced_models/similarity_matrices_assignment_autograder/

How it works
  * Graded functions are checked one at a time: the student's version is exec'd on top of
    the reference versions of the other graded functions (reference_solution.py) and
    compared with the reference on the grader's own random inputs. A bug in one function
    therefore costs points only in its own section.
  * The cell-level TODOs (random projection heads, training, the temperature sweep) are
    checked by re-running the whole submission on CPU, one top-level statement at a time,
    with the graded functions swapped for the reference versions. A statement that raises
    is skipped, like a notebook cell that errors. The variables it leaves behind
    (W_img, W_txt, S_random, S_trained, results) are compared with recomputed values.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import signal
import warnings
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import numpy as np
import torch

SOURCE_DIR = Path(__file__).resolve().parent
REFERENCE_SRC = (SOURCE_DIR / "reference_solution.py").read_text(encoding="utf-8")

LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [Path("/autograder/submission"), SOURCE_DIR.parent]

GRADED_FUNCS = {"cosine_similarity", "l2_normalize", "similarity_matrix", "retrieval_accuracy",
                "info_nce_loss"}
REFLECTIONS = ["part2_reflection", "part4_reflection", "part5_reflection"]
TAUS = [0.01, 0.07, 0.2, 0.5, 1.0, 5.0]
CHECK_TIMEOUT_S = 120
RUN_TIMEOUT_S = 300


# --------------------------------------------------------------------------- #
# Loading the submission
# --------------------------------------------------------------------------- #
def find_submission(submission_dir: Path) -> Path:
    for var in ("AUTOGRADE_FILE", "AUTOGRADE_NOTEBOOK"):
        env = os.environ.get(var)
        if env:
            p = Path(env)
            if not p.exists():
                raise FileNotFoundError(f"{var} not found: {p}")
            return p

    pys = sorted(
        p
        for p in submission_dir.rglob("*.py")
        if "__pycache__" not in p.parts and not any("_autograder" in part for part in p.parts)
    )
    if not pys:
        has_nb = any(submission_dir.rglob("*.ipynb"))
        hint = (
            " You uploaded a .ipynb -- export it to a .py first (see the Submitting section)."
            if has_nb
            else ""
        )
        raise FileNotFoundError(
            "No .py file found in your submission. Submit firstname_lastname_similarity.py." + hint
        )
    for p in pys:
        if p.name.endswith("_similarity.py"):
            return p
    return pys[0]


def sanitize(source: str) -> str:
    """Neutralise Jupyter/Colab export artefacts so ast.parse works."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")) or "get_ipython()" in stripped:
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + "pass")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def read_source(path: Path) -> str:
    """A .py submission, or (for local testing) the code cells of a notebook."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".ipynb":
        nb = json.loads(text)
        text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return sanitize(text)


def reference_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "graded"}
    exec(compile(REFERENCE_SRC, "reference_solution.py", "exec"), ns)
    return ns


class Submission:
    def __init__(self, path: Path):
        self.path = path
        self.source = read_source(path)
        self.tree = ast.parse(self.source, filename=path.name)
        self.defs: dict[str, ast.stmt] = {}
        self.helpers: list[ast.stmt] = []
        self.assigns: dict[str, ast.stmt] = {}
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.helpers.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in GRADED_FUNCS:
                    self.defs[node.name] = node            # last definition wins, like Python
                else:
                    self.helpers.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in REFLECTIONS:
                    self.assigns[node.targets[0].id] = node
        self._run_ns: dict[str, Any] | None = None

    def _exec(self, nodes: list[ast.stmt], ns: dict[str, Any]) -> None:
        for node in nodes:
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
            except Exception:
                pass  # a broken statement only matters if something graded needs it

    def namespace(self, *student_funcs: str) -> dict[str, Any]:
        """Reference functions, the student's imports and helpers, and their versions of `student_funcs`."""
        ns = reference_namespace()
        with contextlib.redirect_stdout(io.StringIO()):
            self._exec(self.helpers, ns)
        for name in GRADED_FUNCS - set(student_funcs):     # a helper must not shadow a reference
            ns[name] = reference_namespace()[name]
        for name in student_funcs:
            node = self.defs.get(name)
            if node is None:
                raise AssertionError(f"`{name}` is not defined at the top level of your file.")
            exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
        return ns

    def value(self, name: str) -> Any:
        node = self.assigns.get(name)
        if node is None:
            raise AssertionError(f"`{name}` is not assigned at the top level of your file.")
        ns: dict[str, Any] = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
        return ns[name]

    def run(self) -> dict[str, Any]:
        """The namespace left by running the whole file with the reference graded functions (cached)."""
        if self._run_ns is None:
            import matplotlib.pyplot as plt
            ns = reference_namespace()
            body = [n for n in self.tree.body
                    if not (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in GRADED_FUNCS)]
            old_show = plt.show
            plt.show = lambda *a, **k: plt.close("all")
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self._exec(body, ns)
            finally:
                plt.show = old_show
                plt.close("all")
            self._run_ns = ns
        return self._run_ns


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
GRADER_RNG_SEED = 1234


def _rng() -> np.random.Generator:
    return np.random.default_rng(GRADER_RNG_SEED)


def _number(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        assert x.numel() == 1, "should return a single number"
        x = x.item()
    if isinstance(x, np.ndarray):
        assert x.size == 1, "should return a single number"
        x = x.item()
    assert isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool), (
        f"should return a number, got {type(x).__name__}")
    return float(x)


def _array(x: Any, shape: tuple) -> np.ndarray:
    assert x is not None, "returned None"
    a = np.asarray(x.detach().numpy() if isinstance(x, torch.Tensor) else x, dtype=float)
    assert a.shape == shape, f"expected shape {shape}, got {a.shape}"
    return a


def word_count(s: Any) -> int:
    return len(re.findall(r"\w+", s)) if isinstance(s, str) else 0


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
Check = Callable[[Submission], None]
REF = reference_namespace()


# ---- Part 1: cosine similarity ------------------------------------------- #
def t_cosine_values(sub):
    f = sub.namespace("cosine_similarity")["cosine_similarity"]
    rng = _rng()
    for dim in (2, 3, 5, 12):
        u, v = rng.normal(size=dim), rng.normal(size=dim)
        assert abs(_number(f(u, v)) - REF["cosine_similarity"](u, v)) < 1e-9, "wrong value on random vectors"


def t_cosine_properties(sub):
    f = sub.namespace("cosine_similarity")["cosine_similarity"]
    u = np.array([1.0, -2.0, 0.5, 3.0])
    assert abs(_number(f(u, 7.5 * u)) - 1.0) < 1e-9, "length should not matter"
    assert abs(_number(f(u, -0.2 * u)) + 1.0) < 1e-9, "opposite vectors should give -1"
    assert abs(_number(f(np.array([1.0, 0.0, 0.0]), np.array([0.0, 4.0, 0.0])))) < 1e-9, (
        "perpendicular vectors should give 0")
    assert abs(_number(f(np.array([1.0, 2.0]), np.array([2.0, 1.0]))) - 0.8) < 1e-9


# ---- Part 2: similarity matrix ------------------------------------------- #
def t_l2_normalize(sub):
    f = sub.namespace("l2_normalize")["l2_normalize"]
    X = _rng().normal(size=(6, 5)) * 3
    X_before = X.copy()
    out = _array(f(X), (6, 5))
    assert np.allclose(out, REF["l2_normalize"](X_before)), "rows should be divided by their own length"
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0), "every row should have length 1"
    assert np.array_equal(X, X_before), "l2_normalize should not change its input"


def t_similarity_square(sub):
    f = sub.namespace("similarity_matrix")["similarity_matrix"]
    rng = _rng()
    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    assert np.allclose(_array(f(A, B), (4, 4)), REF["similarity_matrix"](A, B)), "wrong values"
    assert np.allclose(_array(f(A, A), (4, 4)).diagonal(), 1.0), "a vector's similarity with itself is 1"


def t_similarity_rect(sub):
    f = sub.namespace("similarity_matrix")["similarity_matrix"]
    rng = _rng()
    A, B = rng.normal(size=(5, 3)) * 4, rng.normal(size=(9, 3))
    assert np.allclose(_array(f(A, B), (5, 9)), REF["similarity_matrix"](A, B)), (
        "S[i, j] should compare row i of A with row j of B, for any number of rows")


# ---- Part 2: retrieval accuracy ------------------------------------------ #
def t_retrieval_values(sub):
    f = sub.namespace("retrieval_accuracy")["retrieval_accuracy"]
    rng = _rng()
    for n in (3, 8, 20):
        S = rng.normal(size=(n, n)) + 1.5 * np.eye(n)
        assert abs(_number(f(S)) - REF["retrieval_accuracy"](S)) < 1e-12, "wrong value on a random matrix"


def t_retrieval_direction(sub):
    f = sub.namespace("retrieval_accuracy")["retrieval_accuracy"]
    S = np.array([[0.9, 0.8, 0.0], [0.95, 0.1, 0.0], [0.0, 0.2, 0.7]])
    assert abs(_number(f(S)) - 2 / 3) < 1e-12, "use each row's best column"
    assert abs(_number(f(S.T)) - 1 / 3) < 1e-12, "use each row's best column, not each column's"
    assert _number(f(np.eye(4)[::-1].copy())) == 0.0


# ---- Part 3: random projection heads ------------------------------------- #
def t_random_shapes(sub):
    ns = sub.run()
    for name, rows in (("W_img", "IMG_DIM"), ("W_txt", "TXT_DIM")):
        assert name in ns, f"`{name}` is not defined"
        _array(ns[name], (ns[rows], ns["D"]))


def t_random_similarity(sub):
    ns = sub.run()
    S = _array(ns.get("S_random"), (ns["N_TEST"], ns["N_TEST"]))
    if "W_img" in ns and "W_txt" in ns:
        expected = REF["similarity_matrix"](ns["img_test"] @ np.asarray(ns["W_img"]),
                                            ns["txt_test"] @ np.asarray(ns["W_txt"]))
        assert np.allclose(S, expected), "S_random should compare the projected TEST features"
    assert np.all(np.abs(S) <= 1 + 1e-9), "S_random should hold cosine similarities"
    assert REF["retrieval_accuracy"](S) < 0.2, "random heads should not line the spaces up"


# ---- Part 4: InfoNCE loss ------------------------------------------------- #
def _loss_inputs(n=6, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


def t_infonce_value(sub):
    f = sub.namespace("info_nce_loss")["info_nce_loss"]
    a, b = _loss_inputs()
    got = _number(f(a, b, 0.07))
    assert abs(got - REF["info_nce_loss"](a, b, 0.07).item()) < 1e-4, "wrong loss at tau = 0.07"


def t_infonce_tau_symmetric(sub):
    f = sub.namespace("info_nce_loss")["info_nce_loss"]
    for seed, n, tau in ((1, 5, 0.5), (2, 8, 1.0), (3, 3, 0.2)):
        a, b = _loss_inputs(n, 4, seed)
        got = _number(f(a, b, tau))
        assert abs(got - REF["info_nce_loss"](a, b, tau).item()) < 1e-4, (
            "average the image->text and text->image losses, dividing the logits by tau")


def t_infonce_normalized(sub):
    f = sub.namespace("info_nce_loss")["info_nce_loss"]
    a, b = _loss_inputs(7, 5, 4)
    base = _number(f(a, b, 0.1))
    scaled = _number(f(a * 9.0, b * 0.3, 0.1))
    assert abs(base - scaled) < 1e-4, "normalize the embeddings before comparing them"


def t_infonce_grad(sub):
    f = sub.namespace("info_nce_loss")["info_nce_loss"]
    a, b = _loss_inputs(6, 4, 5)
    a.requires_grad_(True)
    b.requires_grad_(True)
    loss = f(a, b, 0.07)
    assert isinstance(loss, torch.Tensor), "return a PyTorch tensor so training can call backward()"
    loss.backward()
    assert a.grad is not None and b.grad is not None and a.grad.abs().sum() > 0, (
        "the loss must stay differentiable (no .item() or NumPy inside)")


# ---- Part 4: training ----------------------------------------------------- #
def _trained_reference(ns, tau):
    h_img, h_txt, _ = ns["train_heads"](tau=tau)
    return ns["test_similarity"](h_img, h_txt)


def t_trained_matrix(sub):
    ns = sub.run()
    S = _array(ns.get("S_trained"), (ns["N_TEST"], ns["N_TEST"]))
    assert np.allclose(S, _trained_reference(ns, 0.07), atol=1e-5), (
        "S_trained should be the test similarity of heads trained with tau = 0.07")


def t_trained_accuracy(sub):
    ns = sub.run()
    S = _array(ns.get("S_trained"), (ns["N_TEST"], ns["N_TEST"]))
    assert REF["retrieval_accuracy"](S) > 0.4, "trained heads should beat random heads by a wide margin"


# ---- Part 5: temperature sweep ------------------------------------------- #
def _results(sub) -> dict:
    ns = sub.run()
    results = ns.get("results")
    assert isinstance(results, dict), "`results` should be a dictionary"
    assert set(results) == set(TAUS), "store one accuracy per temperature in TAUS"
    return results


def t_sweep_keys(sub):
    results = _results(sub)
    for tau in TAUS:
        acc = _number(results[tau])
        assert 0.0 <= acc <= 1.0, "each value should be an accuracy between 0 and 1"


def t_sweep_values(sub):
    ns = sub.run()
    results = _results(sub)
    for tau in TAUS:
        expected = REF["retrieval_accuracy"](_trained_reference(ns, tau))
        assert abs(_number(results[tau]) - expected) < 1e-9, (
            f"results[{tau}] should be the test image->text accuracy of heads trained at that tau")


# ---- Reflections --------------------------------------------------------- #
def _reflection(sub, name, min_words):
    r = sub.value(name)
    assert isinstance(r, str) and "TODO" not in r, f"{name} is not filled in"
    assert word_count(r) >= min_words, f"{name}: write at least {min_words} words"
    return r.lower()


def t_p2(sub):
    r = _reflection(sub, "part2_reflection", 30)
    assert "outdoor" in r, "part2_reflection should use the 'outdoors' dimension"
    assert any(k in r for k in ("animal", "direction", "proportion", "angle", "ratio")), (
        "part2_reflection should explain why the true pair scores below 1")


def t_p4(sub):
    r = _reflection(sub, "part4_reflection", 40)
    assert any(k in r for k in ("map", "transform", "project", "general", "space")), (
        "part4_reflection should say what the heads learned")
    assert "noise" in r or "noisy" in r, "part4_reflection should explain the role of the noise"


def t_p5(sub):
    r = _reflection(sub, "part5_reflection", 40)
    assert any(k in r for k in ("softmax", "uniform", "flat", "gradient", "probabilit")), (
        "part5_reflection should explain what a high temperature does to the softmax")


GROUPS: list[tuple[str, list[tuple[str, float, Check]]]] = [
    (
        "Part 1: cosine_similarity",
        [
            ("matches the formula on random vectors", 1, t_cosine_values),
            ("length does not matter; -1, 0 and 1 cases", 1, t_cosine_properties),
        ],
    ),
    (
        "Part 2: l2_normalize and similarity_matrix",
        [
            ("l2_normalize", 1, t_l2_normalize),
            ("similarity_matrix on square inputs", 1, t_similarity_square),
            ("similarity_matrix on different numbers of rows", 1, t_similarity_rect),
        ],
    ),
    (
        "Part 2: retrieval_accuracy",
        [
            ("matches the reference on random matrices", 1, t_retrieval_values),
            ("uses each row's best column", 1, t_retrieval_direction),
        ],
    ),
    (
        "Part 3: random projection heads",
        [
            ("W_img and W_txt have the right shapes", 1, t_random_shapes),
            ("S_random compares the projected test features", 1, t_random_similarity),
        ],
    ),
    (
        "Part 4: info_nce_loss",
        [
            ("matches the reference at tau = 0.07", 1, t_infonce_value),
            ("both directions, other temperatures", 1, t_infonce_tau_symmetric),
            ("normalizes the embeddings", 1, t_infonce_normalized),
            ("differentiable PyTorch tensor", 1, t_infonce_grad),
        ],
    ),
    (
        "Part 4: training the heads",
        [
            ("S_trained is the test similarity of the trained heads", 1, t_trained_matrix),
            ("trained heads beat random heads", 1, t_trained_accuracy),
        ],
    ),
    (
        "Part 5: temperature sweep",
        [
            ("one accuracy per temperature", 1, t_sweep_keys),
            ("accuracies match retraining at each tau", 1, t_sweep_values),
        ],
    ),
    (
        "Reflections",
        [
            ("part2_reflection", 1, t_p2),
            ("part4_reflection", 1, t_p4),
            ("part5_reflection", 1, t_p5),
        ],
    ),
]
MAX_TOTAL = float(sum(pts for _, tests in GROUPS for _, pts, _ in tests))  # 20


class CheckTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CheckTimeout("check took too long")


def run_checks(sub: Submission) -> list[dict[str, Any]]:
    """One Gradescope entry per section. Scores only: students never see failure details.

    Set AUTOGRADE_DEBUG=1 when running locally to print which sub-checks failed.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(RUN_TIMEOUT_S)               # running the whole file happens once, up front
    try:
        sub.run()
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        sub._run_ns = reference_namespace()
        if os.environ.get("AUTOGRADE_DEBUG"):
            print(f"[debug] running the file: {type(exc).__name__}: {exc}")
    finally:
        if use_alarm:
            signal.alarm(0)

    out: list[dict[str, Any]] = []
    for group, tests in GROUPS:
        earned = 0.0
        total = 0.0
        for name, pts, fn in tests:
            total += pts
            if use_alarm:
                signal.alarm(CHECK_TIMEOUT_S)
            try:
                fn(sub)
                earned += pts
            except BaseException as exc:  # includes NotImplementedError from untouched stubs
                if isinstance(exc, KeyboardInterrupt):
                    raise
                if os.environ.get("AUTOGRADE_DEBUG"):
                    print(f"[debug] {group}: {name}: {type(exc).__name__}: {exc}")
            finally:
                if use_alarm:
                    signal.alarm(0)
        out.append(
            {
                "name": group,
                "score": float(earned),
                "max_score": float(total),
                "output": "",
                "visibility": "visible",
            }
        )
    return out


def write_and_print(results_dir: Path, payload: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def failure_payload(name: str, msg: str) -> dict[str, Any]:
    if os.environ.get("AUTOGRADE_DEBUG"):
        print(f"[debug] {name}: {msg}")
    return {
        "score": 0.0,
        "output": msg,
        "tests": [
            {"name": name, "score": 0.0, "max_score": MAX_TOTAL, "output": msg, "visibility": "visible"}
        ],
    }


def main() -> int:
    on_gradescope = Path("/autograder").exists()
    results_dir = Path("/autograder/results") if on_gradescope else LOCAL_RESULTS

    env_sub = os.environ.get("AUTOGRADE_SUBMISSION")
    if env_sub:
        submission_dir = Path(env_sub)
    else:
        submission_dir = next((p for p in SUBMISSION_CANDIDATES if p.exists()), SUBMISSION_CANDIDATES[0])

    try:
        path = find_submission(submission_dir)
    except Exception as exc:
        write_and_print(results_dir, failure_payload("Submission check", str(exc)))
        return 0

    try:
        sub = Submission(path)
    except SyntaxError as exc:
        msg = (
            f"{path.name} has a syntax error and could not be parsed:\n"
            f"  line {exc.lineno}: {exc.msg}\n"
            "Re-export your notebook to .py after running it top to bottom without errors."
        )
        write_and_print(results_dir, failure_payload("File parses", msg))
        return 0

    tests_out = run_checks(sub)
    score = round(sum(t["score"] for t in tests_out), 2)
    payload = {
        "score": score,
        "output": "",
        "visibility": "visible",
        "stdout_visibility": "hidden",
        "tests": tests_out,
    }
    write_and_print(results_dir, payload)
    print(f"Score: {score} / {MAX_TOTAL:g}")
    for t in tests_out:
        print(f"{t['name']}: {t['score']:g}/{t['max_score']:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
