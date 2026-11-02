#!/usr/bin/env python3
"""
Gradescope autograder for diffusion_assignment (submitted as a .py file)

Package lives at:
  Assignments/2_multimodal_+_advanced_models/diffusion_assignment_autograder/

How it works
  * The submission is parsed with `ast`; the whole file is never run, so the student's
    training loop, GPU code, SD-Turbo download and plots are skipped.
  * The graded functions are exec'd one at a time on top of the PROVIDED code
    (provided.py, generated from the handout) and the reference versions of the other
    graded functions (reference_solution.py). A bug in one function therefore costs points
    only in its own section.
  * The functions are run on the grader's own tiny models (a "perfect" model that knows the
    answer, and a few imperfect ones) and random inputs, and compared with the reference
    implementation. Everything runs on a CPU in seconds.
  * The prompts and the written reflections are checked for form and length only; a manual
    read is still recommended.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

SOURCE_DIR = Path(__file__).resolve().parent
PROVIDED_SRC = (SOURCE_DIR / "provided.py").read_text(encoding="utf-8")
REFERENCE_SRC = (SOURCE_DIR / "reference_solution.py").read_text(encoding="utf-8")

LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [Path("/autograder/submission"), SOURCE_DIR.parent]

GRADED_FUNCS = {"make_alpha_bar", "add_noise", "predict_clean", "noise_prediction_loss",
                "ddim_sample", "slerp"}
REFLECTIONS = [f"part{i}_reflection" for i in range(1, 8)]
SAFE_ASSIGN_NAMES = {"MY_PROMPTS", "FAVORITE", *REFLECTIONS}
ALLOWED_IMPORTS = {"math", "os", "random", "time", "numpy", "torch", "PIL", "typing", "functools",
                   "itertools", "collections", "copy", "re", "json", "statistics"}
CHECK_TIMEOUT_S = 120


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
            "No .py file found in your submission. Submit firstname_lastname_diffusion.py." + hint
        )
    for p in pys:
        if p.name.endswith("_diffusion.py"):
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


def base_namespace() -> dict[str, Any]:
    """PROVIDED code plus reference versions of every graded function."""
    ns: dict[str, Any] = {"__name__": "graded"}
    exec(compile(PROVIDED_SRC, "provided.py", "exec"), ns)
    exec(compile(REFERENCE_SRC, "reference_solution.py", "exec"), ns)
    return ns


REF = base_namespace()
PROVIDED_NAMES = set(REF) - GRADED_FUNCS


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
                if self._import_allowed(node):
                    self.helpers.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in GRADED_FUNCS:
                    self.defs[node.name] = node            # last definition wins, like Python
                elif node.name not in PROVIDED_NAMES:
                    self.helpers.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in SAFE_ASSIGN_NAMES:
                    self.assigns[name] = node
                elif name not in PROVIDED_NAMES and _is_literal(node.value):
                    self.helpers.append(node)          # student constants used by their functions

    @staticmethod
    def _import_allowed(node: ast.stmt) -> bool:
        if isinstance(node, ast.Import):
            return all(a.name.split(".")[0] in ALLOWED_IMPORTS for a in node.names)
        return bool(node.module) and node.module.split(".")[0] in ALLOWED_IMPORTS and node.level == 0

    def _exec(self, nodes: list[ast.stmt], ns: dict[str, Any]) -> None:
        for node in nodes:
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
            except Exception:
                pass  # a broken helper only matters if a graded function needs it

    def namespace(self, *student_funcs: str) -> dict[str, Any]:
        """Provided + reference code, with the student's versions of `student_funcs`."""
        ns = base_namespace()
        self._exec(self.helpers, ns)
        for name in student_funcs:
            node = self.defs.get(name)
            if node is None:
                raise AssertionError(f"`{name}` is not defined at the top level of your file.")
            exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
        return ns

    def fn(self, name: str) -> Callable:
        return self.namespace(name)[name]

    def value(self, name: str) -> Any:
        node = self.assigns.get(name)
        if node is None:
            raise AssertionError(f"`{name}` is not assigned at the top level of your file.")
        ns = base_namespace()
        exec(compile(ast.Module(body=[node], type_ignores=[]), self.path.name, "exec"), ns)
        return ns[name]


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Shared test material
# --------------------------------------------------------------------------- #
def alpha_bar_tensor(T: int = 1000) -> torch.Tensor:
    return torch.tensor(REF["make_alpha_bar"](T), dtype=torch.float32)


def oracle_model(target: torch.Tensor, ab: torch.Tensor, calls: list | None = None) -> Callable:
    """A 'perfect' model that knows the clean pictures: it returns the exact noise."""
    def model(x_t, t):
        if calls is not None:
            calls.append(t.clone())
        a = ab[t].view(-1, 1, 1, 1)
        return (x_t - a ** 0.5 * target) / (1 - a) ** 0.5
    return model


def imperfect_models(ab: torch.Tensor) -> dict[str, Callable]:
    """Models that are wrong in different ways but depend on both the picture and the step."""
    target = torch.linspace(-0.8, 0.8, 2 * 1 * 8 * 8).view(2, 1, 8, 8)

    def tanh_model(x, t):
        return torch.tanh(x) * (1 + t.float().view(-1, 1, 1, 1) / 1000.0)

    def blurred_oracle(x, t):
        blurred = (target + torch.roll(target, 1, dims=-1) + torch.roll(target, 1, dims=-2)) / 3
        a = ab[t].view(-1, 1, 1, 1)
        return (x - a ** 0.5 * blurred) / (1 - a) ** 0.5

    return {"tanh": tanh_model, "blurred": blurred_oracle}


def _close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    a = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a, dtype=torch.float64)
    b = torch.as_tensor(np.asarray(b) if not torch.is_tensor(b) else b, dtype=torch.float64)
    return a.shape == b.shape and bool(torch.allclose(a, b, atol=atol, rtol=1e-4))


def word_count(s: Any) -> int:
    return len(re.findall(r"\w+", s)) if isinstance(s, str) else 0


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
Check = Callable[[Submission], None]


# ---- Forward diffusion ---------------------------------------------------- #
def t_alpha_bar(sub):
    f = sub.fn("make_alpha_bar")
    ref = REF["make_alpha_bar"]
    default = np.asarray(f(), dtype=float)
    assert default.shape == (1000,), "make_alpha_bar() should return 1000 numbers by default"
    assert _close(default, ref())
    for args in [(1000,), (10, 0.1, 0.5), (50, 1e-3, 0.05), (200, 1e-4, 0.02)]:
        got = np.asarray(f(*args), dtype=float)
        assert got.shape == (args[0],), f"make_alpha_bar{args} should return {args[0]} numbers"
        assert _close(got, ref(*args)), f"make_alpha_bar{args} has the wrong values"
    assert np.all(np.diff(default) < 0), "alpha_bar should keep falling"


def t_add_noise(sub):
    f = sub.fn("add_noise")
    ref = REF["add_noise"]
    rng = np.random.default_rng(0)
    for shape in [(8, 8, 3), (5,), (2, 1, 4, 4)]:
        x0, eps = rng.uniform(-1, 1, shape), rng.standard_normal(shape)
        x0_copy = x0.copy()
        for a in (1.0, 0.7, 0.25, 0.0):
            assert _close(f(x0, a, eps), ref(x0, a, eps)), f"numpy inputs, abar = {a}"
        assert np.array_equal(x0, x0_copy), "do not change x0 in place"
    x0 = torch.rand(6, 1, 8, 8) * 2 - 1
    eps = torch.randn_like(x0)
    ab = alpha_bar_tensor()[torch.tensor([0, 10, 100, 400, 800, 999])].view(-1, 1, 1, 1)
    got = f(x0, ab, eps)
    assert torch.is_tensor(got), "add_noise should also work on PyTorch tensors (use ** 0.5)"
    assert _close(got, ref(x0, ab, eps)), "tensor inputs with one abar per picture"


def t_predict_clean(sub):
    f = sub.fn("predict_clean")
    ref = REF["predict_clean"]
    rng = np.random.default_rng(1)
    for a in (0.95, 0.5, 0.05):
        x0, eps = rng.uniform(-1, 1, (8, 8, 3)), rng.standard_normal((8, 8, 3))
        x_t = REF["add_noise"](x0, a, eps)
        assert _close(f(x_t, eps, a), x0), f"with the true noise x0 must come back exactly (abar = {a})"
        guess = rng.standard_normal((8, 8, 3))
        assert _close(f(x_t, guess, a), ref(x_t, guess, a)), f"wrong value for an imperfect guess (abar = {a})"
    x_t = torch.randn(4, 1, 8, 8)
    guess = torch.randn_like(x_t)
    ab = alpha_bar_tensor()[torch.tensor([5, 50, 500, 900])].view(-1, 1, 1, 1)
    got = f(x_t, guess, ab)
    assert torch.is_tensor(got) and _close(got, ref(x_t, guess, ab)), "tensor inputs with one abar per picture"


# ---- Training loss -------------------------------------------------------- #
def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(B, 1, 8, 8, generator=g) * 2 - 1
    t = torch.randint(0, 1000, (B,), generator=g)
    eps = torch.randn(B, 1, 8, 8, generator=g)
    return x0, t, eps


def t_loss_values(sub):
    f = sub.fn("noise_prediction_loss")
    ab = alpha_bar_tensor()
    x0, t, eps = _batch()
    oracle = oracle_model(x0, ab)
    assert f(oracle, x0, t, eps, ab).item() < 1e-4, (
        "a perfect model should give a loss near 0 (use a separate abar for each picture)")
    torch.manual_seed(3)
    conv = nn.Conv2d(1, 1, 3, padding=1)
    model = lambda x, tt: conv(x) * (1 + tt.float().view(-1, 1, 1, 1) / 1000)
    for seed in (0, 1, 2):
        x0, t, eps = _batch(seed=seed)
        got = f(model, x0, t, eps, ab)
        want = REF["noise_prediction_loss"](model, x0, t, eps, ab)
        assert _close(got.detach(), want.detach()), f"loss differs from the reference (batch {seed})"


def t_loss_grad(sub):
    f = sub.fn("noise_prediction_loss")
    ab = alpha_bar_tensor()
    x0, t, eps = _batch(B=4, seed=5)
    torch.manual_seed(4)
    conv = nn.Conv2d(1, 1, 3, padding=1)
    loss = f(lambda x, tt: conv(x), x0, t, eps, ab)
    assert torch.is_tensor(loss) and loss.dim() == 0, "return a single number (a 0-dimensional tensor)"
    loss.backward()
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0, "the loss must keep the gradient"


# ---- Sampling loop -------------------------------------------------------- #
def _noise(B=2, seed=0):
    return torch.randn(B, 1, 8, 8, generator=torch.Generator().manual_seed(seed))


def t_sample_oracle(sub):
    f = sub.fn("ddim_sample")
    ab = alpha_bar_tensor()
    target = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(9)) * 2 - 1
    for steps in (1, 4, 20):
        out = f(oracle_model(target, ab), _noise(), ab, steps)
        assert torch.is_tensor(out) and out.shape == target.shape, "return a tensor shaped like x_T"
        assert torch.allclose(out, target, atol=1e-3), f"a perfect model must recover the target in {steps} step(s)"


def t_sample_matches(sub):
    f = sub.fn("ddim_sample")
    ab = alpha_bar_tensor()
    ref = REF["ddim_sample"]
    for name, model in imperfect_models(ab).items():
        for steps in (1, 3, 7, 25):
            got = f(model, _noise(seed=steps), ab, steps)
            want = ref(model, _noise(seed=steps), ab, steps)
            assert _close(got, want, atol=1e-4), f"{name} model, {steps} steps: result differs from the reference"
    # the model must be called once per step, with the right step numbers as a LongTensor
    target = torch.zeros(2, 1, 8, 8)
    mine, theirs = [], []
    f(oracle_model(target, ab, mine), _noise(), ab, 4)
    ref(oracle_model(target, ab, theirs), _noise(), ab, 4)
    assert len(mine) == 4, "call the model exactly once per step"
    for m, w in zip(mine, theirs):
        assert m.dtype == torch.long and m.shape == (2,) and torch.equal(m, w), (
            "pass t as a LongTensor with one entry per picture, counting down from the noisiest step")


def t_sample_clamp(sub):
    f = sub.fn("ddim_sample")
    ab = alpha_bar_tensor()
    for model in (lambda x, t: 50 * torch.ones_like(x), lambda x, t: -50 * torch.ones_like(x)):
        out = f(model, _noise(), ab, 5)
        assert out.shape == (2, 1, 8, 8), "return a tensor shaped like x_T"
        assert out.min() >= -1 - 1e-6 and out.max() <= 1 + 1e-6, "clamp the guessed clean picture to [-1, 1]"


# ---- slerp ---------------------------------------------------------------- #
def t_slerp(sub):
    f = sub.fn("slerp")
    ref = REF["slerp"]
    for shape, seed in [((1, 28, 28), 0), ((3, 8, 8), 1), ((64,), 2)]:
        g = torch.Generator().manual_seed(seed)
        z0, z1 = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
        assert torch.allclose(f(z0, z1, 0.0), z0, atol=1e-4), "alpha = 0 should give z0"
        assert torch.allclose(f(z0, z1, 1.0), z1, atol=1e-4), "alpha = 1 should give z1"
        for a in (0.25, 0.5, 0.9):
            got = f(z0, z1, a)
            assert got.shape == z0.shape and _close(got, ref(z0, z1, a), atol=1e-4), f"alpha = {a}"
    g = torch.Generator().manual_seed(7)
    z0, z1 = torch.randn(1, 28, 28, generator=g), torch.randn(1, 28, 28, generator=g)
    assert abs(f(z0, z1, 0.5).norm().item() - z0.norm().item()) < 0.15 * z0.norm().item(), (
        "slerp should keep the size of the noise")


# ---- Your prompts --------------------------------------------------------- #
def t_my_prompts(sub):
    prompts = sub.value("MY_PROMPTS")
    assert isinstance(prompts, list) and len(prompts) >= 4, "write at least 4 prompts"
    assert all(isinstance(p, str) and len(p.split()) >= 2 for p in prompts), "every prompt needs at least 2 words"
    assert not any("todo" in p.lower() for p in prompts), "replace the placeholders"
    assert len(set(prompts)) == len(prompts), "the prompts must all be different"
    assert len(prompts[-1].split()) >= len(prompts[0].split()) + 3, "the last prompt should be at least 3 words longer"
    assert len(prompts[0].split()) <= len(prompts[-1].split()), "order the prompts from shortest to longest"


def t_favorite(sub):
    fav = sub.value("FAVORITE")
    prompts = sub.value("MY_PROMPTS")
    assert isinstance(fav, dict) and {"prompt", "seed", "steps", "why"} <= set(fav), "FAVORITE needs prompt, seed, steps, why"
    assert fav["prompt"] in prompts, "FAVORITE['prompt'] must be one of your MY_PROMPTS, copied exactly"
    assert isinstance(fav["seed"], int) and not isinstance(fav["seed"], bool) and fav["seed"] >= 0
    assert fav["steps"] in (1, 2, 3, 4), "steps must be 1, 2, 3, or 4"
    assert word_count(fav["why"]) >= 10, "write at least 10 words in 'why'"


# ---- Reflections ---------------------------------------------------------- #
def _reflection(sub, name, min_words, all_of=(), any_groups=()):
    r = sub.value(name)
    assert isinstance(r, str), f"{name} must be a string"
    low = r.lower()
    assert "todo" not in low, f"{name}: replace the placeholder with your answer"
    assert word_count(r) >= min_words, f"{name}: write at least {min_words} words"
    for kw in all_of:
        assert kw in low, f"{name} should use the word '{kw}'"
    for group in any_groups:
        assert any(k in low for k in group), f"{name} should use one of {group}"


def t_p1(sub):
    _reflection(sub, "part1_reflection", 40, all_of=("noise", "signal"))


def t_p2(sub):
    _reflection(sub, "part2_reflection", 50, any_groups=(("error", "mistake"), ("step",)))


def t_p3(sub):
    _reflection(sub, "part3_reflection", 50, all_of=("noise",))


def t_p4(sub):
    _reflection(sub, "part4_reflection", 50, all_of=("step", "noise"))


def t_p5(sub):
    _reflection(sub, "part5_reflection", 40, all_of=("noise",))


def t_p6(sub):
    _reflection(sub, "part6_reflection", 60, all_of=("prompt",), any_groups=(("detail", "style"),))


def t_p7(sub):
    _reflection(sub, "part7_reflection", 50,
                any_groups=(("consent", "permission", "fake", "label", "harm", "copyright"),))


GROUPS: list[tuple[str, list[tuple[str, float, Check]]]] = [
    (
        "Forward diffusion (make_alpha_bar, add_noise)",
        [
            ("make_alpha_bar", 1, t_alpha_bar),
            ("add_noise", 1, t_add_noise),
        ],
    ),
    (
        "Recovering the picture (predict_clean)",
        [("predict_clean", 1, t_predict_clean)],
    ),
    (
        "Training loss (noise_prediction_loss)",
        [
            ("loss values", 1, t_loss_values),
            ("loss shape and gradients", 0.5, t_loss_grad),
        ],
    ),
    (
        "Sampling loop (ddim_sample)",
        [
            ("a perfect model recovers the target", 1, t_sample_oracle),
            ("matches the reference, right step numbers", 1, t_sample_matches),
            ("clamps the guessed picture", 0.5, t_sample_clamp),
        ],
    ),
    (
        "Morphing (slerp)",
        [("slerp", 1, t_slerp)],
    ),
    (
        "Your prompts (MY_PROMPTS, FAVORITE)",
        [
            ("MY_PROMPTS", 1, t_my_prompts),
            ("FAVORITE", 1, t_favorite),
        ],
    ),
    (
        "Reflections",
        [
            ("part1_reflection", 1.5, t_p1),
            ("part2_reflection", 1.5, t_p2),
            ("part3_reflection", 1.5, t_p3),
            ("part4_reflection", 2, t_p4),
            ("part5_reflection", 1, t_p5),
            ("part6_reflection", 1.5, t_p6),
            ("part7_reflection", 1, t_p7),
        ],
    ),
]
MAX_TOTAL = float(sum(pts for _, tests in GROUPS for _, pts, _ in tests))  # 20


class CheckTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CheckTimeout(f"check took longer than {CHECK_TIMEOUT_S} s")


def run_checks(sub: Submission) -> list[dict[str, Any]]:
    """One Gradescope entry per section. Scores only: students never see failure details.

    Set AUTOGRADE_DEBUG=1 when running locally to print which sub-checks failed.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    out: list[dict[str, Any]] = []
    for group, tests in GROUPS:
        earned = 0.0
        total = 0.0
        for name, pts, fn in tests:
            total += pts
            if use_alarm:
                signal.alarm(CHECK_TIMEOUT_S)
            try:
                torch.manual_seed(0)
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
