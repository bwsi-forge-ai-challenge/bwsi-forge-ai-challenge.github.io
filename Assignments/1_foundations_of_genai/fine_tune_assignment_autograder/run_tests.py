#!/usr/bin/env python3
"""
Gradescope autograder for fine_tune_assignment (submitted as a .py file)

Package lives at:
  Assignments/1_foundations_of_genai/fine_tune_assignment_autograder/

Why static? The real script downloads distilgpt2, logs in to Weights & Biases and
trains twice -- far too slow / network-dependent for an autograder. Instead we:
  * parse the submission with `ast` (never run the whole file), and
  * evaluate only the safe pieces (constants + the two prompt-format functions).
Commented-out code is invisible to the AST, so "uncomment the training cell"
is checked for real.
"""

from __future__ import annotations

import ast
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Callable

SOURCE_DIR = Path(__file__).resolve().parent
DATASET_ID = "dvgodoy/yoda_sentences"

LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [Path("/autograder/submission"), SOURCE_DIR.parent]

# Top-level names we are willing to evaluate from the student's file.
SAFE_ASSIGN_NAMES = {
    "SEED",
    "MODEL_NAME",
    "TASK_NAME",
    "DATASET_NAME",
    "INPUT_COLUMN",
    "OUTPUT_COLUMN",
    "PROMPT_TEMPLATE",
    "LEARNING_RATE_RUN1",
    "LEARNING_RATE_RUN2",
    "NUM_EPOCHS",
    "BATCH_SIZE",
    "MAX_LENGTH",
    "FINETUNED_DIR",
    "part1_reflection",
    "part4_reflection",
}
SAFE_FUNCS = {"format_training_text", "format_inference_prompt"}


# --------------------------------------------------------------------------- #
# Loading the submission
# --------------------------------------------------------------------------- #
def find_submission(submission_dir: Path) -> Path:
    # Local overrides: AUTOGRADE_FILE (or legacy AUTOGRADE_NOTEBOOK) = path to the .py
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
            " You uploaded a .ipynb -- export it to a .py first (see Part 5 of the assignment)."
            if has_nb
            else ""
        )
        raise FileNotFoundError(
            "No .py file found in your submission. Submit firstname_lastname_fine_tune.py." + hint
        )
    for p in pys:
        if p.name.endswith("_fine_tune.py"):
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


class Submission:
    def __init__(self, path: Path, root: Path | None = None):
        self.path = path
        self.root = root or path.parent
        self.source = sanitize(path.read_text(encoding="utf-8", errors="replace"))
        self.tree = ast.parse(self.source, filename=path.name)
        self.ns: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self._load_safe_values()

    def _load_safe_values(self) -> None:
        """Exec only whitelisted top-level assignments / functions, one by one."""
        for node in self.tree.body:
            names: set[str] = set()
            if isinstance(node, ast.Assign):
                names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = {node.target.id}
            elif isinstance(node, ast.FunctionDef) and node.name in SAFE_FUNCS:
                names = {node.name}
            else:
                continue
            if not names or not (names <= SAFE_ASSIGN_NAMES | SAFE_FUNCS):
                continue
            mod = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(mod, self.path.name, "exec"), self.ns, self.ns)
            except Exception as exc:  # keep going; tests report what is missing
                for n in names:
                    self.errors[n] = f"{type(exc).__name__}: {exc}"
                    self.ns.pop(n, None)

    def images(self) -> list[Path]:
        """Non-trivial image files uploaded next to the .py (existence check only)."""
        exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        return [
            p
            for p in self.root.rglob("*")
            if p.suffix.lower() in exts
            and p.is_file()
            and p.stat().st_size >= 1024
            and "__pycache__" not in p.parts
            and not any("_autograder" in part for part in p.parts)
        ]

    # -- AST helpers -------------------------------------------------------- #
    def function_def(self, name: str) -> ast.FunctionDef | None:
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def calls_outside(self, func_name: str, exclude_def: str | None = None) -> list[tuple[ast.Call, bool, int]]:
        """Calls to `func_name` at module level (not inside `exclude_def`).

        Returns (call_node, inside_loop, loop_multiplicity).
        """
        found: list[tuple[ast.Call, bool, int]] = []

        def visit(node: ast.AST, loop_mult: int | None) -> None:
            if isinstance(node, ast.FunctionDef) and node.name == exclude_def:
                return
            if isinstance(node, ast.For):
                mult = 1
                if isinstance(node.iter, (ast.List, ast.Tuple)):
                    mult = max(1, len(node.iter.elts))
                for child in ast.iter_child_nodes(node):
                    visit(child, mult)
                return
            if isinstance(node, ast.Call) and _call_name(node) == func_name:
                found.append((node, loop_mult is not None, loop_mult or 1))
            for child in ast.iter_child_nodes(node):
                visit(child, loop_mult)

        visit(self.tree, None)
        return found

    def any_call(self, dotted: str) -> bool:
        return any(
            isinstance(n, ast.Call) and _call_name(n) == dotted for n in ast.walk(self.tree)
        )


def _call_name(call: ast.Call) -> str:
    f = call.func
    parts: list[str] = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _arg(call: ast.Call, index: int, kw: str) -> ast.expr | None:
    if len(call.args) > index:
        return call.args[index]
    for k in call.keywords:
        if k.arg == kw:
            return k.value
    return None


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
Check = Callable[[Submission], None]


def require(sub: Submission, name: str) -> Any:
    if name not in sub.ns:
        why = sub.errors.get(name)
        raise AssertionError(
            f"`{name}` is not defined at the top level of your file"
            + (f" (error evaluating it: {why})" if why else ".")
        )
    return sub.ns[name]


def word_count(s: Any) -> int:
    return len(re.findall(r"\w+", s)) if isinstance(s, str) else 0


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
# ---- Dataset ------------------------------------------------------- #
def t_loads_yoda_dataset(sub):
    for node in ast.walk(sub.tree):
        if isinstance(node, ast.Call) and _call_name(node) == "load_dataset":
            arg = _arg(node, 0, "path")
            if isinstance(arg, ast.Constant) and arg.value == DATASET_ID:
                return
            if isinstance(arg, ast.Name) and sub.ns.get(arg.id) == DATASET_ID:
                return
    raise AssertionError(f"Expected load_dataset({DATASET_ID!r}, ...) in your file.")


def t_format_training_text(sub):
    fn = require(sub, "format_training_text")
    got = fn("hello there", "Greetings.")
    assert isinstance(got, str), "format_training_text must return a string."
    expected = "### Input:\nhello there\n\n### Output:\nGreetings."
    assert got == expected, f"Expected {expected!r} but got {got!r}"


def t_format_inference_prompt(sub):
    fn = require(sub, "format_inference_prompt")
    got = fn("hello there")
    expected = "### Input:\nhello there\n\n### Output:\n"
    assert got == expected, f"Expected {expected!r} but got {got!r}"
    train = require(sub, "format_training_text")("hello there", "Greetings.")
    assert train.startswith(got), "Inference prompt must be the prefix of the training text."


# ---- Hyperparameters ---------------------------------------------- #
def _lr(sub, name):
    v = require(sub, name)
    assert v is not None, f"{name} is still None -- set a value in the hyperparameters cell."
    assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{name} must be a number."
    assert 0 < v < 1, f"{name}={v} should be a positive learning rate below 1 (e.g. 5e-5)."


def t_lr1(sub):
    _lr(sub, "LEARNING_RATE_RUN1")


def t_lr2(sub):
    _lr(sub, "LEARNING_RATE_RUN2")


def t_lr_differ(sub):
    a, b = require(sub, "LEARNING_RATE_RUN1"), require(sub, "LEARNING_RATE_RUN2")
    assert a is not None and b is not None, "Both learning rates must be set."
    assert a != b, "The two runs must use different learning rates to compare them."


def t_epochs(sub):
    v = require(sub, "NUM_EPOCHS")
    assert isinstance(v, (int, float)) and not isinstance(v, bool), "NUM_EPOCHS must be a number (still None?)."
    assert v >= 1, f"NUM_EPOCHS={v} must be at least 1."
    assert v <= 50, f"NUM_EPOCHS={v} is unreasonably large for this dataset."


def t_batch(sub):
    v = require(sub, "BATCH_SIZE")
    assert isinstance(v, int) and not isinstance(v, bool), "BATCH_SIZE must be an integer (still None?)."
    assert 1 <= v <= 64, f"BATCH_SIZE={v} should be between 1 and 64."


# ---- Training function (TODO 2) ------------------------------------ #
def _train_fn(sub):
    fn = sub.function_def("train_finetune_run")
    assert fn is not None, "`train_finetune_run` was removed -- keep the provided function."
    return fn


def _calls_in(fn, name):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _call_name(n) == name]


def _kwargs(call):
    return {k.arg for k in call.keywords if k.arg}


def t_train_args(sub):
    calls = _calls_in(_train_fn(sub), "TrainingArguments")
    assert calls, "train_finetune_run never calls TrainingArguments(...)."
    needed = {"output_dir", "num_train_epochs", "per_device_train_batch_size", "learning_rate"}
    missing = needed - _kwargs(calls[0])
    assert not missing, f"TrainingArguments is missing: {sorted(missing)}"


def t_train_report_to(sub):
    calls = _calls_in(_train_fn(sub), "TrainingArguments")
    assert calls, "train_finetune_run never calls TrainingArguments(...)."
    ok = any(
        k.arg == "report_to" and isinstance(k.value, ast.Constant) and k.value.value == "wandb"
        for k in calls[0].keywords
    )
    assert ok, 'TrainingArguments needs report_to="wandb".'


def t_train_trainer(sub):
    calls = _calls_in(_train_fn(sub), "Trainer")
    assert calls, "train_finetune_run never creates a Trainer(...)."
    missing = {"model", "args", "train_dataset", "data_collator"} - _kwargs(calls[0])
    assert not missing, f"Trainer is missing: {sorted(missing)}"


def t_train_runs_and_saves(sub):
    names = {_call_name(n) for n in ast.walk(_train_fn(sub)) if isinstance(n, ast.Call)}
    assert any(n.endswith(".train") for n in names), "train_finetune_run never calls trainer.train()."
    assert any(n.endswith(".save_model") for n in names), "train_finetune_run never saves the model."


def t_plot_uploaded(sub):
    imgs = sub.images()
    assert imgs, "No plot image (.png/.jpg) found in the submission; upload one from your W&B page."


def t_wandb(sub):
    login = sub.any_call("wandb.login")
    disabled = re.search(r"WANDB_MODE[\"']\s*\]\s*=\s*[\"']disabled", sub.source) is not None
    uses_wandb_report = re.search(r"report_to\s*=\s*[\"']wandb[\"']", sub.source) is not None
    assert login or disabled, (
        "No active `wandb.login()` call found (or WANDB_MODE='disabled' fallback). "
        "Make sure it is not commented out."
    )
    assert uses_wandb_report or disabled, "Training must log to W&B (report_to='wandb')."


def t_two_runs(sub):
    calls = sub.calls_outside("train_finetune_run", exclude_def="train_finetune_run")
    total = sum(mult for _, _, mult in calls)
    assert total >= 2, (
        f"Found {total} active call(s) to train_finetune_run; "
        "the training step needs two (uncomment both calls)."
    )


def t_runs_distinct(sub):
    calls = sub.calls_outside("train_finetune_run", exclude_def="train_finetune_run")
    if any(in_loop and mult >= 2 for _, in_loop, mult in calls):
        return  # looped over >= 2 configs: fine
    assert len(calls) >= 2, "Need two active train_finetune_run(...) calls first."

    def resolved(call):
        lr = _arg(call, 1, "learning_rate")
        out = _arg(call, 2, "output_dir")
        assert lr is not None and out is not None, (
            "Each call needs a learning rate and output directory: "
            "train_finetune_run(train_data, lr, output_dir, run_name)"
        )
        lr_src = ast.unparse(lr)
        if isinstance(lr, ast.Name) and lr.id in sub.ns:
            lr_src = repr(sub.ns[lr.id])
        return lr_src, ast.unparse(out)

    (lr_a, out_a), (lr_b, out_b) = resolved(calls[0][0]), resolved(calls[1][0])
    assert lr_a != lr_b, "The two training runs use the same learning rate."
    assert out_a != out_b, "The two runs must save to different output directories."


# ---- Part 4: evaluation ---------------------------------------------------- #
def t_eval_uses_template(sub):
    finetuned_dir = sub.ns.get("FINETUNED_DIR")
    if finetuned_dir is not None:
        assert isinstance(finetuned_dir, str) and finetuned_dir.strip(), (
            "FINETUNED_DIR should point at one of your two saved runs."
        )
    fi_calls = [
        n
        for n in ast.walk(sub.tree)
        if isinstance(n, ast.Call)
        and _call_name(n) == "generate_text"
        and any(
            isinstance(a, ast.Call) and _call_name(a) == "format_inference_prompt"
            for a in ast.walk(n)
        )
    ]
    assert fi_calls, (
        "Part 4 should call generate_text(finetuned_model, format_inference_prompt(prompt)) "
        "so the fine-tuned model sees the training format."
    )


# ---- Reflections ------------------------------------------------------------ #
def t_p1_length(sub):
    r = require(sub, "part1_reflection")
    n = word_count(r)
    assert n >= 20, f"part1_reflection has {n} words; write at least 20."


def t_p1_content(sub):
    r = require(sub, "part1_reflection")
    low = r.lower() if isinstance(r, str) else ""
    keys = ["continue", "next token", "next-token", "predict", "complet", "instruct", "autocomplete", "pretrain"]
    assert any(k in low for k in keys), (
        "part1_reflection should explain that the base model continues/predicts text "
        "rather than following instructions (and why)."
    )


def t_p4_length(sub):
    r = require(sub, "part4_reflection")
    n = word_count(r)
    assert n >= 40, f"part4_reflection has {n} words; write at least 40."


def t_p4_content(sub):
    r = require(sub, "part4_reflection")
    low = r.lower() if isinstance(r, str) else ""
    assert "loss" in low, "part4_reflection should describe your loss curves."
    assert any(k in low for k in ("curve", "wandb", "w&b", "weights & biases", "weights and biases", "learning rate")), (
        "part4_reflection should refer to what you saw in W&B / the two learning rates."
    )
    assert any(k in low for k in ("base", "fine-tuned", "finetuned", "fine tuned", "before", "after")), (
        "part4_reflection should compare base vs fine-tuned behavior."
    )


GROUPS: list[tuple[str, list[tuple[str, float, Check]]]] = [
    (
        "Dataset",
        [
            ("loads dvgodoy/yoda_sentences", 1, t_loads_yoda_dataset),
            ("format_training_text correct", 1, t_format_training_text),
            ("format_inference_prompt correct", 1, t_format_inference_prompt),
        ],
    ),
    (
        "Training function",
        [
            ("TrainingArguments has the required settings", 2, t_train_args),
            ('report_to="wandb"', 1, t_train_report_to),
            ("Trainer built with model, args, dataset, collator", 1, t_train_trainer),
            ("trains and saves the model", 1, t_train_runs_and_saves),
        ],
    ),
    (
        "Hyperparameters",
        [
            ("LEARNING_RATE_RUN1 set", 1, t_lr1),
            ("LEARNING_RATE_RUN2 set", 1, t_lr2),
            ("learning rates differ", 1, t_lr_differ),
            ("NUM_EPOCHS set", 1, t_epochs),
            ("BATCH_SIZE set", 1, t_batch),
        ],
    ),
    (
        "Training runs",
        [
            ("two active train_finetune_run calls", 1, t_two_runs),
            ("runs differ in learning rate and output dir", 1, t_runs_distinct),
            ("W&B plot image uploaded", 1, t_plot_uploaded),
        ],
    ),
    (
        "Reflections",
        [
            ("part1_reflection length", 1, t_p1_length),
            ("part1_reflection content", 1, t_p1_content),
            ("part4_reflection length", 1, t_p4_length),
            ("part4_reflection content", 1, t_p4_content),
        ],
    ),
]
MAX_TOTAL = float(sum(pts for _, tests in GROUPS for _, pts, _ in tests))  # 20


def run_checks(sub: Submission) -> list[dict[str, Any]]:
    """One Gradescope entry per section. Scores only: students never see failure details.

    Set AUTOGRADE_DEBUG=1 when running locally to print which sub-checks failed.
    """
    out: list[dict[str, Any]] = []
    for group, tests in GROUPS:
        earned = 0.0
        total = 0.0
        for name, pts, fn in tests:
            total += pts
            try:
                fn(sub)
                earned += pts
            except Exception as exc:
                if os.environ.get("AUTOGRADE_DEBUG"):
                    print(f"[debug] {group}: {name}: {exc}")
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
        "output": "",
        "tests": [
            {"name": name, "score": 0.0, "max_score": MAX_TOTAL, "output": "", "visibility": "visible"}
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
        payload = failure_payload("Submission check", str(exc))
        write_and_print(results_dir, payload)
        return 0

    try:
        sub = Submission(path, submission_dir)
    except SyntaxError as exc:
        msg = (
            f"{path.name} has a syntax error and could not be parsed:\n"
            f"  line {exc.lineno}: {exc.msg}\n"
            "Re-export your notebook to .py after running it top to bottom without errors."
        )
        payload = failure_payload("File parses", msg)
        write_and_print(results_dir, payload)
        return 0

    tests_out = run_checks(sub)
    score = round(sum(t["score"] for t in tests_out), 2)
    header = f"Graded file: {path.name}\n"
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
