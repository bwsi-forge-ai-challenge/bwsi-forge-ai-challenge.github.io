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
STARTER_PATH = SOURCE_DIR / "starter_pairs.json"

LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [Path("/autograder/submission"), SOURCE_DIR.parent]

# Top-level names we are willing to evaluate from the student's file.
SAFE_ASSIGN_NAMES = {
    "SEED",
    "MODEL_NAME",
    "TASK_NAME",
    "TRAINING_PAIRS",
    "MIN_EXAMPLES",
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
    def __init__(self, path: Path):
        self.path = path
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


def starter_inputs() -> set[str]:
    data = json.loads(STARTER_PATH.read_text(encoding="utf-8"))
    return {_norm(p[0]) for p in data}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def good_pairs(sub: Submission) -> list[tuple[str, str]]:
    pairs = require(sub, "TRAINING_PAIRS")
    if not isinstance(pairs, (list, tuple)):
        raise AssertionError("TRAINING_PAIRS must be a list of (input, output) tuples.")
    out = []
    for i, p in enumerate(pairs):
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or not all(isinstance(x, str) and x.strip() for x in p)
        ):
            raise AssertionError(
                f"TRAINING_PAIRS[{i}] = {p!r} is not a pair of two non-empty strings."
            )
        out.append((p[0], p[1]))
    return out


def word_count(s: Any) -> int:
    return len(re.findall(r"\w+", s)) if isinstance(s, str) else 0


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
# ---- Dataset ------------------------------------------------------- #
def t_pairs_well_formed(sub):
    good_pairs(sub)


def t_at_least_50(sub):
    n = len(good_pairs(sub))
    assert n >= 50, f"TRAINING_PAIRS has {n} examples; need at least 50."


def t_added_new_pairs(sub):
    pairs = good_pairs(sub)
    starters = starter_inputs()
    new = [p for p in pairs if _norm(p[0]) not in starters]
    assert len(new) >= 5, (
        f"Found {len(new)} pairs that are not in the starter set; add at least 5 of your own."
    )


def t_no_duplicates(sub):
    pairs = good_pairs(sub)
    seen: dict[str, int] = {}
    for i, (inp, _) in enumerate(pairs):
        key = _norm(inp)
        assert key not in seen, (
            f"Duplicate input {inp!r} at positions {seen[key]} and {i}."
        )
        seen[key] = i


def t_new_pairs_quality(sub):
    pairs = good_pairs(sub)
    starters = starter_inputs()
    new = [p for p in pairs if _norm(p[0]) not in starters]
    bad = [p for p in new if _norm(p[0]) == _norm(p[1]) or word_count(p[1]) < 2]
    assert not bad, (
        "Your added pairs need a real output (different from the input, 2+ words). "
        f"Problem pair: {bad[0]!r}"
    )


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
    assert v <= 50, f"NUM_EPOCHS={v} is unreasonably large for 50 examples."


def t_batch(sub):
    v = require(sub, "BATCH_SIZE")
    assert isinstance(v, int) and not isinstance(v, bool), "BATCH_SIZE must be an integer (still None?)."
    assert 1 <= v <= 64, f"BATCH_SIZE={v} should be between 1 and 64."


# ---- Training ------------------------------------------------------ #
def t_train_fn_intact(sub):
    fn = sub.function_def("train_finetune_run")
    assert fn is not None, "`train_finetune_run` was removed -- keep the provided function."
    names = {_call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    for needed in ("Trainer", "TrainingArguments"):
        assert needed in names, f"train_finetune_run no longer calls {needed}(...)."
    assert any(n.endswith("train") for n in names if n.startswith("trainer")), (
        "train_finetune_run no longer calls trainer.train()."
    )


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
            "train_finetune_run(TRAINING_PAIRS, lr, output_dir, run_name)"
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
    assert n >= 30, f"part1_reflection has {n} words; write at least 30."


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
    assert n >= 60, f"part4_reflection has {n} words; write at least 60."


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
        "Dataset & prompt format",
        [
            ("pairs are well-formed (input, output) strings", 5, t_pairs_well_formed),
            ("at least 50 examples", 5, t_at_least_50),
            ("added at least 5 new examples", 8, t_added_new_pairs),
            ("no duplicate inputs", 4, t_no_duplicates),
            ("new examples have real outputs", 3, t_new_pairs_quality),
            ("format_training_text", 3, t_format_training_text),
            ("format_inference_prompt", 2, t_format_inference_prompt),
        ],
    ),
    (
        "Hyperparameters",
        [
            ("LEARNING_RATE_RUN1 set", 3, t_lr1),
            ("LEARNING_RATE_RUN2 set", 3, t_lr2),
            ("learning rates differ", 3, t_lr_differ),
            ("NUM_EPOCHS set", 3, t_epochs),
            ("BATCH_SIZE set", 3, t_batch),
        ],
    ),
    (
        "Training runs",
        [
            ("train_finetune_run still uses Trainer", 6, t_train_fn_intact),
            ("W&B login active", 6, t_wandb),
            ("two training runs executed", 10, t_two_runs),
            ("runs use different learning rate & output dir", 8, t_runs_distinct),
        ],
    ),
    (
        "Part 4 - evaluation",
        [("fine-tuned model prompted with the training format", 5, t_eval_uses_template)],
    ),
    (
        "Reflections",
        [
            ("part1_reflection length", 5, t_p1_length),
            ("part1_reflection content", 5, t_p1_content),
            ("part4_reflection length", 5, t_p4_length),
            ("part4_reflection content", 5, t_p4_content),
        ],
    ),
]


def run_checks(sub: Submission) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group, tests in GROUPS:
        for name, pts, fn in tests:
            try:
                fn(sub)
                score, msg = float(pts), "Passed"
            except AssertionError as exc:
                score, msg = 0.0, str(exc)
            except Exception:
                score, msg = 0.0, traceback.format_exc()
            out.append(
                {
                    "name": f"{group}: {name}",
                    "score": score,
                    "max_score": float(pts),
                    "output": msg,
                    "visibility": "visible",
                }
            )
    return out


def write_and_print(results_dir: Path, payload: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def failure_payload(name: str, msg: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "output": msg,
        "tests": [
            {"name": name, "score": 0.0, "max_score": 100.0, "output": msg, "visibility": "visible"}
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
        print(json.dumps(payload, indent=2))
        return 0

    try:
        sub = Submission(path)
    except SyntaxError as exc:
        msg = (
            f"{path.name} has a syntax error and could not be parsed:\n"
            f"  line {exc.lineno}: {exc.msg}\n"
            "Re-export your notebook to .py after running it top to bottom without errors."
        )
        payload = failure_payload("File parses", msg)
        write_and_print(results_dir, payload)
        print(msg)
        return 0

    tests_out = run_checks(sub)
    score = round(sum(t["score"] for t in tests_out), 2)
    header = f"Graded file: {path.name}\n"
    payload = {
        "score": score,
        "output": header,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "tests": tests_out,
    }
    write_and_print(results_dir, payload)
    print(header)
    print(f"Score: {score} / 100")
    for t in tests_out:
        status = "PASS" if t["score"] >= t["max_score"] - 1e-9 else "FAIL"
        print(f"[{status}] {t['name']}: {t['score']:.2f}/{t['max_score']:.2f}")
        if status == "FAIL":
            print("       " + t["output"].replace("\n", "\n       "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
