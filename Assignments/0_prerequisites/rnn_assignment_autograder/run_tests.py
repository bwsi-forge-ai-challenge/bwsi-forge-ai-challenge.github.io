#!/usr/bin/env python3
"""
Gradescope autograder for rnn_assignment.ipynb

Package lives at:
  Assignments/0_prerequisites/rnn_assignment_autograder/

Shared zip / smoke-test helpers live one level up in Assignments/.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import traceback
import unittest
from pathlib import Path
from typing import Any

import nbformat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SOURCE_DIR = Path(__file__).resolve().parent
DATA_PATH = SOURCE_DIR / "data" / "tinyshakespeare.txt"

# Local paths when testing outside Gradescope
LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [
    Path("/autograder/submission"),
    SOURCE_DIR.parent,  # local: assignment folder
]
RESULTS_CANDIDATES = [
    Path("/autograder/results"),
    LOCAL_RESULTS,
]


def find_dir(candidates: list[Path], create: bool = False) -> Path:
    for p in candidates:
        if p.exists():
            return p
    chosen = candidates[-1] if create else candidates[0]
    if create:
        chosen.mkdir(parents=True, exist_ok=True)
    return chosen


def on_gradescope() -> bool:
    return Path("/autograder").exists()


def find_notebook(submission_dir: Path) -> Path:
    # Local override examples:
    #   AUTOGRADE_NOTEBOOK=/path/to/rnn_assignment_solution.ipynb
    #   AUTOGRADE_SUBMISSION=/tmp/rnn_grade   # directory containing the .ipynb
    env_nb = os.environ.get("AUTOGRADE_NOTEBOOK")
    if env_nb:
        p = Path(env_nb)
        if not p.exists():
            raise FileNotFoundError(f"AUTOGRADE_NOTEBOOK not found: {p}")
        return p

    preferred = [
        "rnn_assignment.ipynb",
        "rnn_assignment_solution.ipynb",
    ]
    for name in preferred:
        hit = submission_dir / name
        if hit.exists():
            return hit
    notebooks = sorted(submission_dir.rglob("*.ipynb"))
    notebooks = [n for n in notebooks if ".ipynb_checkpoints" not in str(n)]
    if not notebooks:
        raise FileNotFoundError(
            f"No .ipynb found under {submission_dir}. "
            "Submit rnn_assignment.ipynb."
        )
    return notebooks[0]


def should_skip_cell(source: str) -> bool:
    """Skip cells that are slow, network-dependent, or self-testing."""
    markers = [
        "run_unit_tests(",
        "NUM_EPOCHS",
        "for epoch in range",
        "sample = generate(",
        "print(generate(",
        "SHAKESPEARE_URL",
        "urllib.request.urlopen",
    ]
    return any(m in source for m in markers)


def load_fixture_text() -> str:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing fixture corpus at {DATA_PATH}")
    return DATA_PATH.read_text(encoding="utf-8")


def exec_notebook(nb_path: Path) -> dict[str, Any]:
    """Execute student code cells into a shared namespace."""
    nb = nbformat.read(nb_path, as_version=4)
    ns: dict[str, Any] = {
        "__name__": "__main__",
        "torch": torch,
        "nn": nn,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "random": random,
        "math": math,
        "unittest": unittest,
        "time": __import__("time"),
        "urllib": __import__("urllib"),
    }

    # Force CPU for grading stability / speed
    ns["device"] = torch.device("cpu")
    ns["text"] = load_fixture_text()

    # Provide a no-op helper if student cells reference it
    def run_unit_tests(test_case_class, name=""):
        return None

    ns["run_unit_tests"] = run_unit_tests

    random.seed(17)
    torch.manual_seed(17)

    executed = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)
        if not source.strip() or should_skip_cell(source):
            continue
        try:
            # Keep device on CPU even if student cell reassigns it
            compiled = compile(source, f"{nb_path.name}:cell{i}", "exec")
            exec(compiled, ns, ns)
            ns["device"] = torch.device("cpu")
            executed += 1
        except Exception as exc:
            # Keep going so later partial credit can still be awarded when possible
            ns.setdefault("_cell_errors", []).append(
                {"cell": i, "error": f"{type(exc).__name__}: {exc}"}
            )
    ns["_executed_cells"] = executed

    # Ensure vocab maps exist even if vocab cell failed oddly
    if "char_to_idx" not in ns or "idx_to_char" not in ns or "vocab_size" not in ns:
        chars = sorted(set(ns["text"]))
        ns["chars"] = chars
        ns["vocab_size"] = len(chars)
        ns["char_to_idx"] = {ch: i for i, ch in enumerate(chars)}
        ns["idx_to_char"] = {i: ch for i, ch in enumerate(chars)}

    return ns


class GradescopeTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes: list[unittest.TestCase] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def run_suite(suite: unittest.TestSuite) -> GradescopeTestResult:
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=GradescopeTestResult)
    result = runner.run(suite)
    result.stream_output = stream.getvalue()
    return result  # type: ignore[return-value]


def make_todo1_tests(ns: dict[str, Any]):
    encode = ns.get("encode")
    CharDataset = ns.get("CharDataset")
    text = ns["text"]
    char_to_idx = ns["char_to_idx"]

    class TestTODO1(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            if encode is None or CharDataset is None:
                raise unittest.SkipTest("encode / CharDataset not defined in notebook")
            # Keep as plain callables (do NOT assign bare functions on the class —
            # that turns them into bound methods and breaks encode(s)).
            cls._encode = staticmethod(encode)
            cls._CharDataset = CharDataset
            cls.sample = text[:50]
            cls.seq_len = 4

        def test_encode_returns_correct_indices(self):
            sample = text[:30]
            result = self._encode(sample)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), len(sample))
            self.assertEqual(result, [char_to_idx[c] for c in sample])

        def test_encode_empty_string(self):
            self.assertEqual(self._encode(""), [])

        def test_dataset_length(self):
            ds = self._CharDataset(self.sample, self.seq_len)
            self.assertEqual(len(ds), len(self._encode(self.sample)) - self.seq_len)

        def test_dataset_getitem_shapes_and_types(self):
            ds = self._CharDataset(self.sample, self.seq_len)
            x, y = ds[0]
            self.assertEqual(x.shape, torch.Size([self.seq_len]))
            self.assertEqual(y.shape, torch.Size([]))
            self.assertEqual(x.dtype, torch.long)
            self.assertEqual(y.dtype, torch.long)

        def test_dataset_sliding_window_content(self):
            ds = self._CharDataset(self.sample, self.seq_len)
            idx = 2
            x, y = ds[idx]
            self.assertEqual(x.tolist(), self._encode(self.sample[idx : idx + self.seq_len]))
            self.assertEqual(y.item(), self._encode(self.sample[idx + self.seq_len])[0])

        def test_dataset_last_valid_index(self):
            ds = self._CharDataset(self.sample, self.seq_len)
            idx = len(ds) - 1
            x, y = ds[idx]
            self.assertEqual(x.tolist(), self._encode(self.sample[idx : idx + self.seq_len]))
            self.assertEqual(y.item(), self._encode(self.sample[idx + self.seq_len])[0])

    return TestTODO1


def make_todo2_tests(ns: dict[str, Any]):
    CharRNN = ns.get("CharRNN")
    vocab_size = ns["vocab_size"]
    device = ns["device"]

    class TestTODO2(unittest.TestCase):
        def setUp(self):
            if CharRNN is None:
                self.skipTest("CharRNN not defined in notebook")
            self.test_model = CharRNN(vocab_size, embed_dim=32, hidden_dim=64).to(device)

        def test_has_required_layers(self):
            self.assertIsInstance(self.test_model.embedding, nn.Embedding)
            self.assertIsInstance(self.test_model.rnn, nn.RNN)
            self.assertIsInstance(self.test_model.fc, nn.Linear)
            self.assertEqual(self.test_model.embedding.num_embeddings, vocab_size)
            self.assertEqual(self.test_model.fc.out_features, vocab_size)

        def test_forward_output_shapes(self):
            batch_size, seq_len = 4, 16
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            logits, hidden = self.test_model(x)
            self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
            self.assertEqual(hidden.shape, (1, batch_size, 64))

        def test_forward_with_provided_hidden_state(self):
            batch_size, seq_len = 2, 8
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            h0 = self.test_model.init_hidden(batch_size)
            logits, h1 = self.test_model(x, h0)
            self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
            self.assertEqual(h1.shape, h0.shape)

        def test_init_hidden_shape(self):
            h = self.test_model.init_hidden(3)
            self.assertEqual(h.shape, (1, 3, 64))

    return TestTODO2


def make_todo3_tests(ns: dict[str, Any]):
    CharRNN = ns.get("CharRNN")
    CharDataset = ns.get("CharDataset")
    train_one_epoch = ns.get("train_one_epoch")
    evaluate = ns.get("evaluate")
    criterion = ns.get("criterion")
    vocab_size = ns["vocab_size"]
    device = ns["device"]
    text = ns["text"]
    embed_dim = ns.get("EMBED_DIM", 64)
    hidden_dim = ns.get("HIDDEN_DIM", 256)

    class TestTODO3(unittest.TestCase):
        def setUp(self):
            missing = [
                name
                for name, obj in [
                    ("CharRNN", CharRNN),
                    ("CharDataset", CharDataset),
                    ("train_one_epoch", train_one_epoch),
                    ("evaluate", evaluate),
                    ("criterion", criterion),
                ]
                if obj is None
            ]
            if missing:
                self.skipTest("Missing from notebook: " + ", ".join(missing))
            self.test_model = CharRNN(vocab_size, embed_dim, hidden_dim).to(device)
            self.test_optimizer = torch.optim.Adam(self.test_model.parameters(), lr=0.003)
            self.tiny_loader = DataLoader(
                CharDataset(text[:400], seq_length=8),
                batch_size=4,
                shuffle=False,
                drop_last=True,
            )
            self.train_one_epoch = train_one_epoch
            self.evaluate = evaluate
            self.criterion = criterion

        def test_train_one_epoch_returns_finite_loss(self):
            loss = self.train_one_epoch(
                self.test_model, self.tiny_loader, self.test_optimizer, self.criterion
            )
            self.assertIsInstance(loss, float)
            self.assertFalse(math.isnan(loss))
            self.assertFalse(math.isinf(loss))
            self.assertGreater(loss, 0.0)

        def test_evaluate_returns_finite_loss(self):
            loss = self.evaluate(self.test_model, self.tiny_loader, self.criterion)
            self.assertIsInstance(loss, float)
            self.assertFalse(math.isnan(loss))
            self.assertFalse(math.isinf(loss))
            self.assertGreater(loss, 0.0)

        def test_training_updates_weights(self):
            before = [p.detach().clone() for p in self.test_model.parameters()]
            self.train_one_epoch(
                self.test_model, self.tiny_loader, self.test_optimizer, self.criterion
            )
            after = [p.detach() for p in self.test_model.parameters()]
            changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
            self.assertTrue(changed, "Expected at least one model parameter to change after training")

        def test_evaluate_does_not_change_weights(self):
            before = [p.detach().clone() for p in self.test_model.parameters()]
            self.evaluate(self.test_model, self.tiny_loader, self.criterion)
            after = [p.detach() for p in self.test_model.parameters()]
            for b, a in zip(before, after):
                self.assertTrue(torch.allclose(b, a))

    return TestTODO3


def reflection_score(nb_path: Path) -> dict[str, Any]:
    """Light check that students filled the reflection markdown cell."""
    nb = nbformat.read(nb_path, as_version=4)
    # Last markdown cell after the reflection prompt is where answers go
    answers = ""
    for i, cell in enumerate(nb.cells):
        src = "".join(cell.get("source", []))
        if "Part 8: Reflection Questions" in src:
            # look at following markdown cells
            for j in range(i + 1, len(nb.cells)):
                if nb.cells[j].cell_type == "markdown":
                    answers += "".join(nb.cells[j].get("source", []))
            break
    cleaned = answers.strip()
    # Require a short non-empty answer block (manual review still recommended)
    ok = len(cleaned) >= 80
    return {
        "name": "Reflection answers present (auto check)",
        "score": 5.0 if ok else 0.0,
        "max_score": 5.0,
        "output": (
            "Found reflection text (>= 80 characters)."
            if ok
            else "Reflection answer cell looks empty or too short. "
            "Write your answers in the markdown cell under Part 8."
        ),
        "visibility": "visible",
    }


def suite_to_gradescope(
    name: str,
    result: GradescopeTestResult,
    max_score: float,
) -> list[dict[str, Any]]:
    """Convert unittest results into per-test Gradescope entries."""
    tests: list[dict[str, Any]] = []
    # Collect outcomes
    failed = {id(t): (t, err) for t, err in result.failures + result.errors}
    skipped = {id(t): reason for t, reason in getattr(result, "skipped", [])}

    all_tests = list(result.successes)
    all_tests += [t for t, _ in result.failures + result.errors]
    all_tests += [t for t, _ in getattr(result, "skipped", [])]

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for t in all_tests:
        tid = id(t)
        if tid in seen:
            continue
        seen.add(tid)
        ordered.append(t)

    if not ordered:
        return [
            {
                "name": name,
                "score": 0.0,
                "max_score": max_score,
                "output": "No tests were collected/run.\n" + getattr(result, "stream_output", ""),
                "visibility": "visible",
            }
        ]

    per = max_score / len(ordered)
    for t in ordered:
        tid = id(t)
        test_name = f"{name}: {t.id().split('.')[-1]}"
        if tid in failed:
            _, err = failed[tid]
            tests.append(
                {
                    "name": test_name,
                    "score": 0.0,
                    "max_score": per,
                    "output": err,
                    "visibility": "visible",
                }
            )
        elif tid in skipped:
            tests.append(
                {
                    "name": test_name,
                    "score": 0.0,
                    "max_score": per,
                    "output": f"Skipped: {skipped[tid]}",
                    "visibility": "visible",
                }
            )
        else:
            tests.append(
                {
                    "name": test_name,
                    "score": per,
                    "max_score": per,
                    "output": "Passed",
                    "visibility": "visible",
                }
            )
    return tests


def main() -> int:
    env_sub = os.environ.get("AUTOGRADE_SUBMISSION")
    if env_sub:
        submission_dir = Path(env_sub)
    else:
        submission_dir = find_dir(SUBMISSION_CANDIDATES)

    if on_gradescope():
        results_dir = Path("/autograder/results")
    else:
        results_dir = LOCAL_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    tests_out: list[dict[str, Any]] = []
    score_total = 0.0
    max_total = 100.0

    try:
        nb_path = find_notebook(submission_dir)
    except Exception as exc:
        payload = {
            "score": 0.0,
            "output": str(exc),
            "tests": [
                {
                    "name": "Submission check",
                    "score": 0.0,
                    "max_score": 100.0,
                    "output": str(exc),
                    "visibility": "visible",
                }
            ],
        }
        (results_dir / "results.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    try:
        ns = exec_notebook(nb_path)
    except Exception:
        payload = {
            "score": 0.0,
            "output": "Failed to load/execute notebook:\n" + traceback.format_exc(),
            "tests": [
                {
                    "name": "Notebook execution",
                    "score": 0.0,
                    "max_score": 100.0,
                    "output": traceback.format_exc(),
                    "visibility": "visible",
                }
            ],
        }
        (results_dir / "results.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    # Point allocations: TODO1 30, TODO2 35, TODO3 30, reflection 5
    groups = [
        ("TODO 1 — encode & CharDataset", make_todo1_tests(ns), 30.0),
        ("TODO 2 — CharRNN", make_todo2_tests(ns), 35.0),
        ("TODO 3 — train_one_epoch & evaluate", make_todo3_tests(ns), 30.0),
    ]

    for title, case_cls, pts in groups:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(case_cls)
            result = run_suite(suite)
            group_tests = suite_to_gradescope(title, result, pts)
        except Exception:
            group_tests = [
                {
                    "name": title,
                    "score": 0.0,
                    "max_score": pts,
                    "output": traceback.format_exc(),
                    "visibility": "visible",
                }
            ]
        tests_out.extend(group_tests)

    try:
        tests_out.append(reflection_score(nb_path))
    except Exception:
        tests_out.append(
            {
                "name": "Reflection answers present (auto check)",
                "score": 0.0,
                "max_score": 5.0,
                "output": traceback.format_exc(),
                "visibility": "visible",
            }
        )

    score_total = sum(t["score"] for t in tests_out)
    cell_errors = ns.get("_cell_errors", [])
    header = (
        f"Graded notebook: {nb_path.name}\n"
        f"Executed code cells: {ns.get('_executed_cells', 0)}\n"
        f"Fixture corpus chars: {len(ns.get('text', '')):,}\n"
    )
    if cell_errors:
        header += "\nSome notebook cells raised errors during load:\n"
        for err in cell_errors[:8]:
            header += f"- cell {err['cell']}: {err['error']}\n"

    payload = {
        "score": round(score_total, 2),
        "output": header,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "tests": tests_out,
    }
    (results_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(header)
    print(f"Score: {payload['score']} / {max_total}")
    for t in tests_out:
        status = "PASS" if t["score"] >= t["max_score"] - 1e-9 else "FAIL"
        print(f"[{status}] {t['name']}: {t['score']:.2f}/{t['max_score']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
