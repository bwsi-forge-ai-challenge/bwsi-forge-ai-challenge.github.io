#!/usr/bin/env python3
"""
Gradescope autograder for multimodal_evaluation_assignment (submitted as a .py file)

Package lives at:
  Assignments/2_multimodal_+_advanced_models/multimodal_evaluation_assignment_autograder/

How it works
  * The submission is parsed with `ast`; the whole file is never run, so the student's
    model download, GPU code and plots are skipped.
  * The graded evaluation functions are exec'd one at a time on top of the PROVIDED code
    (provided.py, generated from the handout) and the reference versions of the other
    graded functions (reference_solution.py). A bug in one function therefore costs points
    only in its own section.
  * The functions are run on the grader's own local models: a grounded model that always
    looks, a model that never looks, and a partly grounded one, on two hidden test sets of
    simple generated images (the grader never needs the class photos). Results are compared
    with the reference implementation.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import signal
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

SOURCE_DIR = Path(__file__).resolve().parent
PROVIDED_SRC = (SOURCE_DIR / "provided.py").read_text(encoding="utf-8")
REFERENCE_SRC = (SOURCE_DIR / "reference_solution.py").read_text(encoding="utf-8")

LOCAL_RESULTS = SOURCE_DIR.parent / "_autograder_local_results"
SUBMISSION_CANDIDATES = [Path("/autograder/submission"), SOURCE_DIR.parent]

GRADED_FUNCS = {"normalize", "extract_answer", "run_condition", "reliance_gap", "change_rate",
                "flip_rate", "evaluate_model"}
SAFE_ASSIGN_NAMES = {"MODEL_INFO", "MY_ITEMS", "part3_reflection", "part4_reflection"}
ALLOWED_IMPORTS = {"json", "os", "random", "string", "time", "math", "statistics", "re", "collections",
                   "itertools", "functools", "copy", "typing", "numpy", "PIL"}
CONDITIONS = ["full", "blind", "mismatched", "counterfactual"]
CHECK_TIMEOUT_S = 120
ASK_DELAY_S = 0.003          # every grader model call sleeps this long
LOAD_DELAY_S = 0.05          # added to image loading in the timing check


# --------------------------------------------------------------------------- #
# The grader's own test images: simple drawn scenes, saved as files so that the
# provided load_image() opens them exactly as it opens the class photos
# --------------------------------------------------------------------------- #
GRADER_IMAGE_DIR = tempfile.mkdtemp(prefix="mm_eval_grader_images_")
SIZE = 384
COLORS = {
    "red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 160, 70),
    "yellow": (240, 200, 30), "purple": (140, 60, 180), "orange": (245, 130, 30),
}
GRID = [64, 192, 320]
R = 44
WORDS = ["cat", "dog", "sun", "tree", "fish", "bird", "car", "book", "hat", "cup"]
SLOTS = [(x, y) for y in GRID for x in GRID]


def _font(size):
    try:
        return ImageFont.load_default(size=size)            # Pillow >= 10.1
    except TypeError:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def draw_scene(scene):
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    for shape, color, x, y in scene.get("shapes", []):
        fill = COLORS[color]
        if shape == "circle":
            d.ellipse([x - R, y - R, x + R, y + R], fill=fill)
        elif shape == "square":
            d.rectangle([x - R, y - R, x + R, y + R], fill=fill)
        else:
            d.polygon([(x, y - R), (x - R, y + R), (x + R, y + R)], fill=fill)
    if scene.get("word"):
        d.text((SIZE / 2, SIZE / 2), scene["word"].upper(), fill="black", font=_font(96), anchor="mm")
    return img


def build_grader_set(seed: int, prefix: str) -> list[dict]:
    """30 items in the class test set's format, with drawn images saved to GRADER_IMAGE_DIR."""
    rng = random.Random(seed)
    colors = list(COLORS)
    items: list[dict] = []

    def add(category, question, answer, choices, scene, cf_scene, cf_answer):
        item_id = f"{category}-{sum(i['category'] == category for i in items)}"
        names = [f"{prefix}{item_id}.png", f"{prefix}{item_id}_cf.png"]
        for name, sc in zip(names, [scene, cf_scene]):
            draw_scene(sc).save(os.path.join(GRADER_IMAGE_DIR, name))
        items.append({"id": item_id, "category": category, "question": question, "answer": answer,
                      "choices": choices, "image": names[0], "cf_image": names[1], "cf_answer": cf_answer})

    for _ in range(6):
        c, c2, d1, d2 = rng.sample(colors, 4)
        (cx, cy), s1, s2 = rng.sample(SLOTS, 3)
        others = [("square", d1, *s1), ("triangle", d2, *s2)]
        add("color", "What color is the circle?", c, colors,
            {"shapes": [("circle", c, cx, cy)] + others}, {"shapes": [("circle", c2, cx, cy)] + others}, c2)
    for n in [1, 2, 3, 4, 5, 3]:
        n2 = n + 1 if n < 3 else n - 1
        color = rng.choice(colors)
        slots = rng.sample(SLOTS, 7)
        distractor = [("circle", rng.choice(colors), *slots[6])]
        squares = [("square", color, x, y) for x, y in slots[:6]]
        add("count", "How many squares are in the image?", str(n), [str(k) for k in range(10)],
            {"shapes": squares[:n] + distractor}, {"shapes": squares[:n2] + distractor}, str(n2))
    for present in [True, False, True, False, True, False]:
        slots = rng.sample(SLOTS, 3)
        base = [("circle", rng.choice(colors), *slots[0]), ("square", rng.choice(colors), *slots[1])]
        with_tri, without_tri = {"shapes": base + [("triangle", rng.choice(colors), *slots[2])]}, {"shapes": base}
        if present:
            add("existence", "Is there a triangle in the image?", "yes", ["yes", "no"], with_tri, without_tri, "no")
        else:
            add("existence", "Is there a triangle in the image?", "no", ["yes", "no"], without_tri, with_tri, "yes")
    for side in ["left", "right", "left", "right", "left", "right"]:
        y, other_y = rng.choice(GRID), rng.choice(GRID)
        cx, mirrored = (64, 320) if side == "left" else (320, 64)
        circle_color, square_color = rng.sample(colors, 2)
        square = ("square", square_color, 192, other_y)
        add("position", "Is the circle on the left or right?", side, ["left", "right"],
            {"shapes": [("circle", circle_color, cx, y), square]},
            {"shapes": [("circle", circle_color, mirrored, y), square]}, "right" if side == "left" else "left")
    for w, w2 in zip(rng.sample(WORDS, 6), rng.sample(WORDS, 10)[::-1]):
        if w2 == w:
            w2 = next(x for x in WORDS if x != w)
        add("text", "What word is written in the image?", w, WORDS, {"word": w}, {"word": w2}, w2)
    return items


GRADER_SETS = {"A": build_grader_set(0, "a_"), "B": build_grader_set(7, "b_")}


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
            "No .py file found in your submission. Submit firstname_lastname_model_eval.py." + hint
        )
    for p in pys:
        if p.name.endswith("_model_eval.py"):
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
    ns["IMAGE_DIR"] = GRADER_IMAGE_DIR
    exec(compile(REFERENCE_SRC, "reference_solution.py", "exec"), ns)
    return ns


PROVIDED_NAMES = set(base_namespace()) - GRADED_FUNCS


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
                elif node.name not in PROVIDED_NAMES and node.name != "ask":
                    self.helpers.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in SAFE_ASSIGN_NAMES or name == "EVAL_REPORT":
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
# The grader's local models
# --------------------------------------------------------------------------- #
class Recorder:
    """Wraps an ask function: adds a fixed delay and records every question asked."""

    def __init__(self, ask_fn: Callable[[Any, str], str]):
        self.ask_fn = ask_fn
        self.questions: list[str] = []

    def __call__(self, image, question):
        self.questions.append(question)
        time.sleep(ASK_DELAY_S)
        return self.ask_fn(image, question)


def grader_models(ns: dict[str, Any], items: list[dict]) -> dict[str, Callable]:
    perfect = ns["make_perfect_model"](items)
    guess = ns["guessing_ask"]

    def partial(image, question):   # looks only for color and word questions
        q = question.lower()
        return perfect(image, question) if ("color" in q or "word" in q) else guess(image, question)

    return {"perfect": perfect, "guessing": guess, "partial": partial}


def item_sets(ns: dict[str, Any]) -> dict[str, list[dict]]:
    """The grader's two hidden test sets (fresh copies)."""
    return copy.deepcopy(GRADER_SETS)


def _strip(results: list[dict]) -> list[tuple]:
    return [(r["id"], r["category"], r["prediction"], r["extracted"], bool(r["correct"])) for r in results]


def _close(a: Any, b: float) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(float(a) - b) < 1e-9


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
Check = Callable[[Submission], None]


# ---- Answer scoring ------------------------------------------------------ #
def t_normalize_basic(sub):
    f = sub.namespace("normalize")["normalize"]
    assert f("The circle is RED.") == ["the", "circle", "is", "red"]
    assert f("  Blue  ") == ["blue"]
    assert f("") == []


def t_normalize_numbers_punct(sub):
    f = sub.namespace("normalize")["normalize"]
    assert f("Three squares!") == ["3", "squares"]
    assert f("yes,it's there") == ["yes", "it", "s", "there"]
    assert f("Ten, (maybe) eleven? Zero.") == ["10", "maybe", "eleven", "0"]
    assert f("left/right") == ["left", "right"]


def t_extract_basic(sub):
    f = sub.namespace("extract_answer")["extract_answer"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    assert f("Red.", colors) == "red"
    assert f("The circle is purple", colors) == "purple"
    assert f("I can't tell", colors) is None
    assert f("", colors) is None


def t_extract_hard(sub):
    f = sub.namespace("extract_answer")["extract_answer"]
    colors = ["red", "blue", "green"]
    digits = [str(k) for k in range(10)]
    assert f("red or blue", colors) is None
    assert f("redder", colors) is None
    assert f("Blue. Definitely BLUE!", colors) == "blue"
    assert f("There are three.", digits) == "3"
    assert f("2 or 3", digits) is None
    assert f("Yes, there is.", ["yes", "no"]) == "yes"


# ---- run_condition ------------------------------------------------------- #
def _check_run(sub, conditions):
    ns = sub.namespace("run_condition")
    ref = base_namespace()
    for set_name, items in item_sets(ns).items():
        ns["add_mismatches"](items)
        for model_name, model in grader_models(ns, items).items():
            for cond in conditions:
                got = ns["run_condition"](Recorder(model), copy.deepcopy(items), cond)
                want = ref["run_condition"](model, copy.deepcopy(items), cond)
                assert isinstance(got, list) and len(got) == len(items), (
                    f"{cond}: expected one result per item")
                missing = {"id", "category", "prediction", "extracted", "correct", "seconds"} - set(got[0])
                assert not missing, f"results are missing keys {sorted(missing)}"
                assert _strip(got) == _strip(want), f"{cond} results differ ({set_name} set, {model_name} model)"


def t_run_full(sub):
    _check_run(sub, ["full"])


def t_run_blind(sub):
    _check_run(sub, ["blind"])


def t_run_mismatch_cf(sub):
    _check_run(sub, ["mismatched", "counterfactual"])


def t_run_prompt_and_timing(sub):
    ns = sub.namespace("run_condition")
    items = copy.deepcopy(GRADER_SETS["A"])
    ns["add_mismatches"](items)
    rec = Recorder(ns["make_perfect_model"](items))
    load, blank = ns["load_image"], ns["blank_image"]

    def slow_load(name):
        time.sleep(LOAD_DELAY_S)
        return load(name)

    def slow_blank():
        time.sleep(LOAD_DELAY_S)
        return blank()

    ns["load_image"], ns["blank_image"] = slow_load, slow_blank
    for cond in ["full", "blind"]:
        rec.questions.clear()
        got = ns["run_condition"](rec, items[:6], cond)
        assert rec.questions == [i["question"] + ns["PROMPT_SUFFIX"] for i in items[:6]], (
            "ask_fn must be called once per item with question + PROMPT_SUFFIX, in order")
        secs = [r["seconds"] for r in got]
        assert all(isinstance(s, float) for s in secs), "seconds must be a float"
        assert min(secs) >= ASK_DELAY_S * 0.9, "seconds must measure the ask_fn call"
        assert statistics.median(secs) < LOAD_DELAY_S * 0.6, "time only the ask_fn call, not loading the image"


# ---- Reliance metrics ---------------------------------------------------- #
def _mk(ext, ok):
    return {"extracted": ext, "correct": ok, "category": "x", "id": "x", "prediction": str(ext), "seconds": 0.1}


def t_reliance_gap(sub):
    f = sub.namespace("reliance_gap")["reliance_gap"]
    full = [_mk("a", True), _mk("b", True), _mk(None, False), _mk("c", True), _mk("d", True)]
    blind = [_mk(None, False), _mk("b", True), _mk(None, False), _mk(None, False), _mk("x", False)]
    assert _close(f(full, blind), 0.8 - 0.2)
    assert _close(f(blind, full), 0.2 - 0.8), "the gap is full minus blind and can be negative"
    assert _close(f(full, full), 0.0)


def t_change_rate(sub):
    f = sub.namespace("change_rate")["change_rate"]
    full = [_mk("red", True), _mk("blue", True), _mk(None, False), _mk("3", True)]
    mism = [_mk("green", False), _mk("blue", False), _mk(None, False), _mk("4", False)]
    assert _close(f(full, mism), 0.5)
    assert _close(f(full, full), 0.0)
    full2 = [{**_mk("red", True), "prediction": "Red."}, _mk(None, False)]
    mism2 = [{**_mk("red", True), "prediction": "red"}, _mk("blue", False)]
    assert _close(f(full2, mism2), 0.5), "compare the extracted answers, not the raw text"


def t_flip_rate(sub):
    f = sub.namespace("flip_rate")["flip_rate"]
    assert _close(f([_mk("a", True), _mk("b", False), _mk(None, False), _mk("c", True)]), 0.5)
    assert _close(f([_mk("a", True)] * 3), 1.0)
    assert _close(f([_mk(None, False)] * 7), 0.0)


# ---- evaluate_model ------------------------------------------------------ #
REPORT_KEYS = {"full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate",
               "median_latency_s", "category_accuracy", "results"}


def _evaluate_both(sub, model_names):
    ns = sub.namespace("evaluate_model")
    ref = base_namespace()
    for set_name, items in item_sets(ns).items():
        models = grader_models(ns, items)
        for name in model_names:
            got = ns["evaluate_model"](Recorder(models[name]), copy.deepcopy(items))
            want = ref["evaluate_model"](models[name], copy.deepcopy(items))
            yield set_name, name, got, want


def t_eval_structure(sub):
    for _, _, got, want in _evaluate_both(sub, ["perfect"]):
        assert isinstance(got, dict) and set(got) == REPORT_KEYS, f"return exactly the keys {sorted(REPORT_KEYS)}"
        assert set(got["results"]) == set(CONDITIONS), "results must map each condition to its list"
        for cond in CONDITIONS:
            assert _strip(got["results"][cond]) == _strip(want["results"][cond]), f"results['{cond}'] is wrong"


def t_eval_metrics(sub):
    for set_name, name, got, want in _evaluate_both(sub, ["perfect", "guessing", "partial"]):
        for key in ["full_accuracy", "blind_accuracy", "reliance_gap", "change_rate", "flip_rate"]:
            assert _close(got[key], want[key]), f"{key} is wrong ({set_name} set, {name} model)"


def t_eval_category_latency(sub):
    for set_name, name, got, want in _evaluate_both(sub, ["partial"]):
        cats = got["category_accuracy"]
        assert isinstance(cats, dict) and set(cats) == set(want["category_accuracy"])
        assert all(_close(cats[c], v) for c, v in want["category_accuracy"].items()), "category_accuracy is wrong"
        lat = got["median_latency_s"]
        assert isinstance(lat, (int, float)) and ASK_DELAY_S * 0.9 <= lat < ASK_DELAY_S + 0.03, (
            "median_latency_s should be the median seconds of the full run")


# ---- Your test category -------------------------------------------------- #
ITEM_KEYS = {"id", "category", "question", "answer", "choices", "image", "cf_image", "cf_answer"}


def _my_items(sub):
    items = sub.value("MY_ITEMS")
    assert isinstance(items, list) and len(items) >= 4, "MY_ITEMS needs at least 4 items"
    assert all(isinstance(i, dict) and ITEM_KEYS <= set(i) for i in items), f"every item needs {sorted(ITEM_KEYS)}"
    return items


def t_my_items_form(sub):
    items = _my_items(sub)
    ns = base_namespace()
    assert len({i["id"] for i in items}) == len(items), "ids must be unique"
    assert len({i["category"] for i in items}) == 1, "use one category"
    assert items[0]["category"] not in {i["category"] for i in ns["TEST_SET"]}, "pick a new category"
    for i in items:
        assert isinstance(i["question"], str) and i["question"].strip()
        for key in ("image", "cf_image"):
            assert isinstance(i[key], str) and i[key].strip(), f"{key} must be a file name"


def t_my_items_valid(sub):
    """The photos are not submitted, so only the item definitions can be checked."""
    items = _my_items(sub)
    for i in items:
        assert isinstance(i["choices"], list) and len(i["choices"]) >= 2
        assert i["answer"] in i["choices"] and i["cf_answer"] in i["choices"]
        assert i["answer"] != i["cf_answer"], "each counterfactual must change the answer"
        assert i["image"] != i["cf_image"], "image and cf_image must be different photos"
    assert len({i["answer"] for i in items}) >= 2, "use at least two different answers"


# ---- Model choice and evaluation card ------------------------------------ #
WRITTEN_FIELDS = ["summary", "grounding_verdict", "claimed_vs_measured", "limitations_of_this_test",
                  "deployment_recommendation"]


def word_count(s: Any) -> int:
    return len(re.findall(r"\w+", s)) if isinstance(s, str) else 0


def t_model_info(sub):
    info = sub.value("MODEL_INFO")
    for key in ["model_id", "url", "parameters", "developer", "license", "why_this_model", "claimed_benchmark"]:
        assert isinstance(info.get(key), str) and info[key].strip(), f"MODEL_INFO['{key}'] is empty"
    assert re.fullmatch(r"[\w.\-]+/[\w.\-]+", info["model_id"].strip()), "model_id should be organization/model-name"


def _card_fields(sub) -> dict[str, Any]:
    node = sub.assigns.get("EVAL_REPORT")
    assert node is not None and isinstance(node.value, ast.Dict), "EVAL_REPORT must be a dictionary literal"
    fields = {}
    for k, v in zip(node.value.keys, node.value.values):
        if isinstance(k, ast.Constant) and k.value in WRITTEN_FIELDS:
            fields[k.value] = v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else None
    return fields


def t_card_filled(sub):
    fields = _card_fields(sub)
    for key in WRITTEN_FIELDS:
        text = fields.get(key)
        assert isinstance(text, str) and text.strip().upper() != "TODO", f"EVAL_REPORT['{key}'] is not filled in"
        assert word_count(text) >= 10, f"EVAL_REPORT['{key}'] is too short"


def t_card_content(sub):
    fields = _card_fields(sub)
    verdict = (fields.get("grounding_verdict") or "").lower()
    assert any(k in verdict for k in ("gap", "change", "flip", "blind", "%")) or re.search(r"\d", verdict), (
        "grounding_verdict should cite the reliance numbers")
    deploy = (fields.get("deployment_recommendation") or "").lower()
    assert any(k in deploy for k in ("yes", "no", "not", "would", "should")), "give a recommendation"


# ---- Reflections --------------------------------------------------------- #
def t_p3(sub):
    r = sub.value("part3_reflection")
    assert word_count(r) >= 40, "part3_reflection: write at least 40 words"
    assert any(c in r.lower() for c in ("color", "colour", "count", "direction", "state", "text", "word")), (
        "part3_reflection should name the categories")


def t_p4(sub):
    r = sub.value("part4_reflection")
    low = r.lower() if isinstance(r, str) else ""
    assert word_count(r) >= 50, "part4_reflection: write at least 50 words"
    assert "model a" in low or "model b" in low, "part4_reflection should compare with Model A / Model B"
    assert "blind" in low or "blank" in low, "part4_reflection should discuss the blind condition"


GROUPS: list[tuple[str, list[tuple[str, float, Check]]]] = [
    (
        "Answer scoring (normalize, extract_answer)",
        [
            ("normalize basics", 1, t_normalize_basic),
            ("normalize numbers and punctuation", 1, t_normalize_numbers_punct),
            ("extract_answer basics", 1, t_extract_basic),
            ("extract_answer hedges, whole words, digits", 1, t_extract_hard),
        ],
    ),
    (
        "run_condition",
        [
            ("full condition", 1, t_run_full),
            ("blind condition", 1, t_run_blind),
            ("mismatched and counterfactual conditions", 1, t_run_mismatch_cf),
            ("prompt suffix and timing", 1, t_run_prompt_and_timing),
        ],
    ),
    (
        "Reliance metrics",
        [
            ("reliance_gap", 1, t_reliance_gap),
            ("change_rate", 1, t_change_rate),
            ("flip_rate", 1, t_flip_rate),
        ],
    ),
    (
        "evaluate_model",
        [
            ("report structure and results", 1, t_eval_structure),
            ("reliance numbers on the grader's models", 1, t_eval_metrics),
            ("category accuracy and latency", 1, t_eval_category_latency),
        ],
    ),
    (
        "Your test category (MY_ITEMS)",
        [
            ("well-formed items in a new category", 1, t_my_items_form),
            ("valid counterfactuals", 1, t_my_items_valid),
        ],
    ),
    (
        "Model choice and evaluation card",
        [
            ("MODEL_INFO filled in", 0.5, t_model_info),
            ("evaluation card fields written", 1, t_card_filled),
            ("evaluation card cites results and recommends", 0.5, t_card_content),
        ],
    ),
    (
        "Reflections",
        [
            ("part3_reflection", 1, t_p3),
            ("part4_reflection", 1, t_p4),
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
