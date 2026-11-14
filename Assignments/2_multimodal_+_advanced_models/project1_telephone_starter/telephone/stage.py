"""The shared Stage interface and the backend registry.

A *stage* is one of the four jobs in the loop:
    C  Captioner  image -> text
    M  Composer   text  -> music
    L  Listener   music -> text
    G  Generator  text  -> image

A *backend* is a class that does one or more of those jobs with a particular model. Backends register
themselves by name, and a member's YAML config picks a backend for each stage, so swapping a model is a
config change, not a code change.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from . import media

STAGE_IO = {"C": ("image", "text"), "M": ("text", "music"), "L": ("music", "text"), "G": ("text", "image")}
STAGE_NAMES = {"C": "Captioner", "M": "Composer", "L": "Listener", "G": "Generator"}

_BACKENDS = {}


def register(name: str):
    """Class decorator: make a backend available to configs as `backend: <name>`."""
    def deco(cls):
        cls.backend = name
        _BACKENDS[name] = cls
        return cls
    return deco


def build_stage(member: str, kind: str, cfg: dict) -> "Stage":
    cfg = dict(cfg)
    name = cfg.pop("backend")
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Registered: {', '.join(sorted(_BACKENDS))}")
    cls = _BACKENDS[name]
    if kind not in cls.kinds:
        raise ValueError(f"Backend {name!r} can do stages {cls.kinds}, not {kind!r}")
    return cls(member=member, kind=kind, **cfg)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256_path(path: Path) -> str:
    """SHA-256 over a checkpoint file, or over every file in a checkpoint folder (sorted by name)."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        h.update(f.relative_to(path.parent if path.is_file() else path).as_posix().encode())
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def hub_revision(repo_id: str) -> str:
    """The commit hash of a cached Hugging Face model, which pins the exact base weights."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return "unknown"
    for filename in ("config.json", "model_index.json"):  # cached at .../snapshots/<commit>/<filename>
        path = try_to_load_from_cache(repo_id, filename)
        if isinstance(path, str):
            return Path(path).parent.name
    return "unknown"


class Stage:
    """Base class for every backend.

    Subclasses set `kinds` and implement `generate`; heavy models go in `load`, which runs once before
    the first call. `run` is the shared call signature from the interface contract.
    """

    kinds: tuple = ()
    backend: str = ""

    def __init__(self, member, kind, model=None, checkpoint=None, style_prompt="", device=None, **params):
        self.member = member
        self.kind = kind
        self.model_name = model
        self.checkpoint = checkpoint  # e.g. a LoRA folder from your fine-tuning run
        self.style_prompt = style_prompt
        self.device_request = device
        self.params = params
        self._loaded = False
        self._hash = None

    @property
    def name(self) -> str:
        return f"{self.member}.{self.kind}"

    @property
    def input_type(self) -> str:
        return STAGE_IO[self.kind][0]

    @property
    def output_type(self) -> str:
        return STAGE_IO[self.kind][1]

    def load(self):
        """Load weights. Override in backends that need a model."""

    def generate(self, inp, seed: int):
        raise NotImplementedError

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True

    def run(self, inp, seed: int):
        self.ensure_loaded()
        seed_everything(seed)
        return media.conform(self.generate(inp, seed), self.output_type)

    def checkpoint_hash(self) -> str:
        if self._hash is None:
            if self.checkpoint and Path(self.checkpoint).exists():
                self._hash = "sha256:" + sha256_path(Path(self.checkpoint))
            elif self.model_name:
                self._hash = f"hf:{self.model_name}@{hub_revision(self.model_name)}"
            else:  # no weights: hash the settings, so a changed setting shows up in the logs
                blob = json.dumps([self.backend, self.style_prompt, self.params], sort_keys=True)
                self._hash = "config:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self._hash

    def describe(self) -> dict:
        return {
            "member": self.member,
            "stage": self.kind,
            "backend": self.backend,
            "model": self.model_name,
            "checkpoint": self.checkpoint,
            "checkpoint_hash": self.checkpoint_hash(),
            "style_prompt": self.style_prompt,
        }
