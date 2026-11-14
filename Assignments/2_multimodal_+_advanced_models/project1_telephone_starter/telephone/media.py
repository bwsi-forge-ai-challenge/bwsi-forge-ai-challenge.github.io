"""The three data types that move between stages, and the interface contract they must meet.

Every stage reads and writes exactly one of these:
    image -> PIL.Image, RGB, IMAGE_SIZE x IMAGE_SIZE
    text  -> str, UTF-8, at most MAX_WORDS words
    music -> Audio, mono float32 at SAMPLE_RATE, MIN_SECONDS to MAX_SECONDS long

Change the constants below if your team agrees on a different contract.
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly

IMAGE_SIZE = 512
MAX_WORDS = 60
SAMPLE_RATE = 32_000
MIN_SECONDS, MAX_SECONDS = 10, 30

EXTENSIONS = {"image": ".png", "text": ".txt", "music": ".wav"}


@dataclass
class Audio:
    samples: np.ndarray  # 1D float32 in [-1, 1]
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def resample(samples: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return samples.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(samples, sr_to // g, sr_from // g).astype(np.float32)


def conform_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        side = min(img.size)  # center-crop to a square, then resize
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return img


def conform_text(text: str) -> str:
    return " ".join(str(text).split()[:MAX_WORDS])


def conform_audio(audio: Audio) -> Audio:
    x = np.asarray(audio.samples, dtype=np.float32)
    if x.ndim > 1:  # (channels, samples) or (samples, channels) -> mono
        x = x.mean(axis=0 if x.shape[0] < x.shape[1] else 1)
    x = resample(x, audio.sample_rate, SAMPLE_RATE)
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    if len(x) < MIN_SECONDS * SAMPLE_RATE:
        x = np.pad(x, (0, MIN_SECONDS * SAMPLE_RATE - len(x)))
    return Audio(np.clip(x, -1.0, 1.0), SAMPLE_RATE)


def conform(obj, kind: str):
    """Force a stage's output to meet the contract, so a sloppy stage can't break the next one."""
    if kind == "image":
        return conform_image(obj)
    if kind == "text":
        return conform_text(obj)
    if kind == "music":
        return conform_audio(obj)
    raise ValueError(f"Unknown data type {kind!r}")


def save(obj, path_stem: Path, kind: str) -> Path:
    path_stem = Path(path_stem)
    path = path_stem.parent / (path_stem.name + EXTENSIONS[kind])  # not with_suffix: "A.C" isn't an extension
    if kind == "image":
        obj.save(path)
    elif kind == "text":
        path.write_text(obj + "\n", encoding="utf-8")
    else:
        wavfile.write(path, obj.sample_rate, (obj.samples * 32767).astype(np.int16))
    return path


def load(path: Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return conform_image(Image.open(path))
    if suffix == ".txt":
        return conform_text(path.read_text(encoding="utf-8"))
    if suffix == ".wav":
        sr, x = wavfile.read(path)
        if np.issubdtype(x.dtype, np.integer):
            x = x / np.iinfo(x.dtype).max
        return conform_audio(Audio(x.T if x.ndim > 1 else x, sr))
    raise ValueError(f"Don't know how to load {path}")


def kind_of(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    return {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
            ".txt": "text", ".wav": "music"}[suffix]


def estimate_bpm(audio: Audio, lo: int = 60, hi: int = 200) -> int:
    """Rough tempo from the autocorrelation of an onset envelope. Good enough for a template caption;
    expect the usual half/double-tempo mistakes."""
    hop = 512
    x = audio.samples
    frames = len(x) // hop
    rms = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    onset = np.maximum(np.diff(rms), 0)
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    fps = audio.sample_rate / hop
    lags = np.arange(int(fps * 60 / hi), int(fps * 60 / lo) + 1)
    best = lags[np.argmax(ac[lags])]
    return int(round(60 * fps / best))
