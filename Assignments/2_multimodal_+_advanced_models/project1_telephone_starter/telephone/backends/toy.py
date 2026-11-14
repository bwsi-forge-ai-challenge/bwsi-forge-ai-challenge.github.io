"""Toy backends: tiny hand-written stand-ins for all four stages. No downloads, no GPU, seconds per chain.

Use them to test your interface, chain runner and eval scripts before real models are ready. They also
show the telephone game in miniature: the captioner names colors and brightness, the composer turns the
main color into a drone pitch and brightness into tempo, the listener recovers what it can, and the
generator paints whatever words survive. The accent color has no way through the music, so it dies at the
music hop, just like literal concepts do in the real pipeline.
"""

import re

import numpy as np
from PIL import Image, ImageDraw

from .. import media
from ..stage import Stage, register

PALETTE = {
    "red": (210, 45, 45), "orange": (235, 135, 35), "yellow": (235, 215, 60), "green": (60, 160, 75),
    "blue": (45, 95, 215), "purple": (135, 65, 185), "pink": (235, 135, 180), "brown": (120, 80, 45),
    "black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128),
}
# Each color gets a drone pitch, in semitones above 220 Hz (A3).
SEMITONES = {"red": 0, "orange": 2, "yellow": 4, "green": 5, "blue": 7, "purple": 9, "pink": 11,
             "brown": -3, "black": -5, "white": 12, "gray": -1}
BRIGHT_WORDS = {"bright", "lively", "sunny", "happy", "fast"}


def colors_in(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w in PALETTE]


@register("toy_captioner")
class ToyCaptioner(Stage):
    kinds = ("C",)

    def generate(self, image, seed):
        px = np.asarray(image.resize((64, 64)), dtype=np.float32)
        names = list(PALETTE)
        centers = np.array([PALETTE[n] for n in names], dtype=np.float32)
        nearest = np.argmin(((px[:, :, None, :] - centers) ** 2).sum(-1), axis=-1)
        counts = np.bincount(nearest.ravel(), minlength=len(names))
        main, accent = (names[i] for i in np.argsort(counts)[::-1][:2])
        mood = "bright" if px.mean() / 255 > 0.45 else "dark"
        prefix = f"{self.style_prompt} " if self.style_prompt else ""
        return f"{prefix}A {mood} picture dominated by {main}, with touches of {accent}."


@register("toy_composer")
class ToyComposer(Stage):
    kinds = ("M",)

    def generate(self, text, seed):
        rng = np.random.default_rng(seed)
        colors = colors_in(text) or ["gray"]
        bright = bool(BRIGHT_WORDS & set(text.lower().split()))
        bpm = 140 if bright else 75
        root = 220 * 2 ** (SEMITONES[colors[0]] / 12)
        chord = [0, 4, 7, 12] if bright else [0, 3, 7, 12]  # major if bright, minor if dark
        seconds = float(self.params.get("duration", 10))
        sr = media.SAMPLE_RATE
        t = np.arange(int(seconds * sr)) / sr
        audio = 0.3 * np.sin(2 * np.pi * root * t)  # the drone carries the main color
        beat = int(sr * 60 / bpm)
        for start in range(0, len(t), beat):
            freq = 2 * root * 2 ** (rng.choice(chord) / 12)
            n = min(beat, len(t) - start)
            env = np.exp(-6 * np.arange(n) / sr)
            audio[start:start + n] += 0.25 * env * np.sin(2 * np.pi * freq * t[:n])
        return media.Audio((audio / np.abs(audio).max() * 0.8).astype(np.float32), sr)


@register("toy_listener")
class ToyListener(Stage):
    kinds = ("L",)

    def generate(self, audio, seed):
        bpm = media.estimate_bpm(audio)
        spectrum = np.abs(np.fft.rfft(audio.samples))
        freqs = np.fft.rfftfreq(len(audio.samples), 1 / audio.sample_rate)
        band = (freqs > 150) & (freqs < 480)
        peak = freqs[band][np.argmax(spectrum[band])]
        semis = 12 * np.log2(peak / 220)
        color = min(SEMITONES, key=lambda c: abs(SEMITONES[c] - semis))
        mood = "lively, bright" if bpm > 105 else "slow, dark"
        prefix = f"{self.style_prompt} " if self.style_prompt else ""
        return f"{prefix}A {mood} tune at about {bpm} BPM, droning on a deep {color} tone."


@register("toy_generator")
class ToyGenerator(Stage):
    kinds = ("G",)

    def generate(self, text, seed):
        rng = np.random.default_rng(seed)
        colors = colors_in(f"{text} {self.style_prompt}") or ["gray"]
        main = np.array(PALETTE[colors[0]], dtype=np.float32)
        accent = np.array(PALETTE[colors[1] if len(colors) > 1 else colors[0]], dtype=np.float32)
        if not BRIGHT_WORDS & set(re.findall(r"[a-z]+", text.lower())):
            main, accent = main * 0.55, accent * 0.55
        size = media.IMAGE_SIZE
        ramp = np.linspace(0, 1, size)[:, None, None]
        img = Image.fromarray((main * (1 - 0.4 * ramp) + 0.4 * ramp * accent).repeat(size, axis=1).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for _ in range(6):  # a few seeded blobs, so different seeds give different pictures
            x, y, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(20, 80))]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in accent))
        return img
