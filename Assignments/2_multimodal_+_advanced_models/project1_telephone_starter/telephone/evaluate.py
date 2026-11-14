"""Score a finished run: how much of the seed survives each hop, and how well each output fits its input.

    python -m telephone.evaluate runs/<run>                # CLIP, sentence embeddings, CLAP
    python -m telephone.evaluate runs/<run> --metrics toy  # pixel/word overlap, no downloads

Writes metrics.csv (one row per hop) and drift.png (similarity to the seed vs. hop) into the run folder.
Columns:
    image_sim_to_seed   cosine similarity of each generated image to the seed image (G hops)
    text_sim_to_first   cosine similarity of each caption to the first caption in the chain (C and L hops)
    fit                 how well the output matches its own input: CLIP image-text cosine for C and G,
                        CLAP music-text cosine for M (raw cosines, not rescaled CLIPScore)

These are a minimum. Style metrics (your classifier, similarity to your style set), attribute checks and
concept/mood survival from the project spec go on top.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from . import media


class ToyMetrics:
    """Cheap stand-ins so the eval pipeline can be tested without models."""

    def image_image(self, a, b):
        x, y = (np.asarray(im.resize((16, 16)), dtype=np.float32).ravel() for im in (a, b))
        x, y = x - 127.5, y - 127.5
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

    def text_text(self, a, b):
        wa, wb = (set(re.findall(r"[a-z]+", t.lower())) for t in (a, b))
        return len(wa & wb) / max(len(wa | wb), 1)  # Jaccard word overlap

    def image_text(self, image, text):
        return None

    def music_text(self, audio, text):
        return None


class HFMetrics:
    """CLIP for images, a sentence-embedding model for text, CLAP for music. Frozen and off the shelf,
    so every member is scored by the same ruler."""

    def __init__(self, clip="openai/clip-vit-base-patch32", text="sentence-transformers/all-MiniLM-L6-v2",
                 clap="laion/clap-htsat-unfused"):
        import torch
        from transformers import AutoModel, AutoTokenizer, ClapModel, ClapProcessor, CLIPModel, CLIPProcessor
        self.torch = torch
        self.clip, self.clip_proc = CLIPModel.from_pretrained(clip).eval(), CLIPProcessor.from_pretrained(clip)
        self.st, self.st_tok = AutoModel.from_pretrained(text).eval(), AutoTokenizer.from_pretrained(text)
        self.clap_name = clap
        self.clap = None  # loaded on first use; only M hops need it

    def _unit(self, out):
        out = out if self.torch.is_tensor(out) else out.pooler_output
        return self.torch.nn.functional.normalize(out, dim=-1)

    def _image(self, im):
        with self.torch.no_grad():
            return self._unit(self.clip.get_image_features(**self.clip_proc(images=im, return_tensors="pt")))

    def image_image(self, a, b):
        return float(self._image(a) @ self._image(b).T)

    def image_text(self, image, text):
        with self.torch.no_grad():
            tok = self.clip_proc.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            return float(self._image(image) @ self._unit(self.clip.get_text_features(**tok)).T)

    def _sentence(self, text):
        with self.torch.no_grad():
            tok = self.st_tok([text], padding=True, truncation=True, return_tensors="pt")
            hidden = self.st(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            return self.torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)

    def text_text(self, a, b):
        return float(self._sentence(a) @ self._sentence(b).T)

    def music_text(self, audio, text):
        from transformers import ClapModel, ClapProcessor
        if self.clap is None:
            self.clap = ClapModel.from_pretrained(self.clap_name).eval()
            self.clap_proc = ClapProcessor.from_pretrained(self.clap_name)
        x = media.resample(audio.samples, audio.sample_rate, 48_000)
        with self.torch.no_grad():
            a = self._unit(self.clap.get_audio_features(
                **self.clap_proc.feature_extractor(x, sampling_rate=48_000, return_tensors="pt")))
            t = self._unit(self.clap.get_text_features(
                **self.clap_proc.tokenizer([text], padding=True, truncation=True, return_tensors="pt")))
        return float(a @ t.T)


def evaluate(run_dir, metrics="hf") -> list[dict]:
    run_dir = Path(run_dir)
    m = ToyMetrics() if metrics == "toy" else HFMetrics()
    records = [json.loads(line) for line in open(run_dir / "records.jsonl")]
    seed_file = next(run_dir.glob("hop_00_seed.*"))
    seed = media.load(seed_file)
    seed_image = seed if media.kind_of(seed_file) == "image" else None
    first_text = seed if isinstance(seed, str) else None

    rows = []
    for r in records:
        inp, out = media.load(r["input_path"]), media.load(r["output_path"])
        row = {k: r[k] for k in ("hop", "cycle", "member", "stage", "checkpoint_hash")}
        row.update(image_sim_to_seed=None, text_sim_to_first=None, fit=None)
        if r["stage"] in ("C", "L"):
            first_text = first_text or out
            row["text_sim_to_first"] = m.text_text(first_text, out)
        if r["stage"] == "G" and seed_image is not None:
            row["image_sim_to_seed"] = m.image_image(seed_image, out)
        if r["stage"] == "C":
            row["fit"] = m.image_text(inp, out)
        elif r["stage"] == "G":
            row["fit"] = m.image_text(out, inp)
        elif r["stage"] == "M":
            row["fit"] = m.music_text(out, inp)
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_drift(rows, run_dir / "drift.png")
    return rows


def plot_drift(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    series = (("G", "image_sim_to_seed", "G images vs. seed image", "o"),
              ("C", "text_sim_to_first", "C captions vs. first text", "s"),
              ("L", "text_sim_to_first", "L music descriptions vs. first text", "^"))
    for stage, col, label, marker in series:
        pts = [(r["hop"], r[col]) for r in rows if r["stage"] == stage and r[col] is not None]
        if pts:
            ax.plot(*zip(*pts), marker=marker, label=label)
    ax.set_xlabel("hop")
    ax.set_ylabel("similarity to the seed")
    ax.set_title("Drift from the seed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--metrics", choices=["hf", "toy"], default="hf")
    args = p.parse_args(argv)
    for row in evaluate(args.run_dir, args.metrics):
        print(row)
    print(f"Wrote {Path(args.run_dir) / 'metrics.csv'} and drift.png")


if __name__ == "__main__":
    main()
