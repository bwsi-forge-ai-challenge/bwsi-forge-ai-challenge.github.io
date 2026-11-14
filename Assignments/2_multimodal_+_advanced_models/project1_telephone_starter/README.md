# Project 1 starter: the multimodal telephone game

A small, working version of the whole pipeline: **image → caption → music → description → image**, with
the shared interface, the chain runner, per-call logging and basic drift metrics already in place. Every
model sits behind a common `Stage` interface and is chosen in a YAML config, so you can swap a base model for
another one, or for your fine-tuned checkpoint, without touching the chain code.

| Stage | Job | Default base model | Backend name |
| --- | --- | --- | --- |
| C | image → text | BLIP base (`Salesforce/blip-image-captioning-base`) | `blip` |
| M | text → music | MusicGen small (`facebook/musicgen-small`) | `musicgen` |
| L | music → text | CLAP zero-shot tags filled into a template (`laion/clap-htsat-unfused`) | `clap_tags` |
| G | text → image | SD-Turbo, 1 step (`stabilityai/sd-turbo`) | `sd_turbo` |

There are also four **toy** backends (`toy_captioner`, `toy_composer`, `toy_listener`, `toy_generator`)
that need no downloads and run a full chain in about a second. Use them to test your code before the real
models are ready.

## Quick start

```bash
pip install -r requirements.txt

# 1. Toy chain: seconds, no downloads. Member A's own loop, 3 cycles.
python -m telephone.chain --members configs/toy --seed-image seeds/demo.png --order A.C,A.M,A.L,A.G --cycles 3
python -m telephone.evaluate runs/<run folder> --metrics toy

# 2. Real base models (downloads about 7 GB the first time).
python -m telephone.chain --members configs/hf --seed-image seeds/demo.png --order A.C,A.M,A.L,A.G
python -m telephone.evaluate runs/<run folder>
```

The four base models together need about 10 GB of GPU memory. On a smaller GPU, a laptop or free Colab,
add `--low-memory`: each model is unloaded after its hop and reloaded when it is next needed, which is
slower but fits. MusicGen is by far the slowest stage, at roughly 5–10 minutes per 10-second clip on a
laptop CPU or Apple GPU, so use the toy backends while you develop.

## What a run produces

Each run gets its own folder under `runs/`:

```
runs/20261113-101500_demo_A.C-A.M-A.L-A.G_x1_s0/
  chain.json         the exact command, ordering, seed and member configs used
  records.jsonl      one JSON line per hop (the contract's per-call record)
  hop_00_seed.png    the seed
  hop_01_A.C.txt     member A's caption
  hop_02_A.M.wav     member A's music
  hop_03_A.L.txt     member A's description of the music
  hop_04_A.G.png     member A's new image
  metrics.csv        written by telephone.evaluate
  drift.png          similarity to the seed, hop by hop
```

A record looks like this:

```json
{"member": "A", "stage": "C", "backend": "blip", "model": "Salesforce/blip-image-captioning-base",
 "checkpoint": null, "checkpoint_hash": "hf:Salesforce/blip-image-captioning-base@82a3776...",
 "style_prompt": "", "hop": 1, "cycle": 1, "seed": 1, "wall_clock_s": 0.84,
 "input_path": "runs/.../hop_00_seed.png", "output_path": "runs/.../hop_01_A.C.txt"}
```

`checkpoint_hash` is the SHA-256 of your checkpoint folder when you set `checkpoint:`, or the Hugging Face
commit of the base weights when you don't.

## The interface contract

Defined in `telephone/media.py`. Change the constants there if your team agrees on something different.

| Type | Python object | Contract |
| --- | --- | --- |
| image | `PIL.Image` | RGB, 512 × 512 |
| text | `str` | at most 60 words |
| music | `telephone.Audio(samples, sample_rate)` | mono, 32 kHz, 10–30 s |

Every stage is called the same way, `stage.run(input, seed) -> output`. `run` seeds Python, NumPy and
PyTorch, calls your backend, then forces the output to meet the contract (resize, truncate, resample), so
one member's sloppy output can't crash the next member's stage.

## Orderings

An ordering is a comma-separated list of `MEMBER.STAGE` hops. Member names come from the config files
(`configs/hf/A.yaml` is member `A`).

```bash
--order A.C,A.M,A.L,A.G --cycles 5      # E2: member A's self-loop for 5 cycles
--order A.C,B.M,C.L,A.G --cycles 3      # E3: a cross-member chain, 3 rotations
--order B.C,C.M,A.L,B.G --cycles 3      # E3: a different ordering
--seed-input seeds/demo_text.txt --order A.M,A.L,A.G,A.C   # start from a seed text (E5)
```

The runner checks that each hop's output type matches the next hop's input type before anything runs.

## Configs: swapping models

Each member has one YAML file. For each stage you choose a backend and its settings:

```yaml
member: B
device: auto                      # cuda, then mps, then cpu
stages:
  C:
    backend: blip
    model: Salesforce/blip-image-captioning-base
    style_prompt: "a grainy black and white noir photograph of"
    checkpoint: members/B/lora_captioner     # add once you have fine-tuned
  M:
    backend: musicgen
    model: facebook/musicgen-medium          # a different base model: change one line
    duration: 15
```

This is how the required experiments map onto configs:

- **E1 (does fine-tuning work?):** three configs for the same member: no `style_prompt` and no `checkpoint`
  (zero-shot), `style_prompt` only (prompted), and `checkpoint` set (fine-tuned). `configs/hf/A.yaml` is the
  zero-shot variant and `configs/hf/B.yaml` the prompted variant.
- **E4 (who broke the message?):** add a member such as `A_base.yaml` that copies `A.yaml` without the
  checkpoints, then swap one hop at a time, e.g. `A.C,B.M,C.L,A.G` → `A_base.C,B.M,C.L,A.G`.

## Adding your own backend

Write a class, register it and import its module in `telephone/backends/__init__.py`. Here is the whole
shape of a music captioner backend:

```python
# telephone/backends/my_listener.py
from ..stage import Stage, pick_device, register

@register("my_listener")
class MyListener(Stage):
    kinds = ("L",)                      # which stages this backend can fill

    def load(self):                     # runs once, before the first call
        self.device = pick_device(self.device_request)
        self.model = ...                # load self.model_name, then your LoRA from self.checkpoint

    def generate(self, audio, seed):    # audio is a telephone.Audio at 32 kHz
        return "a caption in your style"
```

Then use `backend: my_listener` in a config. Extra keys in the config arrive in `self.params`.

`clap_tags` is a placeholder for L, not a model you can fine-tune: it picks the closest genre, mood and
instrument words with CLAP and fills them into a template, the "tags filled into templates" approach from
the Music Captioning lesson. Replace it with a trained captioner (LP-MusicCaps, Qwen2-Audio, or a frozen
CLAP/MERT encoder plus a small decoder).

## Metrics

`python -m telephone.evaluate <run>` scores every hop:

| Column | Hops | What it measures |
| --- | --- | --- |
| `image_sim_to_seed` | G | CLIP cosine similarity between the new image and the seed image |
| `text_sim_to_first` | C, L | Sentence-embedding cosine similarity to the first text in the chain |
| `fit` | C, G, M | How well the output matches its own input: CLIP image–text cosine (C, G), CLAP music–text cosine (M) |

These are raw cosines, not rescaled CLIPScore, so compare them only with each other. The project also
needs style metrics, attribute checks, concept and mood survival, and the human study. Add them in
`telephone/evaluate.py` or in your own scripts that read `records.jsonl`.

## Licenses and safety

MusicGen and SD-Turbo weights are licensed for non-commercial use, which covers coursework; note it in
your README. SD-Turbo ships without a safety checker. The project rules say to keep safety filters on for
G, so add one (for example the diffusers `StableDiffusionSafetyChecker`) or use a base model that includes
one before you run chains on new inputs.

## Layout

```
telephone/
  media.py          data types, the contract, save/load
  stage.py          Stage base class, backend registry, seeding, checkpoint hashing
  chain.py          the chain runner (python -m telephone.chain)
  evaluate.py       per-hop metrics and drift plot (python -m telephone.evaluate)
  backends/
    toy.py          no-download stand-ins for all four stages
    hf.py           BLIP, MusicGen, CLAP tags, SD-Turbo
configs/
  toy/              two toy members
  hf/               member A (zero-shot base models) and B (prompted noir style)
seeds/              a demo seed image and a demo seed text
```
