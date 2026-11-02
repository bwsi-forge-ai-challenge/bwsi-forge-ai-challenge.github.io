"""Builds the diffusion assignment from one source of truth.

Usage (from anywhere): python3 build_assignment.py

Writes into Assignments/2_multimodal_+_advanced_models/:
  diffusion_assignment.ipynb            student handout
  diffusion_assignment_solution.ipynb   solution
  diffusion_assignment_autograder/provided.py            the PROVIDED code the grader needs, verbatim
  diffusion_assignment_autograder/reference_solution.py  reference graded functions

Code cells given as code(student, solution) differ between the two notebooks. Snippets that the
autograder also needs are defined once below and reused, so the grader always runs exactly the
code students were given.
"""
import json
import os

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG_DIR = os.path.join(OUT, "diffusion_assignment_autograder")
cells = []


def md(src):
    cells.append(("markdown", src.strip("\n"), None))


def code(src, solution=None):
    cells.append(("code", src.strip("\n"), None if solution is None else solution.strip("\n")))


# ===========================================================================
# Shared snippets (notebook + autograder)
# ===========================================================================
IMPORTS_SRC = '''
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

SEED = 0
'''

SCENE_SRC = '''
# Toy picture helpers (PROVIDED)
def make_scene(size=96):
    """Draw a small picture (sky, sun, hills, a house) as a PIL image."""
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):                                     # sky
        f = y / size
        d.line([(0, y), (size, y)], fill=(int(120 + 90 * f), int(180 + 50 * f), 255))
    s = size / 96
    d.ellipse([60 * s, 10 * s, 82 * s, 32 * s], fill=(255, 220, 40))        # sun
    d.polygon([(0, 70 * s), (30 * s, 48 * s), (60 * s, 70 * s), (size, 60 * s), (size, size), (0, size)],
              fill=(60, 160, 80))                                              # hills
    d.rectangle([22 * s, 56 * s, 52 * s, 82 * s], fill=(200, 80, 60))         # house
    d.polygon([(18 * s, 56 * s), (37 * s, 40 * s), (56 * s, 56 * s)], fill=(120, 50, 40))
    d.rectangle([32 * s, 66 * s, 42 * s, 82 * s], fill=(250, 240, 200))       # door
    return img


def to_array(img):
    """PIL image -> float array with values in [-1, 1], shape (height, width, 3)."""
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def to_pil(x):
    """Float array with values near [-1, 1] -> PIL image (values outside the range are clipped)."""
    return Image.fromarray(np.clip((np.asarray(x) + 1.0) * 127.5, 0, 255).astype(np.uint8))
'''

MODEL_SRC = '''
# The small network that will learn to predict noise (PROVIDED)
def timestep_embedding(t, dim):
    """Turn step numbers (a LongTensor of shape (B,)) into vectors of length `dim`."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TinyUNet(nn.Module):
    """A small U-Net: model(x_t, t) -> predicted noise, same shape as x_t. Input: (B, 1, 28, 28)."""

    def __init__(self, ch=64, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.down1 = nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1)               # 28 -> 14
        self.down2 = nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1)           # 14 -> 7
        self.mid = nn.Conv2d(ch * 4, ch * 4, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)    # 7 -> 14
        self.up2 = nn.ConvTranspose2d(ch * 4, ch, 4, stride=2, padding=1)        # 14 -> 28
        self.out = nn.Conv2d(ch * 2, 1, 3, padding=1)
        widths = [ch, ch * 2, ch * 4, ch * 4, ch * 2, ch]
        self.norms = nn.ModuleList([nn.GroupNorm(8, w) for w in widths])
        self.t_proj = nn.ModuleList([nn.Linear(t_dim, w) for w in widths])       # tells every layer the step
        self.act = nn.SiLU()

    def block(self, k, conv, h, emb):
        h = self.norms[k](conv(h)) + self.t_proj[k](emb)[:, :, None, None]
        return self.act(h)

    def forward(self, x, t):
        emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h0 = self.block(0, self.inc, x, emb)
        h1 = self.block(1, self.down1, h0, emb)
        h2 = self.block(2, self.down2, h1, emb)
        h2 = self.block(3, self.mid, h2, emb)
        u1 = self.block(4, self.up1, h2, emb)
        u2 = self.block(5, self.up2, torch.cat([u1, h1], dim=1), emb)
        return self.out(torch.cat([u2, h0], dim=1))
'''

# ---- reference / solution versions of the graded functions ----------------
SOL_ALPHA = '''
def make_alpha_bar(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    return np.cumprod(1.0 - betas)
'''

SOL_NOISE = '''
def add_noise(x0, abar_t, eps):
    return abar_t ** 0.5 * x0 + (1.0 - abar_t) ** 0.5 * eps
'''

SOL_PREDICT = '''
def predict_clean(x_t, eps_hat, abar_t):
    return (x_t - (1.0 - abar_t) ** 0.5 * eps_hat) / abar_t ** 0.5
'''

SOL_LOSS = '''
def noise_prediction_loss(model, x0, t, eps, alpha_bar):
    abar_t = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = add_noise(x0, abar_t, eps)
    eps_hat = model(x_t, t)
    return ((eps - eps_hat) ** 2).mean()
'''

SOL_SAMPLE = '''
def ddim_sample(model, x_T, alpha_bar, steps):
    timesteps = np.linspace(len(alpha_bar) - 1, 0, steps).round().astype(int)
    x = x_T
    for i, t in enumerate(timesteps):
        abar_t = alpha_bar[int(t)]
        abar_prev = alpha_bar[int(timesteps[i + 1])] if i + 1 < len(timesteps) else 1.0
        t_batch = torch.full((x.shape[0],), int(t), dtype=torch.long, device=x.device)
        eps_hat = model(x, t_batch)
        x0_hat = predict_clean(x, eps_hat, abar_t).clamp(-1, 1)
        x = add_noise(x0_hat, abar_prev, eps_hat)
    return x
'''

SOL_SLERP = '''
def slerp(z0, z1, alpha):
    cos = (z0 * z1).sum() / (z0.norm() * z1.norm())
    omega = torch.acos(cos.clamp(-1.0, 1.0))
    return (torch.sin((1 - alpha) * omega) * z0 + torch.sin(alpha * omega) * z1) / torch.sin(omega)
'''

# ===========================================================================
# Notebook
# ===========================================================================
md("""
# Assignment: Noise to Picture

**Unit:** Deep Dive: Vision-Language Models | **Companion reading:** Generating Images by Removing Noise: Diffusion
**Suggested time:** 2 to 2.5 hours

---

## What is this assignment trying to do?

In the lesson you saw that a diffusion model is trained by **destroying** pictures with noise and learning to predict that noise, and that it makes new pictures by starting from pure noise and **removing** it a little at a time. In this assignment you will do both halves yourself:

1. **Part 1:** Destroy a picture with noise, using the diffusion formula from the lesson.
2. **Part 2:** Recover a picture from its noise, and see why a perfect noise prediction matters.
3. **Part 3:** Train a **tiny diffusion model** from scratch on a few thousand small images. It takes a few minutes on a free GPU.
4. **Part 4:** Write the **sampling loop** that turns pure noise into brand-new pictures, and compare few steps with many steps.
5. **Part 5:** Morph between two generated pictures by moving smoothly between their starting noise.
6. **Part 6:** Meet a big pretrained diffusion model (SD-Turbo), write your own prompts, and pick a favorite.
7. **Part 7:** Reflect on responsible use.

Every part ends with a short written **reflection**. The reflections are a big part of your grade, because the goal is to understand what you are seeing, not just to make the code run.

### How this assignment is graded

Your functions are **autograded**: the autograder imports your functions and runs them on its **own** small test models and inputs, so it never needs your trained model or a GPU. That means:

- Keep every function **name and signature** exactly as given.
- Your functions must use the model only through the `model` argument they receive, never through a global variable.
- Code outside the graded functions (training, plots, printing) is not run by the autograder.

Your written reflections, your prompts, and your favorite picture's settings are also checked, and may be read by staff.

### How to work

- Fill in every cell marked `# TODO`. Most parts have a **check** cell that tells you whether your code is right.
- Write reflections as Python strings in the cells provided (there is a template in each one).
- Run every cell top to bottom before you submit.
""")

md("""
---
## Part 0: Setup

**Use a GPU.** In Colab: **Runtime → Change runtime type → T4 GPU**. Then run the next cell. It imports what we need and picks the GPU if there is one.
""")

code('''
# Colab already has PyTorch, NumPy, Pillow, and Matplotlib, so there is nothing to install yet.
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

SEED = 0

import matplotlib.pyplot as plt

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
if DEVICE == "cpu":
    print("No GPU found. Part 3 will be slow on a CPU, so switch to a T4 GPU if you can.")
''')

md("""
### A toy picture (PROVIDED)

For Parts 1 and 2 we use a small drawn picture. `make_scene()` draws it, `to_array()` turns it into numbers between -1 and 1 (the range diffusion models use), and `to_pil()` turns numbers back into a picture. Just run the next two cells.
""")

code(SCENE_SRC)

code('''
scene = make_scene(96)
x0 = to_array(scene)                     # the clean picture, shape (96, 96, 3), values in [-1, 1]
print("Picture shape:", x0.shape, "| smallest value:", x0.min(), "| largest value:", x0.max())
display(scene.resize((192, 192)))
''')

md("""
---
## Part 1: Destroying a Picture with Noise

A diffusion model learns by looking at pictures that have been mixed with random noise. In the lesson, the **shortcut formula** gives the noisy picture at any step $t$ directly:

$$x_t = \\sqrt{\\bar\\alpha_t}\\; x_0 + \\sqrt{1 - \\bar\\alpha_t}\\; \\epsilon$$

- $x_0$ is the clean picture and $\\epsilon$ is a picture of random noise of the same shape.
- $\\bar\\alpha_t$ ("alpha bar") is a number between 0 and 1 for each step $t$. Near 1, almost all of the picture is kept. Near 0, almost nothing is kept.
- $\\bar\\alpha_t$ comes from a **noise schedule**: small numbers $\\beta_t$ say how much noise to add at each step, $\\alpha_t = 1 - \\beta_t$, and $\\bar\\alpha_t$ is the running product $\\alpha_1 \\alpha_2 \\cdots \\alpha_t$.

### TODO 1: `make_alpha_bar(T, beta_start, beta_end)`

Build the schedule. Make `T` evenly spaced $\\beta$ values from `beta_start` to `beta_end` (hint: `np.linspace`), compute $\\alpha = 1 - \\beta$, and return the running product (hint: `np.cumprod`) as a NumPy array of length `T`. Entry `alpha_bar[t]` belongs to step number `t` (counting from 0).
""")

code('''
def make_alpha_bar(T=1000, beta_start=1e-4, beta_end=0.02):
    """Return a NumPy array of T numbers: alpha_bar for every step."""
    # TODO 1
    raise NotImplementedError("TODO 1: make_alpha_bar")
''', SOL_ALPHA)

code('''
# Check TODO 1 (PROVIDED)
_ab = make_alpha_bar(1000)
assert len(_ab) == 1000, "make_alpha_bar(T) should return T numbers"
assert abs(_ab[0] - 0.9999) < 1e-6, "alpha_bar[0] should be 1 - beta_start"
assert all(_ab[i] > _ab[i + 1] for i in range(999)), "alpha_bar should keep falling"
assert _ab[-1] < 1e-3, "by the last step almost no picture should be left"
print(f"TODO 1 passed. alpha_bar goes from {_ab[0]:.4f} (nearly all picture) to {_ab[-1]:.1e} (nearly all noise).")
''')

md("""
### TODO 2: `add_noise(x0, abar_t, eps)`

Write the shortcut formula. It should return `sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps`.

**Hint:** write the square root as `** 0.5` (for example `abar_t ** 0.5`). That works for plain numbers, NumPy arrays, *and* PyTorch tensors, and you will use this same function with all three.
""")

code('''
def add_noise(x0, abar_t, eps):
    """Mix the clean picture x0 with noise eps, as if t steps of noise had been added."""
    # TODO 2
    raise NotImplementedError("TODO 2: add_noise")
''', SOL_NOISE)

code('''
# Check TODO 2 (PROVIDED)
_x = np.random.default_rng(1).uniform(-1, 1, (8, 8, 3))
_e = np.random.default_rng(2).standard_normal((8, 8, 3))
assert np.allclose(add_noise(_x, 1.0, _e), _x), "with abar = 1 nothing should change"
assert np.allclose(add_noise(_x, 0.0, _e), _e), "with abar = 0 only noise should be left"
assert np.allclose(add_noise(_x, 0.25, _e), 0.5 * _x + (0.75 ** 0.5) * _e), "check the two weights"
assert torch.allclose(add_noise(torch.tensor(_x), 0.25, torch.tensor(_e)), torch.tensor(0.5 * _x + (0.75 ** 0.5) * _e)), \\
    "add_noise should also work on PyTorch tensors (use ** 0.5)"
print("TODO 2 passed.")
''')

md("""
### Watch the picture disappear (PROVIDED)

The next cell builds the schedule for 1000 steps, draws one batch of random noise, and shows the picture at six different steps. The curve shows how much of the picture is kept ($\\bar\\alpha_t$) at every step.
""")

code('''
T = 1000
alpha_bar = make_alpha_bar(T)
rng = np.random.default_rng(SEED)
eps = rng.standard_normal(x0.shape)

shown = [0, 50, 100, 250, 500, 999]
fig, axes = plt.subplots(1, len(shown), figsize=(16, 3.2))
for ax, t in zip(axes, shown):
    ax.imshow(to_pil(add_noise(x0, alpha_bar[t], eps)))
    ax.set_title(f"t = {t}\\nalpha_bar = {alpha_bar[t]:.3f}")
    ax.axis("off")
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(alpha_bar)
plt.xlabel("step t")
plt.ylabel("alpha_bar (how much picture is kept)")
plt.show()
''')

md("""
### Part 1 reflection

Look at the six noisy pictures. Around which step does the scene stop being recognizable? What is $\\bar\\alpha_t$ there? In your own words, explain what a **small** $\\bar\\alpha_t$ means for the picture. Write at least 40 words, and use the words *noise* and *signal* (the part of the picture that is left).
""")

code('''
part1_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 2: Recovering the Picture from Its Noise

The network in a diffusion model does not try to draw a clean picture directly. It looks at a noisy picture and tries to **predict the noise** in it. Why is that enough? Because if you know the noise, you can undo the shortcut formula. Start from

$$x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1 - \\bar\\alpha_t}\\, \\epsilon$$

and solve for $x_0$, replacing the true noise $\\epsilon$ with the network's guess $\\hat\\epsilon$:

$$\\hat{x}_0 = \\frac{x_t - \\sqrt{1 - \\bar\\alpha_t}\\; \\hat\\epsilon}{\\sqrt{\\bar\\alpha_t}}$$

### TODO 3: `predict_clean(x_t, eps_hat, abar_t)`

Return $\\hat{x}_0$. As in TODO 2, use `** 0.5` for square roots so it also works on tensors.
""")

code('''
def predict_clean(x_t, eps_hat, abar_t):
    """Given a noisy picture and a guess of its noise, return the guessed clean picture."""
    # TODO 3
    raise NotImplementedError("TODO 3: predict_clean")
''', SOL_PREDICT)

code('''
# Check TODO 3 (PROVIDED)
for _a in (0.9, 0.5, 0.05):
    _xt = add_noise(_x, _a, _e)
    assert np.allclose(predict_clean(_xt, _e, _a), _x), f"with the true noise you should get x0 back exactly (abar = {_a})"
assert torch.allclose(predict_clean(torch.tensor(add_noise(_x, 0.5, _e)), torch.tensor(_e), 0.5), torch.tensor(_x))
print("TODO 3 passed.")
''')

md("""
### True noise versus a slightly wrong guess (PROVIDED)

For two steps, the next cell recovers the picture twice: once with the true noise (perfect), and once with an **imperfect** guess that is 70% right and 30% random. The numbers show the average pixel error of each recovered picture.
""")

code('''
rng2 = np.random.default_rng(5)
wrong = 0.7 * eps + 0.3 * rng2.standard_normal(eps.shape)       # an imperfect noise prediction

fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
for row, t in zip(axes, [250, 750]):
    x_t = add_noise(x0, alpha_bar[t], eps)
    perfect = predict_clean(x_t, eps, alpha_bar[t])
    imperfect = predict_clean(x_t, wrong, alpha_bar[t])
    err = np.abs(imperfect - x0).mean()
    for ax, img, title in zip(row, [x_t, perfect, imperfect],
                              [f"noisy x_t (t={t})", "recovered, true noise", f"recovered, imperfect\\nerror = {err:.2f}"]):
        ax.imshow(to_pil(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    print(f"t = {t}: alpha_bar = {alpha_bar[t]:.4f} | error with true noise = {np.abs(perfect - x0).mean():.4f} | error with imperfect noise = {err:.3f}")
plt.show()
''')

md("""
### Part 2 reflection

With the true noise the picture came back exactly. With the imperfect guess it did not. Compare the errors at $t = 250$ and $t = 750$: which one was worse, and why do you think that is? (Look at the size of $\\bar\\alpha_t$ and at the division by $\\sqrt{\\bar\\alpha_t}$ in the formula.) Then explain in a few sentences why this makes it hard for a model to jump from pure noise to a clean picture in a single step. Write at least 50 words, and use the words *error* (or *mistake*) and *step*.
""")

code('''
part2_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 3: Train a Tiny Diffusion Model

Now we teach a real network to predict noise. We will use small 28 x 28 grayscale pictures of clothing (the Fashion-MNIST collection). If it cannot be downloaded, the next cell draws simple shapes instead. Either way the pictures are scaled to values between -1 and 1.
""")

code('''
# The training pictures (PROVIDED)
def shapes_dataset(n):
    """Fallback pictures: a bright circle, square, or triangle on a dark background."""
    rng = np.random.default_rng(SEED)
    imgs = []
    for _ in range(n):
        img = Image.new("L", (28, 28), 0)
        d = ImageDraw.Draw(img)
        cx, cy, r = rng.integers(9, 19), rng.integers(9, 19), rng.integers(4, 9)
        kind = rng.integers(3)
        if kind == 0:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        elif kind == 1:
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=255)
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=255)
        imgs.append(np.asarray(img))
    return torch.tensor(np.stack(imgs)).float().div(127.5).sub(1).unsqueeze(1)


def load_images(n=60000):
    try:
        from torchvision import datasets
        ds = datasets.FashionMNIST("data", train=True, download=True)
        return ds.data[:n].float().div(127.5).sub(1).unsqueeze(1), "Fashion-MNIST"
    except Exception as exc:
        print("Could not load Fashion-MNIST (", type(exc).__name__, "). Drawing shapes instead.")
        return shapes_dataset(min(n, 20000)), "drawn shapes"


def show_images(x, ncols=8, title=""):
    """Show a batch of (N, 1, 28, 28) pictures with values in [-1, 1] as a grid."""
    x = x.detach().cpu().clamp(-1, 1)
    nrows = (len(x) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 * ncols, 1.2 * nrows + 0.4), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < len(x):
            ax.imshow(x[k, 0], cmap="gray", vmin=-1, vmax=1)
    fig.suptitle(title)
    plt.show()


DATA, DATA_NAME = load_images()
DATA = DATA.to(DEVICE)
print(f"Loaded {len(DATA)} pictures from {DATA_NAME}; shape {tuple(DATA.shape)}; values from {DATA.min():.0f} to {DATA.max():.0f}")
show_images(DATA[:16], title="Some training pictures")
''')

md("""
### The network (PROVIDED)

`TinyUNet` is a small U-Net (see the lesson): it takes a noisy picture `x_t` and the step number `t`, and returns its guess of the noise, `model(x_t, t)`. The step number tells the network how noisy its input is. You do not need to change this cell.
""")

code(MODEL_SRC)

md("""
### The training idea

For every training step, we:

1. take a batch of clean pictures `x0`,
2. pick a random step `t` for each picture and draw random noise `eps`,
3. build the noisy pictures with `add_noise`,
4. ask the network to predict the noise, and
5. score the guess with the **mean squared error** against the true noise, then nudge the weights to make that error smaller.

### TODO 4: `noise_prediction_loss(model, x0, t, eps, alpha_bar)`

Write steps 3 to 5 for one batch.

- `x0` has shape `(B, 1, 28, 28)`, `t` is a `LongTensor` of shape `(B,)`, and `eps` has the same shape as `x0`.
- `alpha_bar` is a **tensor** of length `T`. Each picture has its own step, so use `alpha_bar[t].view(-1, 1, 1, 1)` to get one number per picture that lines up with the pixels.
- Build `x_t` with your `add_noise`, get the guess with `model(x_t, t)`, and return the mean squared error `((eps - eps_hat) ** 2).mean()` as a single number (a tensor).
""")

code('''
def noise_prediction_loss(model, x0, t, eps, alpha_bar):
    """Mean squared error between the true noise and the model's guess, for one batch."""
    # TODO 4
    raise NotImplementedError("TODO 4: noise_prediction_loss")
''', SOL_LOSS)

code('''
# Check TODO 4 (PROVIDED)
_ab_t = torch.tensor(make_alpha_bar(1000), dtype=torch.float32)
_xb = torch.rand(4, 1, 8, 8) * 2 - 1
_tb = torch.tensor([0, 100, 500, 999])
_eb = torch.randn_like(_xb)


def _oracle(x_t, t):                    # a "perfect" model that knows the clean pictures
    a = _ab_t[t].view(-1, 1, 1, 1)
    return (x_t - a ** 0.5 * _xb) / (1 - a) ** 0.5


assert noise_prediction_loss(_oracle, _xb, _tb, _eb, _ab_t).item() < 1e-4, \\
    "a perfect model should get a loss near 0 (check how you pick abar for each picture)"
_conv = nn.Conv2d(1, 1, 3, padding=1)
_loss = noise_prediction_loss(lambda x, t: _conv(x), _xb, _tb, _eb, _ab_t)
assert _loss.dim() == 0 and _loss.item() > 0, "return a single number"
_loss.backward()
assert _conv.weight.grad is not None, "the loss must keep the gradient so training can work"
print("TODO 4 passed. Loss of a random model:", round(_loss.item(), 3))
''')

md("""
### Train it (PROVIDED)

This cell trains the network with your loss function. On a free T4 GPU it takes **roughly 3 to 5 minutes** (on a CPU it would take far too long, so it only does one pass). While it runs, think about the question in the reflection below.
""")

code('''
ALPHA_BAR = torch.tensor(alpha_bar, dtype=torch.float32, device=DEVICE)

model = TinyUNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
EPOCHS = 10 if DEVICE == "cuda" else 1
BATCH = 128
losses = []

start = time.time()
for epoch in range(EPOCHS):
    order = torch.randperm(len(DATA), device=DEVICE)
    for i in range(0, len(DATA) - BATCH + 1, BATCH):
        x0_batch = DATA[order[i:i + BATCH]]
        t = torch.randint(0, T, (BATCH,), device=DEVICE)
        eps_batch = torch.randn_like(x0_batch)
        loss = noise_prediction_loss(model, x0_batch, t, eps_batch, ALPHA_BAR)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}/{EPOCHS}: average loss {np.mean(losses[-100:]):.4f}   ({time.time() - start:.0f} s)")

model.eval()
plt.figure(figsize=(6, 3))
plt.plot(np.convolve(losses, np.ones(50) / 50, mode="valid"))
plt.xlabel("training step")
plt.ylabel("loss (smoothed)")
plt.show()
''')

md("""
### Part 3 reflection

Describe the loss curve: where did it start, how did it change, and did it flatten out? The loss is the average squared error of the **noise** prediction. Why is this a sensible number to minimize, even though we want to generate *pictures*? Say in your own words what the network has learned to do. Write at least 50 words, and use the word *noise*.
""")

code('''
part3_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 4: Generating New Pictures

To generate, start from a batch of pure noise and walk backward. At each step the network guesses the noise, we use **your** `predict_clean` to guess the clean picture, and then we step to a slightly less noisy picture. This is the **DDIM** sampler, a fast way to take big steps.

### How the sampler works

Pick `steps` steps spread evenly from the noisiest (`t = T - 1`) down to the cleanest (`t = 0`). For each step `t`:

1. Ask the model for its noise guess `eps_hat` (the model needs the step number as a `LongTensor` with one entry per picture).
2. Guess the clean picture: `x0_hat = predict_clean(x, eps_hat, abar_t)`, then **clamp it to the range [-1, 1]** so it stays a valid picture.
3. Move to the next, less noisy step: `x = add_noise(x0_hat, abar_prev, eps_hat)`. Here `abar_prev` is `alpha_bar` at the **next** step in your list. For the very last step, use `abar_prev = 1.0`, which means "no noise left", so the final `x` is just `x0_hat`.

### TODO 5: `ddim_sample(model, x_T, alpha_bar, steps)`

- `x_T` is the starting noise (a tensor), `alpha_bar` is a tensor of length `T`, and `steps` is how many steps to take.
- Use `timesteps = np.linspace(len(alpha_bar) - 1, 0, steps).round().astype(int)` for the step numbers (it counts down from the noisiest).
- Return the final `x`.
""")

code('''
def ddim_sample(model, x_T, alpha_bar, steps):
    """Turn the noise x_T into pictures by removing noise in `steps` steps."""
    # TODO 5
    raise NotImplementedError("TODO 5: ddim_sample")
''', SOL_SAMPLE)

code('''
# Check TODO 5 on a perfect model that knows the answer (PROVIDED)
_target = torch.rand(2, 1, 8, 8) * 2 - 1


def _oracle2(x_t, t):
    assert t.dtype == torch.long and t.shape == (x_t.shape[0],), "pass t as a LongTensor with one entry per picture"
    a = _ab_t[t].view(-1, 1, 1, 1)
    return (x_t - a ** 0.5 * _target) / (1 - a) ** 0.5


for _steps in (1, 4, 20):
    _out = ddim_sample(_oracle2, torch.randn(2, 1, 8, 8), _ab_t, _steps)
    assert _out.shape == _target.shape and torch.allclose(_out, _target, atol=1e-3), \\
        f"with a perfect model, {_steps} steps should recover the target exactly"
_wild = ddim_sample(lambda x, t: 50 * torch.ones_like(x), torch.randn(2, 1, 8, 8), _ab_t, 5)
assert _wild.min() >= -1 and _wild.max() <= 1, "clamp x0_hat to [-1, 1]"
print("TODO 5 passed.")
''')

md("""
### Generate and compare steps (PROVIDED)

We make one batch of starting noise and run your sampler with different numbers of steps **from the same noise**. The time is the time for your sampler to turn that batch of noise into pictures.
""")

code('''
g = torch.Generator().manual_seed(SEED)
x_T = torch.randn(16, 1, 28, 28, generator=g).to(DEVICE)
sample_times = {}
with torch.no_grad():
    for n_steps in [2, 5, 10, 50]:
        t0 = time.time()
        pics = ddim_sample(model, x_T, ALPHA_BAR, n_steps)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        sample_times[n_steps] = time.time() - t0
        show_images(pics, ncols=8, title=f"{n_steps} steps  ({sample_times[n_steps]:.2f} s)")

# Same noise, same steps: do you get the same pictures?
with torch.no_grad():
    again = ddim_sample(model, x_T, ALPHA_BAR, 50)
    first = ddim_sample(model, x_T, ALPHA_BAR, 50)
print("Largest difference between two runs from the same noise:", (again - first).abs().max().item())
''')

md("""
### Part 4 reflection

Compare the rows. What changed when you used more steps: the overall shape, the fine detail, or both? Was there a point where more steps stopped helping, and what did it cost in time? Also look at the last number printed: you ran the sampler twice from the **same** starting noise. What happened, and why? (Think about what is random in the process and what is not.) Write at least 50 words, and use the words *step* and *noise*.
""")

code('''
part4_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 5: Morphing Between Two Pictures

Every generated picture comes from its own starting noise. If you slide smoothly from one starting noise to another, the pictures should morph. But you cannot just average the two noises. The noise the network was trained on has a particular "size", and an average of two noises is smaller. **Spherical interpolation (slerp)** slides along the surface of a sphere so the size stays right.

$$\\text{slerp}(z_0, z_1, \\alpha) = \\frac{\\sin((1-\\alpha)\\,\\omega)}{\\sin \\omega}\\, z_0 + \\frac{\\sin(\\alpha\\,\\omega)}{\\sin \\omega}\\, z_1, \\qquad \\cos\\omega = \\frac{z_0 \\cdot z_1}{\\lVert z_0 \\rVert\\, \\lVert z_1 \\rVert}$$

### TODO 6: `slerp(z0, z1, alpha)`

- `z0` and `z1` are tensors of the same shape (treat each as one long vector), and `alpha` is a number from 0 to 1. At `alpha = 0` you get `z0`, and at `alpha = 1` you get `z1`.
- Find the angle: `cos = (z0 * z1).sum() / (z0.norm() * z1.norm())`, then `omega = torch.acos(cos.clamp(-1, 1))`.
- Return the formula above using `torch.sin`.
""")

code('''
def slerp(z0, z1, alpha):
    """Spherical interpolation between two noise tensors."""
    # TODO 6
    raise NotImplementedError("TODO 6: slerp")
''', SOL_SLERP)

code('''
# Check TODO 6 (PROVIDED)
_z0, _z1 = torch.randn(1, 28, 28), torch.randn(1, 28, 28)
assert torch.allclose(slerp(_z0, _z1, 0.0), _z0, atol=1e-5) and torch.allclose(slerp(_z0, _z1, 1.0), _z1, atol=1e-5), \\
    "alpha = 0 should give z0 and alpha = 1 should give z1"
_mid_norm = slerp(_z0, _z1, 0.5).norm().item()
_avg_norm = (0.5 * _z0 + 0.5 * _z1).norm().item()
assert abs(_mid_norm - _z0.norm().item()) < 0.15 * _z0.norm().item(), "slerp should keep the size of the noise"
print(f"TODO 6 passed. Noise size at the midpoint: slerp {_mid_norm:.1f} vs plain average {_avg_norm:.1f} (the endpoints are about {_z0.norm().item():.1f}).")
''')

md("""
### Morph! (PROVIDED)

The next cell picks two starting noises, then generates the pictures along the path between them twice: once with your `slerp` and once with a plain average. Change `SEED_A` and `SEED_B` to morph between different pictures.
""")

code('''
SEED_A, SEED_B = 3, 11
gA, gB = torch.Generator().manual_seed(SEED_A), torch.Generator().manual_seed(SEED_B)
zA = torch.randn(1, 28, 28, generator=gA).to(DEVICE)
zB = torch.randn(1, 28, 28, generator=gB).to(DEVICE)

alphas = [k / 7 for k in range(8)]
path_slerp = torch.stack([slerp(zA, zB, a) for a in alphas])
path_avg = torch.stack([(1 - a) * zA + a * zB for a in alphas])
with torch.no_grad():
    show_images(ddim_sample(model, path_slerp, ALPHA_BAR, 50), ncols=8, title="Morph with slerp")
    show_images(ddim_sample(model, path_avg, ALPHA_BAR, 50), ncols=8, title="Morph with a plain average")
''')

md("""
### Part 5 reflection

Describe what you see along each morph. Does the picture change smoothly, and is anything odd in the middle frames? Compare the slerp row with the plain-average row, and explain why the plain average behaves differently (look at the noise sizes printed in the TODO 6 check). Write at least 40 words, and use the word *noise*.
""")

code('''
part5_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 6: Meet a Big Diffusion Model

Your tiny model makes 28 x 28 gray pictures. Real systems use the same idea at a much larger scale, and steer it with text. **SD-Turbo** is a small version of Stable Diffusion, trained to make a 512 x 512 color picture in just 1 to 4 steps, so it fits on a free GPU. The next cell downloads it (about 2.5 GB, a minute or two) and defines `generate(prompt, seed, steps)`.
""")

code('''
# Colab: install the library for SD-Turbo, then load the model (PROVIDED)
# !pip install -q -U diffusers transformers accelerate
from diffusers import AutoPipelineForText2Image

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16").to(DEVICE)
pipe.set_progress_bar_config(disable=True)


def generate(prompt, seed=0, steps=1):
    """One 512 x 512 picture. The seed fixes the starting noise; SD-Turbo works best with 1 to 4 steps."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0, generator=generator).images[0]


display(generate("a lighthouse on a cliff at sunrise, digital painting", seed=0, steps=1).resize((256, 256)))
''')

md("""
### TODO 7: Write four prompts that grow

Pick a subject you like (a pet, a place, a robot mascot, a dessert, anything). Write **four prompts about the same subject**, getting more detailed each time:

1. The bare subject, like `"a robot"`.
2. Add one idea (a mood, a color, or an action).
3. Add a style (for example `"flat vector illustration"` or `"oil painting"`).
4. Add lighting, setting, and composition as well.

Store them in `MY_PROMPTS`, in order from shortest to longest. Each one must be different, and the last must be at least 3 words longer than the first.
""")

code('''
# TODO 7
MY_PROMPTS = [
    # "...",
    # "...",
    # "...",
    # "...",
]
''', '''
MY_PROMPTS = [
    "a robot",
    "a friendly robot mascot",
    "a friendly robot mascot holding a basketball, flat vector illustration",
    "a friendly robot mascot holding a basketball in a bright gym, soft shadows, flat vector illustration, centered",
]
''')

code('''
# Check TODO 7, then generate your four pictures with the same seed (PROVIDED)
assert isinstance(MY_PROMPTS, list) and len(MY_PROMPTS) >= 4, "write at least 4 prompts"
assert all(isinstance(p, str) and len(p.split()) >= 2 for p in MY_PROMPTS), "every prompt needs at least 2 words"
assert len(set(MY_PROMPTS)) == len(MY_PROMPTS), "the prompts must all be different"
assert len(MY_PROMPTS[-1].split()) >= len(MY_PROMPTS[0].split()) + 3, "the last prompt should be at least 3 words longer than the first"

fig, axes = plt.subplots(1, len(MY_PROMPTS), figsize=(4 * len(MY_PROMPTS), 4.6))
for ax, prompt in zip(np.atleast_1d(axes), MY_PROMPTS):
    ax.imshow(generate(prompt, seed=SEED, steps=1))
    ax.set_title(prompt, fontsize=8, wrap=True)
    ax.axis("off")
plt.show()
print("TODO 7 passed.")
''')

md("""
### TODO 8: Pick your favorite

Try different seeds (and 1, 2, or 4 steps) with your best prompt until you find a picture you like. Then record its settings in `FAVORITE`:

- `prompt`: copy one of your `MY_PROMPTS` **exactly**,
- `seed`: a whole number (0 or more) and `steps`: 1, 2, 3, or 4,
- `why`: at least 10 words on why you picked this one.
""")

code('''
# TODO 8
FAVORITE = {
    "prompt": "",
    "seed": 0,
    "steps": 1,
    "why": "",
}
''', '''
FAVORITE = {
    "prompt": "a friendly robot mascot holding a basketball in a bright gym, soft shadows, flat vector illustration, centered",
    "seed": 7,
    "steps": 2,
    "why": "The robot has a clear face, the basketball is easy to see, and the colors are bright and clean.",
}
''')

code('''
# Check TODO 8, then show and save your favorite (PROVIDED)
assert FAVORITE["prompt"] in MY_PROMPTS, "copy one of your MY_PROMPTS exactly"
assert isinstance(FAVORITE["seed"], int) and FAVORITE["seed"] >= 0
assert FAVORITE["steps"] in (1, 2, 3, 4)
assert len(FAVORITE["why"].split()) >= 10, "write at least 10 words in 'why'"
best = generate(FAVORITE["prompt"], seed=FAVORITE["seed"], steps=FAVORITE["steps"])
best.save("my_favorite.png")
display(best.resize((384, 384)))
print("TODO 8 passed. Saved my_favorite.png (use the Files panel on the left if you want to download it).")
''')

md("""
### Part 6 reflection

Describe what each added detail did to the picture. Which words clearly changed the result, and which seemed to be ignored? Did you see any mistakes (odd hands, garbled text, wrong counts, strange shapes), and what do you think caused them? Then compare SD-Turbo with your tiny model: name **two** things the big model can do that yours cannot. Write at least 60 words, and use the words *prompt* and *detail* (or *style*).
""")

code('''
part6_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Part 7: Responsible Use

Models like this learned from huge collections of pictures gathered from the web, and they can make very realistic images. Being able to make something does not mean you should.

### Part 7 reflection

Pick **one** of these and answer it: (a) describe a prompt you would refuse to run and why; (b) explain who could be hurt by a realistic fake picture of a real person, and how; or (c) say how you would label AI-made pictures when you share them. Explain what you would do and why. Write at least 50 words, and use at least one of these ideas: *consent*, *permission*, *fake*, *label*, *harm*, *copyright*.
""")

code('''
part7_reflection = """
TODO: your answer here
"""
''')

md("""
---
## Submitting

1. Make sure every `TODO` is complete, every check cell prints a passing message, and all seven reflections (`part1_reflection` to `part7_reflection`) are written.
2. Run all cells from top to bottom (**Runtime → Run all** in Colab).
3. Download the notebook as a Python file (**File → Download → Download .py**) and rename it `firstname_lastname_diffusion.py`.
4. Submit that `.py` file on Gradescope (see the assignment page on the course website for the link).

Your pictures are not submitted: the autograder checks your functions, your prompts, and your written answers.
""")

# ===========================================================================
SOLUTION_TEXT = {
    'part1_reflection = """\nTODO: your answer here\n"""':
        'part1_reflection = """\nSample answer (your numbers will differ): the scene was still easy to recognize at t = 100, '
        'where alpha_bar was about 0.9, but by t = 500 the house and the sun were buried in noise and alpha_bar had dropped '
        'to about 0.08. A small alpha_bar means the picture is multiplied by a small number, so almost all that is left is '
        'noise and very little signal remains. By t = 999 the image is pure random noise.\n"""',
    'part2_reflection = """\nTODO: your answer here\n"""':
        'part2_reflection = """\nSample answer: with the true noise the picture came back exactly, because the formula just '
        'undoes the mixing. With the imperfect guess the error was small at t = 250 (about 0.3) and huge at t = 750 (about 6). '
        'At t = 750 alpha_bar is tiny, so dividing by its square root multiplies every mistake in the noise guess by a large '
        'number. This is why a model cannot jump from pure noise to a clean picture in one step: it would have to be perfect. '
        'Many smaller steps only need to fix a little noise at a time.\n"""',
    'part3_reflection = """\nTODO: your answer here\n"""':
        'part3_reflection = """\nSample answer: the loss started high, around 1, dropped quickly in the first epoch, and then '
        'slowly flattened out near 0.05 to 0.1. Minimizing the error of the noise prediction is sensible because if the '
        'network can tell how much of an image is noise, then removing that noise gives back the picture. The network learned '
        'to look at a noisy picture and the step number and say which part is noise and which part is clothing.\n"""',
    'part4_reflection = """\nTODO: your answer here\n"""':
        'part4_reflection = """\nSample answer: with 2 steps the pictures were blurry blobs with the right overall shape, with 10 '
        'steps the edges and details got sharper, and with 50 steps they looked only a little better than 10 but took '
        'about five times as long. The difference between two runs from the same starting noise was 0.0, because the only '
        'random thing is the starting noise and every step after it is a fixed calculation, so the same noise always leads '
        'to the same picture.\n"""',
    'part5_reflection = """\nTODO: your answer here\n"""':
        'part5_reflection = """\nSample answer: with slerp the first picture changed smoothly into the second, passing through '
        'plausible in-between items. The plain-average row looked washed out and blurry in the middle. The check printed '
        'that the noise size at the midpoint of the plain average is smaller than the endpoints (about 0.7 times), so the '
        'network was shown noise of a size it never saw in training, while slerp keeps the noise at the right size.\n"""',
    'part6_reflection = """\nTODO: your answer here\n"""':
        'part6_reflection = """\nSample answer: adding a style such as flat vector illustration changed the whole look of the '
        'picture, and words like bright gym changed the background, but the basketball was sometimes missing, so some words '
        'in the prompt were ignored. The robot hands looked odd, a failure like extra fingers that probably happens because '
        'the model learns local patterns, not an exact count. Compared with my tiny model, SD-Turbo makes large color '
        'pictures and follows a text prompt, which mine cannot.\n"""',
    'part7_reflection = """\nTODO: your answer here\n"""':
        'part7_reflection = """\nSample answer: I would refuse to make a realistic picture of a real classmate doing something '
        'they never did, because it could look like a real photo and cause harm to their reputation without their consent. '
        'If I share an AI-made picture I would put a label on it saying it is AI generated, and I would avoid copying the '
        'style of a living artist without permission.\n"""',
    '# TODO 7\n': '# TODO 7 (sample answer)\n',
    '# TODO 8\n': '# TODO 8 (sample answer)\n',
}


def to_nb(solution):
    nb_cells = []
    for kind, src, sol in cells:
        text = sol if (solution and sol is not None) else src
        if solution:
            for old, new in SOLUTION_TEXT.items():
                text = text.replace(old, new)
        lines = text.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {"cells": nb_cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5}


for solution, name in [(False, "diffusion_assignment.ipynb"), (True, "diffusion_assignment_solution.ipynb")]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_nb(solution), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", name)

GENERATED = "# GENERATED by build_assignment.py from the assignment notebook. Do not edit by hand.\n"
os.makedirs(AG_DIR, exist_ok=True)
with open(os.path.join(AG_DIR, "provided.py"), "w", encoding="utf-8", newline="\n") as f:
    f.write(GENERATED + '"""The PROVIDED code from the handout that the grader needs, verbatim."""\n'
            + IMPORTS_SRC + SCENE_SRC + MODEL_SRC)
with open(os.path.join(AG_DIR, "reference_solution.py"), "w", encoding="utf-8", newline="\n") as f:
    f.write(GENERATED + '"""Reference versions of the graded functions. Exec\'d on top of provided.py."""\n'
            + SOL_ALPHA + SOL_NOISE + SOL_PREDICT + SOL_LOSS + SOL_SAMPLE + SOL_SLERP)
print("wrote provided.py and reference_solution.py")
