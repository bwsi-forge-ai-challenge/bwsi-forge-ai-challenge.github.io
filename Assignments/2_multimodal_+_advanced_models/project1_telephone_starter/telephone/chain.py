"""Run a telephone chain from one seed and an ordering string. This is the "one command" from the
interface contract:

    python -m telephone.chain --members configs/toy --seed-image seeds/demo.png --order A.C,A.M,A.L,A.G --cycles 2

An ordering is a comma-separated list of MEMBER.STAGE hops. `--cycles N` repeats it N times, feeding the
last output back in. Every hop writes its output file and one JSON line to records.jsonl.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import yaml

from . import backends  # noqa: F401  (registers all backends)
from . import media
from .stage import STAGE_IO, build_stage


def load_members(config_dir: Path) -> dict:
    """Read every <member>.yaml in a folder. A member-level `device:` applies to all of their stages."""
    members = {}
    for path in sorted(Path(config_dir).glob("*.y*ml")):
        cfg = yaml.safe_load(path.read_text())
        name = str(cfg.get("member", path.stem))
        for stage_cfg in cfg["stages"].values():
            stage_cfg.setdefault("device", cfg.get("device"))
        members[name] = cfg
    if not members:
        raise SystemExit(f"No member configs found in {config_dir}")
    return members


def parse_order(order: str) -> list[tuple[str, str]]:
    hops = []
    for token in order.split(","):
        member, _, kind = token.strip().rpartition(".")
        if not member or kind not in STAGE_IO:
            raise ValueError(f"Bad hop {token!r}: expected MEMBER.STAGE with STAGE in C, M, L, G (e.g. A.C)")
        hops.append((member, kind))
    for (m1, k1), (m2, k2) in zip(hops, hops[1:]):
        if STAGE_IO[k1][1] != STAGE_IO[k2][0]:
            raise ValueError(f"{m1}.{k1} outputs {STAGE_IO[k1][1]} but {m2}.{k2} expects {STAGE_IO[k2][0]}")
    return hops


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def run_chain(members_dir, order, seed_path, cycles=1, seed=0, out_dir="runs", tag=None, low_memory=False) -> Path:
    members = load_members(members_dir)
    one_cycle = parse_order(order)
    first_in, last_out = STAGE_IO[one_cycle[0][1]][0], STAGE_IO[one_cycle[-1][1]][1]
    if cycles > 1 and first_in != last_out:
        raise ValueError(f"--cycles needs the ordering to end with {first_in}, but it ends with {last_out}")
    seed_kind = media.kind_of(seed_path)
    if seed_kind != first_in:
        raise ValueError(f"The first hop expects {first_in}, but the seed is {seed_kind}")
    for member, kind in one_cycle:
        if member not in members:
            raise ValueError(f"No config for member {member!r} in {members_dir} (found: {', '.join(members)})")

    run_name = tag or f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(seed_path).stem}_{order.replace(',', '-')}_x{cycles}_s{seed}"
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "chain.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "members_dir": str(members_dir), "order": order, "cycles": cycles,
        "seed": seed, "seed_path": str(seed_path),
        "member_configs": {m: members[m] for m in sorted({m for m, _ in one_cycle})},
    }, indent=2))

    current = media.load(seed_path)
    in_path = media.save(current, run_dir / "hop_00_seed", seed_kind)
    stages = {}
    with open(run_dir / "records.jsonl", "w") as log:
        for hop, (member, kind) in enumerate(one_cycle * cycles, start=1):
            key = (member, kind)
            if key not in stages:
                stages[key] = build_stage(member, kind, members[member]["stages"][kind])
                stages[key].ensure_loaded()  # keep model loading out of the timed call
            stage = stages[key]
            hop_seed = seed * 1000 + hop  # fixed per hop, so a rerun is identical
            start = time.perf_counter()
            output = stage.run(current, hop_seed)
            elapsed = time.perf_counter() - start
            out_path = media.save(output, run_dir / f"hop_{hop:02d}_{member}.{kind}", stage.output_type)
            record = {**stage.describe(), "hop": hop, "cycle": (hop - 1) // len(one_cycle) + 1, "seed": hop_seed,
                      "wall_clock_s": round(elapsed, 3), "input_path": str(in_path), "output_path": str(out_path)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            preview = output if isinstance(output, str) else out_path.name
            print(f"[hop {hop:02d}] {stage.name:<6} {elapsed:6.2f}s  {preview}")
            current, in_path = output, out_path
            if low_memory:  # reload next time instead of keeping every model in memory at once
                del stages[key], stage
                free_memory()
    print(f"Done: {run_dir}")
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", required=True, help="folder of <member>.yaml configs")
    p.add_argument("--order", required=True, help="e.g. A.C,B.M,C.L,A.G")
    p.add_argument("--seed-input", "--seed-image", dest="seed_input", required=True,
                   help="seed image (.png/.jpg), or a .txt seed text if the ordering starts with M or G")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", help="name for the run folder (default: timestamp + settings)")
    p.add_argument("--low-memory", action="store_true",
                   help="unload each model after its hop (slower, but fits GPUs/laptops under ~12 GB)")
    args = p.parse_args(argv)
    run_chain(args.members, args.order, args.seed_input, args.cycles, args.seed, args.out, args.tag, args.low_memory)


if __name__ == "__main__":
    main()
