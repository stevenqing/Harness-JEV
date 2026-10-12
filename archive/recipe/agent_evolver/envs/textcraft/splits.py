"""Split the fixed manifest into held-in / held-out / compat100 (SPEC §2.1).

Deterministic: tasks are grouped by depth and sorted by goal; held-in takes the
first k per depth, held-out the next k (non-overlapping); compat100 is a
fixed-seed sample from the full per-depth pool. Run::

    .venv/bin/python -m recipe.agent_evolver.envs.textcraft.splits
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "manifest.jsonl")
SPLITS_PATH = os.path.join(HERE, "splits.json")

HELDIN = {1: 12, 2: 26, 3: 22, 4: 4}
HELDOUT = {1: 12, 2: 26, 3: 22, 4: 4}
COMPAT = {1: 31, 2: 41, 3: 25, 4: 3}
COMPAT_SEED = 20241009


def load_manifest() -> List[Dict]:
    rows: List[Dict] = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build() -> Dict[str, List[str]]:
    rows = load_manifest()
    by_depth: Dict[int, List[str]] = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r["task_id"])
    for d in by_depth:
        by_depth[d].sort()

    heldin: List[str] = []
    heldout: List[str] = []
    compat: List[str] = []
    rng = random.Random(COMPAT_SEED)
    for depth in sorted(by_depth):
        pool = by_depth[depth]
        k_in = HELDIN.get(depth, 0)
        k_out = HELDOUT.get(depth, 0)
        k_c = COMPAT.get(depth, 0)
        if k_in + k_out > len(pool):
            raise RuntimeError(
                f"depth {depth}: held-in+held-out need {k_in}+{k_out} "
                f"tasks but only {len(pool)} available"
            )
        if k_c > len(pool):
            raise RuntimeError(
                f"depth {depth}: compat100 needs {k_c} but only {len(pool)} available"
            )
        heldin.extend(pool[0:k_in])
        heldout.extend(pool[k_in : k_in + k_out])
        if k_c:
            compat.extend(sorted(rng.sample(pool, k_c)))

    return {"heldin": heldin, "heldout": heldout, "compat100": compat}


def main() -> None:
    splits = build()
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for key in ("heldin", "heldout", "compat100"):
        print(f"{key}: {len(splits[key])} tasks")
    print(f"splits -> {SPLITS_PATH}")


if __name__ == "__main__":
    main()
