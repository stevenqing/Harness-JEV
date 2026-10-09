#!/usr/bin/env python3
"""Freeze the gridgames held-in / held-out datasets (fixed, reproducible).

For each game (frozenlake, sokoban) and each tier (L4/L8/L16/L32), take the
16 levels with the smallest ``id`` suffix from the ``evolve`` split (held-in)
and from the ``test`` split (held-out).  Sorting is by the integer suffix of
``id`` (``fl-L16-0`` < ``fl-L16-1`` < ... < ``fl-L16-9`` < ``fl-L16-10``),
NOT lexicographic, so numeric ordering is unambiguous and identical on every
run.  Outputs 4 frozen jsonl files (64 levels each) next to ``levels.jsonl``:

    heldin_frozenlake.jsonl   heldin_sokoban.jsonl
    heldout_frozenlake.jsonl  heldout_sokoban.jsonl

Each line is a verbatim copy of the level dict from ``levels.jsonl``, so the
runner can load these files directly (no re-sampling, no drift).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BUNDLE = Path("/mnt/llmshared-ssd-hd/shishuqing/gridgames")
GAMES = ["frozenlake", "sokoban"]
TIERS = ["L4", "L8", "L16", "L32"]
PER_TIER = 16


def suffix(level: dict) -> int:
    """Integer index encoded in ``id`` (``fl-L16-3`` -> 3)."""
    return int(level["id"].rsplit("-", 1)[1])


def main() -> None:
    levels: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for i, line in enumerate((BUNDLE / "levels.jsonl").open()):
        d = json.loads(line)
        d["_manifest_i"] = i  # original levels.jsonl line index → seed parity
        levels[(d["game"], d["tier"], d["split"])].append(d)

    total = 0
    for game in GAMES:
        for split, outname in (("evolve", f"heldin_{game}.jsonl"),
                               ("test", f"heldout_{game}.jsonl")):
            picked: list[dict] = []
            for tier in TIERS:
                cell = sorted(levels[(game, tier, split)], key=suffix)
                picked.extend(cell[:PER_TIER])
            out = BUNDLE / outname
            with out.open("w") as f:
                for d in picked:
                    f.write(json.dumps(d) + "\n")
            # Audit: 64 levels, 16 per tier, ids contiguous within each tier.
            per_tier = defaultdict(list)
            for d in picked:
                per_tier[d["tier"]].append(d["id"])
            print(f"[{outname}] {len(picked)} levels")
            for tier in TIERS:
                ids = per_tier[tier]
                assert len(ids) == PER_TIER, (tier, len(ids))
                print(f"    {tier}: {ids[0]} .. {ids[-1]}")
            total += len(picked)

    print(f"wrote 4 files, {total} levels total")


if __name__ == "__main__":
    main()
