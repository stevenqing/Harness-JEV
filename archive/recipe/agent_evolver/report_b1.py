# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""B1 readout recompute + side-by-side report (kev vs semif vs floor).

Reads the two ``episodes.jsonl`` written by ``b1_readout`` and reproduces the
per-cell ``success − floors.gate.uniform`` table from the raw records — this is
the B1 half of the §6 "every number recomputable from episodes + tiers.json"
requirement.  No model calls, no network.

    .venv/bin/python -m recipe.agent_evolver.report_b1
"""
from __future__ import annotations

import json
from pathlib import Path

from .b1_readout import _bootstrap_mean_ci

BUNDLE = Path("/mnt/llmshared-ssd-hd/shishuqing/gridgames")
GAMES = ["frozenlake", "sokoban"]
TIERS = ["L4", "L8", "L16", "L32"]
RUNS = Path("runs/agent_evolver/gridgames/b1_readout")


def _floors(bundle: Path) -> dict:
    tiers = json.loads((bundle / "tiers.json").read_text())
    return tiers["floors"]["gate"]


def _load_cells(backend: str, bundle: Path) -> dict:
    path = RUNS / backend / "episodes.jsonl"
    floors = _floors(bundle)
    cells: dict[tuple[str, str], dict] = {
        (g, t): {"levels": {}} for g in GAMES for t in TIERS
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("left_out"):
            continue
        cell = cells[(r["game"], r["tier"])]
        cell["levels"].setdefault(r["level_id"], []).append(1.0 if r["success"] else 0.0)
    return cells, floors


def main() -> int:
    bundle = BUNDLE
    kev, floors = _load_cells("kev", bundle)
    semif, _ = _load_cells("semif", bundle)

    print("# B1 readout — success − floors.gate.uniform (kev vs semif)\n")
    print("| game | tier | floor | kev succ | kev Δ | semif succ | semif Δ | semif 95% CI |")
    print("|---|---|---|---|---|---|---|---|")
    for g in GAMES:
        for t in TIERS:
            floor = floors[g][t]["uniform"]
            def _stats(cells):
                rates = [sum(v) / len(v) for v in cells[(g, t)]["levels"].values()]
                mean = sum(rates) / len(rates) if rates else 0.0
                lo, hi = _bootstrap_mean_ci(rates)
                return mean, lo, hi
            kmean, _, _ = _stats(kev)
            smean, slo, shi = _stats(semif)
            print(f"| {g} | {t} | {floor:.4f} | {kmean:.3f} | {kmean - floor:+.3f} | "
                  f"{smean:.3f} | {smean - floor:+.3f} | [{slo:.3f}, {shi:.3f}] |")

    print("\nkev 95% CIs (per cell, over levels):")
    for g in GAMES:
        for t in TIERS:
            rates = [sum(v) / len(v) for v in kev[(g, t)]["levels"].values()]
            mean = sum(rates) / len(rates) if rates else 0.0
            lo, hi = _bootstrap_mean_ci(rates)
            print(f"  {g} {t}: {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
