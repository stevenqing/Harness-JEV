# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Side-by-side R0 vs TextArena-prompt baseline for FrozenLake / Sokoban.

Reads two ``episodes.jsonl`` (the spec R0 baseline and the ``r0_textarena``
experiment) and prints, per (model, game, tier): success rate, success minus the
uniform-random floor, parse_fail/move, and the change between the two prompts.
This is the A/B that answers "is the R0 prompt (symbols/wording) the problem?".

    python -m recipe.agent_evolver.compare_gridgames_prompts \
        --r0 runs/agent_evolver/gridgames/baseline/episodes.jsonl \
        --ta runs/agent_evolver/gridgames/baseline_textarena/episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_BUNDLE = Path("/mnt/llmshared-ssd-hd/shishuqing/gridgames")


def load_levels():
    return [json.loads(l) for l in (_BUNDLE / "levels.jsonl").read_text().splitlines() if l.strip()]


def load_gate_floors():
    tiers = json.loads((_BUNDLE / "tiers.json").read_text())
    gate = tiers["floors"]["gate"]
    return {(g, t): gate[g][t]["uniform"] for g in gate for t in gate[g]}


def load_episodes(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("left_out"):
            continue
        out.append(rec)
    return out


def cells(episodes):
    # model -> (game, tier) -> list of episode records
    m = {}
    for e in episodes:
        m.setdefault(e["model"], {}).setdefault((e["game"], e["tier"]), []).append(e)
    return m


def stats(eps):
    n = len(eps)
    succ = sum(e["report"]["success"] for e in eps)
    moves = sum(e["report"]["moves"] for e in eps)
    pf = sum(e["report"]["parse_fail"] for e in eps)
    blocked = sum(e["report"]["blocked_moves"] for e in eps)
    dead = sum(e["report"]["dead"] for e in eps)
    return {
        "n": n,
        "success": succ / n if n else 0.0,
        "pf_move": pf / moves if moves else 0.0,
        "blocked_move": blocked / moves if moves else 0.0,
        "dead": dead / n if n else 0.0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r0", default="runs/agent_evolver/gridgames/baseline/episodes.jsonl")
    p.add_argument("--ta", default="runs/agent_evolver/gridgames/baseline_textarena/episodes.jsonl")
    a = p.parse_args(argv)

    floors = load_gate_floors()
    levels_by_id = {l["id"]: l for l in load_levels()}

    r0c = cells(load_episodes(a.r0))
    tac = cells(load_episodes(a.ta))

    models = sorted({**r0c, **tac}.keys())
    tiers = ["L4", "L8", "L16", "L32"]
    games = ["frozenlake", "sokoban"]

    rows = []
    for model in models:
        for game in games:
            for tier in tiers:
                key = (game, tier)
                r0s = stats(r0c.get(model, {}).get(key, []))
                tas = stats(tac.get(model, {}).get(key, []))
                floor = floors.get(key, 0.0)
                d_succ = (tas["success"] - r0s["success"]) * 100
                r0_live = r0s["success"] - floor
                ta_live = tas["success"] - floor
                rows.append({
                    "model": model, "game": game, "tier": tier,
                    "floor": floor,
                    "r0_succ": r0s["success"], "ta_succ": tas["success"],
                    "r0_live": r0_live, "ta_live": ta_live,
                    "d_succ": d_succ,
                    "r0_pf": r0s["pf_move"], "ta_pf": tas["pf_move"],
                    "r0_n": r0s["n"], "ta_n": tas["n"],
                })

    def live(v):
        return "**live**" if v >= 0.10 else ""

    print("| model | game | tier | floor | R0 succ | TA succ | Δ succ (pt) | R0 succ−floor | TA succ−floor | R0 pf/mv | TA pf/mv |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {r['game']} | {r['tier']} | {r['floor']:.4f} | "
            f"{r['r0_succ']*100:5.2f}% | {r['ta_succ']*100:5.2f}% | {r['d_succ']:+6.2f} | "
            f"{r['r0_live']:+.3f}{live(r['r0_live'])} | {r['ta_live']:+.3f}{live(r['ta_live'])} | "
            f"{r['r0_pf']:.3f} | {r['ta_pf']:.3f} |"
        )

    print("\n*succ−floor ≥ 0.10 ⇒ working tier (spec §7 B4). Δ succ is in percentage points.*")

    # aggregate headline: any working tier under either prompt?
    print("\n### working-tier summary")
    print("| model | game | R0 working tier | TA working tier |")
    print("|---|---|---|---|")
    for model in models:
        for game in games:
            r0_wt = [r["tier"] for r in rows if r["model"] == model and r["game"] == game and r["r0_live"] >= 0.10]
            ta_wt = [r["tier"] for r in rows if r["model"] == model and r["game"] == game and r["ta_live"] >= 0.10]
            print(f"| {model} | {game} | {r0_wt[-1] if r0_wt else 'none'} | {ta_wt[-1] if ta_wt else 'none'} |")


if __name__ == "__main__":
    main()
