# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""A1/A2 driver: independent labels vs rule tagger vs kev reviewer.

For one benchmark and one trajectories directory, this

1. builds the A1 oracle label for every failed trajectory,
2. builds the A2 deterministic rule tag,
3. runs the kev reviewer (one noul per failure mode) over the same trajectories,
4. reports accuracy / Cohen's kappa (tag-level) and the AUC of each mode score
   (label-level), for both kev-vs-label and rule-vs-label.

Output is a JSON (full per-trajectory table, for the recompute script) plus a
Markdown table printed to stdout.

    python -m recipe.agent_evolver.a1a2_runner \\
        --benchmark alfworld --trajectories …/R0/trajectories \\
        --env-url http://127.0.0.1:18082 --out /tmp/a1a2_alf.json

gridgames replays the recorded actions in-process (no env server), so it only
needs ``--trajectories`` and the bundle (``--bundle``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from harnessx.meta_harness.decision_prior import analyze_trajectories

from . import reviewer
from .decision_prior import _parse_trajectory, _render_state
from .failure_labels import (
    alfworld_label,
    gamefile_order,
    gridgames_label,
    load_alf_task,
)
from .rule_tagger import alfworld_rule, gridgames_rule

_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR", "/mnt/llmshared-ssd-hd/shishuqing/gridgames")).resolve()


def _load_levels(bundle: Path) -> dict[str, dict]:
    return {
        lvl["id"]: lvl
        for line in (bundle / "levels.jsonl").read_text().splitlines()
        if line.strip()
        for lvl in [json.loads(line)]
    }


def _failed(fields: dict) -> bool:
    return fields is not None


def _render_state_gg(t: dict) -> str:
    """gridgames render: the shared state plus ``final_obs`` (board + last-move
    results), which is where the explicit death/blocked signal lives."""
    base = _render_state(t)
    fo = t.get("final_obs", "")
    return base + (f"final_obs: {fo}\n" if fo else "")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["alfworld", "gridgames"])
    ap.add_argument("--trajectories", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env-url", default="http://127.0.0.1:18082")
    ap.add_argument("--bundle", type=Path, default=_BUNDLE_DIR)
    ap.add_argument("--kev-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="kev")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    traj_dir: Path = args.trajectories
    if not traj_dir.is_dir():
        raise SystemExit(f"trajectories dir not found: {traj_dir}")

    files = sorted(traj_dir.glob("*.md"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 1e9, p.stem))
    fields_by_id: dict[str, dict] = {}
    for p in files:
        f = _parse_trajectory(p)
        if _failed(f):
            fields_by_id[p.stem] = f

    if args.benchmark == "alfworld":
        order = gamefile_order(args.env_url, args.seed)
        # traj_id (the .md stem) *is* the game_idx -> order[game_idx] gamefile
        tasks = {tid: load_alf_task(order[int(tid)]) for tid in fields_by_id}
        labels = {tid: alfworld_label([a for a, _ in f["pairs"]], tasks[tid])
                  for tid, f in fields_by_id.items()}
        rules = {tid: alfworld_rule([a for a, _ in f["pairs"]])
                 for tid, f in fields_by_id.items()}
        modes = reviewer.ALF_MODES
        questions = reviewer.ALF_QUESTIONS
    else:
        levels = _load_levels(args.bundle)
        labels, rules = {}, {}
        for tid, f in fields_by_id.items():
            level_id = tid.rsplit("_e", 1)[0]
            level = levels[level_id]
            game = level["game"]
            acts = [None if inv else a for a, inv in f["pairs"]]
            labels[tid] = gridgames_label(game, level, acts)
            rules[tid] = gridgames_rule(game, f["eval_reason"], acts)
        modes = reviewer.GRIDGAMES_MODES
        questions = reviewer.GRIDGAMES_QUESTIONS

    rows, _elapsed = await analyze_trajectories(
        trajectories_dir=traj_dir,
        questions=questions,
        parse_trajectory=_parse_trajectory,
        render_state=_render_state_gg if args.benchmark == "gridgames" else _render_state,
        model=args.model,
        model_kwargs={"base_url": args.kev_url},
        max_trajectories=100_000,
    )

    # align label / rule / kev by traj_id
    kev_scores: dict[str, dict[str, float]] = {}
    for r in rows:
        scores = {m: r.decisions[i].probabilities.get("true", 0.0) for i, m in enumerate(modes)}
        kev_scores[r.traj_id] = scores

    common = [tid for tid in labels if tid in kev_scores]
    label_v = [labels[t] for t in common]
    rule_v = [rules[t] for t in common]
    kev_tag_v = [reviewer.tag_from_scores(kev_scores[t], modes) for t in common]

    def _metric_block(truth, pred):
        return {
            "n": len(common),
            "accuracy": reviewer.accuracy(truth, pred),
            "cohens_kappa": reviewer.cohens_kappa(truth, pred),
        }

    result = {
        "benchmark": args.benchmark,
        "trajectories": str(traj_dir),
        "n_failed": len(labels),
        "n_scored": len(common),
        "modes": list(modes),
        "kev_vs_label": _metric_block(label_v, kev_tag_v),
        "rule_vs_label": _metric_block(label_v, rule_v),
        "label_distribution": _dist(label_v),
        "rule_distribution": _dist(rule_v),
        "kev_distribution": _dist(kev_tag_v),
        "auc": _auc_block(modes, label_v, kev_scores, common),
        "per_trajectory": [
            {
                "traj_id": t,
                "label": labels[t],
                "rule": rules[t],
                "kev_tag": kev_tag_v[i],
                "kev_scores": {m: round(kev_scores[t].get(m, 0.0), 4) for m in modes},
            }
            for i, t in enumerate(common)
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── human-readable summary ───────────────────────────────────────────────
    print(f"\n# A1/A2 — {args.benchmark}  ({args.out})")
    print(f"failed trajectories: {len(labels)}  scored: {len(common)}")
    print(f"label distribution:  {_dist(label_v)}")
    print(f"rule  distribution:  {_dist(rule_v)}")
    print(f"kev   distribution:  {_dist(kev_tag_v)}\n")
    for name, blk in [("kev vs label", result["kev_vs_label"]), ("rule vs label", result["rule_vs_label"])]:
        print(f"{name}: acc={blk['accuracy']:.3f}  kappa={blk['cohens_kappa']:.3f}")
    for k, v in result["auc"].items():
        print(f"  AUC({k}): {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def _dist(vals: list[str]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(vals))


def _auc_block(modes, label_v, kev_scores, common):
    out: dict = {}
    for m in modes:
        truth = [lab == m for lab in label_v]
        scores = [kev_scores[t].get(m, 0.0) for t in common]
        out[m] = reviewer.auc(truth, scores)
    return out


if __name__ == "__main__":
    asyncio.run(main())
