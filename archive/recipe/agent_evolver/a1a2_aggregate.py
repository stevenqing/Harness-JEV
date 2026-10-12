# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Pool per-run A1/A2 outputs and bootstrap over tasks (spec S3).

The sampling unit for ALFWorld is the *task* (game_idx 0..63); the same 64 tasks
reappear across every run (model × round), so per-task agreement is collapsed
first and the 95% percentile bootstrap resamples the 64 tasks (10k draws).  A
gap is real iff the interval of the difference excludes 0.

    python -m recipe.agent_evolver.a1a2_aggregate --json-dir /tmp/a1a2_json

Prints the pooled table (kev vs rule vs label, with CIs) and writes
``<json-dir>/aggregate.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .reviewer import auc, cohens_kappa
from .stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, bootstrap_ci, bootstrap_diff
import random


def _load(dir_: Path) -> list[dict]:
    out: list[dict] = []
    for jf in sorted(dir_.glob("*.json")):
        if jf.name == "aggregate.json":
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        src = data.get("trajectories", jf.stem)
        for row in data.get("per_trajectory", []):
            out.append({"src": src, **row})
    return out


def _group_key(traj_id: str) -> int | str:
    """Sampling unit: numeric ALF ``game_idx``, or the gridgames ``level_id``
    (the ``_eN`` episode suffix is stripped so episodes group under a level)."""
    if traj_id.isdigit():
        return int(traj_id)
    return traj_id.rsplit("_e", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = _load(args.json_dir)
    if not rows:
        raise SystemExit(f"no per-run JSON under {args.json_dir}")

    # game_idx = numeric traj_id (ALF); level_id otherwise (gridgames) — group instances by unit
    by_task: dict[int | str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[_group_key(r["traj_id"])].append(r)

    kev_agree = {}
    rule_agree = {}
    for task, insts in by_task.items():
        kev_agree[task] = sum(i["kev_tag"] == i["label"] for i in insts) / len(insts)
        rule_agree[task] = sum(i["rule"] == i["label"] for i in insts) / len(insts)

    tasks = sorted(by_task)
    kev_ci = bootstrap_ci([kev_agree[t] for t in tasks])
    rule_ci = bootstrap_ci([rule_agree[t] for t in tasks])
    gap = bootstrap_diff([kev_agree[t] for t in tasks], [rule_agree[t] for t in tasks])

    def _pooled(label_key, pred_key):
        return [i[label_key] for i in rows], [i[pred_key] for i in rows]

    kev_kappa = cohens_kappa(*_pooled("label", "kev_tag"))
    rule_kappa = cohens_kappa(*_pooled("label", "rule"))

    # AUC of each mode score against its binary label, pooled at trajectory level
    aucs = {}
    modes = [k for k in rows[0]["kev_scores"].keys()]
    for m in modes:
        truth = [r["label"] == m for r in rows]
        scores = [r["kev_scores"].get(m, 0.0) for r in rows]
        aucs[m] = auc(truth, scores)

    # task-block bootstrap for kappa + AUC (resample tasks, gather their instances)
    def _task_block_stat(fn):
        vals = []
        rng = random.Random(BOOTSTRAP_SEED)
        n = len(tasks)
        for _ in range(BOOTSTRAP_DRAWS):
            idxs = rng.choices(tasks, k=n)
            insts = [i for t in idxs for i in by_task[t]]
            vals.append(fn(insts))
        vals.sort()
        return vals[int(0.025 * BOOTSTRAP_DRAWS)], vals[int(0.975 * BOOTSTRAP_DRAWS) - 1]

    def _kappa_kev(insts):
        return cohens_kappa([i["label"] for i in insts], [i["kev_tag"] for i in insts])

    def _kappa_rule(insts):
        return cohens_kappa([i["label"] for i in insts], [i["rule"] for i in insts])

    kev_kappa_ci = _task_block_stat(_kappa_kev)
    rule_kappa_ci = _task_block_stat(_kappa_rule)

    auc_ci = {}
    for m in modes:
        auc_ci[m] = _task_block_stat(
            lambda insts, m=m: auc(
                [i["label"] == m for i in insts],
                [i["kev_scores"].get(m, 0.0) for i in insts],
            )
        )

    result = {
        "n_trajectories": len(rows),
        "n_tasks": len(tasks),
        "instances_per_task": sorted({len(v) for v in by_task.values()}),
        "kev_accuracy": {"mean": kev_ci[0], "lo": kev_ci[1], "hi": kev_ci[2]},
        "rule_accuracy": {"mean": rule_ci[0], "lo": rule_ci[1], "hi": rule_ci[2]},
        "gap_kev_minus_rule": {"mean": gap[0], "lo": gap[1], "hi": gap[2], "real": gap[3]},
        "kev_kappa": {"mean": kev_kappa, "lo": kev_kappa_ci[0], "hi": kev_kappa_ci[1]},
        "rule_kappa": {"mean": rule_kappa, "lo": rule_kappa_ci[0], "hi": rule_kappa_ci[1]},
        "auc": {m: {"mean": aucs[m], "lo": auc_ci[m][0], "hi": auc_ci[m][1]} for m in modes},
    }
    (args.json_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def _fmt_ci(d):
        return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"

    print(f"\n# A1/A2 pooled — {len(rows)} trajectories across {len(tasks)} tasks "
          f"(instances/task={result['instances_per_task']})")
    print(f"kev  accuracy: {_fmt_ci(result['kev_accuracy'])}")
    print(f"rule accuracy: {_fmt_ci(result['rule_accuracy'])}")
    print(f"gap  (kev−rule): {_fmt_ci(result['gap_kev_minus_rule'])}  real={result['gap_kev_minus_rule']['real']}")
    print(f"kev  kappa: {_fmt_ci(result['kev_kappa'])}")
    print(f"rule kappa: {_fmt_ci(result['rule_kappa'])}")
    for m, d in result["auc"].items():
        print(f"AUC({m}): {_fmt_ci(d)}")


if __name__ == "__main__":
    main()
