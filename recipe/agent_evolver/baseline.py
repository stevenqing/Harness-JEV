# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Baseline reproduction of Qwen3.5-4B on ALFWorld / WebShop (no evolution).

Runs the agent with the Skill1 / verl-agent base settings (Qwen default system
prompt, think/action template, history, temp 0.4) on a fixed eval set, reports
n/N, and dumps one full episode's context for human review.

Usage (inside the HarnessX .venv)::

    .venv/bin/python -m recipe.agent_evolver.baseline alfworld --num-tasks 64
    .venv/bin/python -m recipe.agent_evolver.baseline webshop  --num-tasks 64
    .venv/bin/python -m recipe.agent_evolver.baseline both     --num-tasks 64
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _lg in ("LiteLLM", "httpx", "httpx2"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

from .agents import make_agent_provider, run_alfworld, run_webshop
from .spec import QWEN_SYSTEM, HarnessSpec

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs" / "agent_evolver" / "baseline"


def _dump_episode(records: list[dict], out: Path, benchmark: str) -> None:
    r = records[0]
    lines = [
        f"# {benchmark} task {r['task_id']} — full context dump",
        "",
        f"- Goal: {r['goal']}",
        f"- Result: success={r['success']}  steps={r['steps']}  reward={r['reward']:.3f}",
        f"- Final obs: {r['final_obs'][:600]}",
        "",
    ]
    for s in r["trajectory"]:
        lines.append(f"## Step {s['step']}")
        lines.append("### Prompt (system + user)")
        lines.append("```")
        lines.append(f"[system]\n{QWEN_SYSTEM}")
        lines.append("")
        lines.append(f"[user]\n{s['prompt']}")
        lines.append("```")
        lines.append("### Raw model output")
        lines.append("```")
        lines.append(s["raw"])
        lines.append("```")
        lines.append(f"### Extracted action: `{s['action']}`  (invalid={s['invalid']})")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


async def run(benchmark: str, num_tasks: int, seed: int, start: int = 0, tag: str = "") -> list[dict]:
    out_dir = RUNS_ROOT / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = make_agent_provider()
    spec = HarnessSpec(system_prompt=QWEN_SYSTEM, guidance="")
    task_ids = list(range(start, start + num_tasks))
    tag = f"_{tag}" if tag else ""

    if benchmark == "alfworld":
        records = await run_alfworld(provider, spec, task_ids)
    else:
        records = await run_webshop(provider, spec, task_ids, seed=seed)

    n = sum(1 for r in records if r["success"])
    total = len(records)
    rate = n / total if total else 0.0
    mean_reward = sum(r["reward"] for r in records) / total if total else 0.0

    print(f"[baseline:{benchmark}] success {n}/{total} = {rate:.3f}  "
          f"(mean reward {mean_reward:.3f}, seed={seed})", flush=True)
    for i, r in enumerate(records):
        print(f"  task {r['task_id']}: {'OK ' if r['success'] else 'FAIL'} "
              f"steps={r['steps']} reward={r['reward']:.3f}", flush=True)

    (out_dir / f"report_seed{seed}_start{start}{tag}.json").write_text(json.dumps({
        "benchmark": benchmark,
        "seed": seed,
        "start": start,
        "num_tasks": total,
        "success": n,
        "rate": rate,
        "mean_reward": mean_reward,
        "per_task": [
            {"task_id": r["task_id"], "success": r["success"], "steps": r["steps"], "reward": r["reward"]}
            for r in records
        ],
    }, indent=2))

    _dump_episode(records, out_dir / f"episode_0_context_seed{seed}_start{start}{tag}.md", benchmark)
    print(f"[baseline:{benchmark}] context dump -> {out_dir / f'episode_0_context_seed{seed}_start{start}{tag}.md'}", flush=True)
    return records


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["alfworld", "webshop", "both"])
    ap.add_argument("--num-tasks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="task index offset (for sharding across GPUs)")
    ap.add_argument("--tag", type=str, default="", help="suffix for output filenames (e.g. model size)")
    args = ap.parse_args()

    benchmarks = ["alfworld", "webshop"] if args.benchmark == "both" else [args.benchmark]
    for b in benchmarks:
        await run(b, args.num_tasks, args.seed, args.start, args.tag)


if __name__ == "__main__":
    asyncio.run(main())
