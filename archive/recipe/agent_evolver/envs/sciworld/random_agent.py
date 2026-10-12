"""Random floor agent for ScienceWorld (SCIWORLD_SPEC.md §9 G1).

Each step uniformly samples from the *valid* action-object combinations, for at
most ``max_steps`` steps.  Every episode is run ``n_runs`` times (default 2) to
report the mean score / success floor.  Runs in the py3.8 ``ee-sciworld`` env::

    ee-sciworld/bin/python -m recipe.agent_evolver.envs.sciworld.random_agent \
        --n-runs 2 --max-steps 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(HERE, "splits.json")

SIMPLIFICATION = "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction"
ENV_STEP_LIMIT = 100
N_WORKERS = int(os.environ.get("SCIWORLD_RANDOM_WORKERS", "8"))

_env = None
_simpl = None
_max_steps = 40
_n_runs = 2


def _init_worker(simpl: str, max_steps: int, n_runs: int) -> None:
    global _env, _simpl, _max_steps, _n_runs
    _simpl = simpl
    _max_steps = max_steps
    _n_runs = n_runs
    from scienceworld import ScienceWorldEnv
    _env = ScienceWorldEnv(envStepLimit=ENV_STEP_LIMIT)


def _run_episode(item) -> Dict:
    episode_id, task_name, variation = item
    scores: List[int] = []
    successes = 0
    rng = random.Random(hash((episode_id, os.getpid())) & 0xFFFFFFFF)
    for _ in range(_n_runs):
        _env.load(task_name, variation, _simpl)
        _obs, info = _env.reset()
        final_score = int(info["score"])
        done = False
        for _ in range(_max_steps):
            valid = list(_env.get_valid_action_object_combinations())
            if not valid:
                break
            action = rng.choice(valid)
            _obs, _r, done, info = _env.step(action)
            final_score = int(info["score"])
            if done or info["score"] < 0:
                break
        scores.append(max(final_score, 0) / 100.0)
        if final_score == 100:
            successes += 1
    return {
        "episode_id": episode_id, "task_name": task_name,
        "scores": scores, "successes": successes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    splits = json.load(open(SPLITS_PATH, encoding="utf-8"))
    items = sorted((eid, ep["task_name"], int(ep["variation"])) for eid, ep in splits["episodes"].items())

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker,
                  initargs=(SIMPLIFICATION, args.max_steps, args.n_runs)) as pool:
        results = pool.map(_run_episode, items)

    all_scores = [s for r in results for s in r["scores"]]
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    success_rate = sum(r["successes"] for r in results) / (len(results) * args.n_runs) if results else 0.0

    print(f"random agent floor: {len(results)} episodes x {args.n_runs} runs", flush=True)
    print(f"  mean score = {mean_score:.4f} (0-1)", flush=True)
    print(f"  success rate = {success_rate:.4f}", flush=True)


if __name__ == "__main__":
    main()
