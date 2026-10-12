"""Gold replay verification for ScienceWorld (SCIWORLD_SPEC.md §9 G1).

For each held-in + held-out episode, load the corresponding variation with
``generateGoldPath=True``, replay the gold action sequence, and check the final
score is 100.  At least 95% of the 128 episodes must reach 100; every failure is
listed with the reason.  Runs in the py3.8 ``ee-sciworld`` env::

    ee-sciworld/bin/python -m recipe.agent_evolver.envs.sciworld.gold_agent
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(HERE, "splits.json")

SIMPLIFICATION = "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction"
ENV_STEP_LIMIT = 100
N_WORKERS = int(os.environ.get("SCIWORLD_GOLD_WORKERS", "8"))

_env = None
_simpl = None


def _init_worker(simpl: str) -> None:
    global _env, _simpl
    _simpl = simpl
    from scienceworld import ScienceWorldEnv
    _env = ScienceWorldEnv(envStepLimit=ENV_STEP_LIMIT)


def _replay(item) -> Dict:
    episode_id, task_name, variation = item
    try:
        _env.load(task_name, variation, _simpl, generateGoldPath=True)
        gold = _env.get_gold_action_sequence()
        _env.reset()
        final_score = 0
        done = False
        for a in gold:
            _obs, _r, done, info = _env.step(a)
            final_score = int(info["score"])
            if final_score < 0:
                break
        return {
            "episode_id": episode_id, "task_name": task_name, "variation": variation,
            "gold_len": len(gold), "final_score": final_score, "done": done,
        }
    except Exception as exc:  # noqa: BLE001
        return {"episode_id": episode_id, "task_name": task_name, "variation": variation,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    splits = json.load(open(SPLITS_PATH, encoding="utf-8"))
    episodes = splits["episodes"]
    items = sorted((eid, ep["task_name"], int(ep["variation"])) for eid, ep in episodes.items())

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(SIMPLIFICATION,)) as pool:
        results = pool.map(_replay, items)

    passed = [r for r in results if r.get("final_score") == 100]
    failed = [r for r in results if "error" in r or r.get("final_score") != 100]
    rate = len(passed) / len(results) if results else 0.0

    print(f"gold replay: {len(passed)}/{len(results)} episodes reach 100 ({rate:.3f})", flush=True)
    for r in failed:
        if "error" in r:
            print(f"  FAIL {r['episode_id']}: {r['error']}", flush=True)
        else:
            print(f"  FAIL {r['episode_id']}: final_score={r['final_score']} done={r['done']} "
                  f"gold_len={r['gold_len']}", flush=True)
    print(f"GATE G1: {'PASS' if rate >= 0.95 else 'FAIL'} (threshold 0.95)", flush=True)


if __name__ == "__main__":
    main()
