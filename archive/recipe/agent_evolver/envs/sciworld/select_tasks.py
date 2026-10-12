"""ScienceWorld task selection (SCIWORLD_SPEC.md §3.1).

For each of the 30 task types, take the first 5 *train* variations, generate the
gold action sequence (``generateGoldPath=True``), record each sequence's length,
then replay it and record the final score.  A task is selected iff BOTH hold:

  * the median gold length over the evaluated variations is <= 25, and
  * every evaluated gold sequence (up to 5 train variations) replays to 100.

Electrical tasks (``power-component*`` / ``test-conductivity*``) raise
``ValueError`` under the ``noElectricalAction`` simplification and are excluded
automatically.

NOTE: the gold path *length* is NOT deterministic — 1.2.3's gold planner
searches in identity-hash order, so the same (task, variation) can give
different lengths within one JVM and across JVMs (e.g. ``[23, 23, 20]``).  The
replay score, however, is always 100.  Consequently ``task_selection.json`` is
NOT re-run-reproducible; it is treated as a one-shot pinned artifact (see
SCIWORLD_BASELINE.md G0).  A spawned pool only parallelises the evaluation, but
that does not remove the underlying planner noise.  Run inside the py3.8
``ee-sciworld`` env::

    ee-sciworld/bin/python -m recipe.agent_evolver.envs.sciworld.select_tasks
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "task_selection.json")

SIMPLIFICATION = "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction"
ENV_STEP_LIMIT = 100
N_VARIATIONS = 5
N_WORKERS = int(os.environ.get("SCIWORLD_SELECT_WORKERS", "8"))

_env = None
_simpl = None


def _init_worker(simpl: str) -> None:
    global _env, _simpl
    _simpl = simpl
    from scienceworld import ScienceWorldEnv
    _env = ScienceWorldEnv(envStepLimit=ENV_STEP_LIMIT)


def _eval_variation(pair) -> Dict:
    task, var = pair
    try:
        try:
            _env.load(task, var, _simpl, generateGoldPath=True)
        except ValueError as exc:
            return {"task": task, "var": var, "error": "load_error", "msg": str(exc)[:160]}
        gold = _env.get_gold_action_sequence()
        _env.reset()
        final_score = 0
        for a in gold:
            _obs, _r, _done, info = _env.step(a)
            final_score = int(info["score"])
            if final_score < 0:
                break
        return {"task": task, "var": var, "gold_len": len(gold), "replay_score": final_score}
    except Exception as exc:  # noqa: BLE001 — a crash must not kill the whole scan
        return {"task": task, "var": var, "error": "exception", "msg": str(exc)[:160]}


def _task_names() -> List[str]:
    # A throwaway env just to read the canonical task-name list in sorted order.
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv(envStepLimit=ENV_STEP_LIMIT)
    try:
        return list(env.get_task_names())
    finally:
        env.close()


def main() -> None:
    tasks = _task_names()
    # get_variations_train() is per-task on the Java side, so a task must be
    # loaded first; load variation 0 for each task just to read its list.
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv(envStepLimit=ENV_STEP_LIMIT)
    train_variations: Dict[str, List[int]] = {}
    load_errors: Dict[str, str] = {}
    try:
        for t in tasks:
            try:
                env.load(t, 0, SIMPLIFICATION)
                train_variations[t] = list(env.get_variations_train())
            except ValueError as exc:
                # Electrical tasks raise under noElectricalAction (spec §3.1);
                # exclude them by leaving the variation list empty.
                train_variations[t] = []
                load_errors[t] = str(exc)[:160]
    finally:
        env.close()
    pairs: List[tuple] = [(t, v) for t in tasks for v in train_variations[t][:N_VARIATIONS]]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(SIMPLIFICATION,)) as pool:
        results = pool.map(_eval_variation, pairs)

    by_task: Dict[str, List[Dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    out: Dict = {"simplification": SIMPLIFICATION, "env_step_limit": ENV_STEP_LIMIT, "tasks": []}
    for t in tasks:
        rows = by_task.get(t, [])
        gold_lens = [r["gold_len"] for r in rows if "gold_len" in r]
        replay_scores = [r["replay_score"] for r in rows if "replay_score" in r]
        errors = [{"var": r["var"], "error": r["error"], "msg": r["msg"]} for r in rows if "error" in r]
        if t in load_errors:
            errors.append({"var": None, "error": "load_error", "msg": load_errors[t]})
        median = statistics.median(gold_lens) if gold_lens else None
        selected = bool(
            gold_lens and not errors
            and median is not None and median <= 25
            and all(s == 100 for s in replay_scores)
        )
        out["tasks"].append({
            "task_name": t,
            "gold_len_median": median,
            "gold_lengths": gold_lens,
            "replay_scores": replay_scores,
            "errors": errors,
            "selected": selected,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    selected = [t for t in out["tasks"] if t["selected"]]
    print(f"selected {len(selected)}/{len(out['tasks'])} tasks:", flush=True)
    for t in selected:
        print(f"  {t['task_name']}: median={t['gold_len_median']} replay={t['replay_scores']}", flush=True)
    print(f"excluded {len(out['tasks']) - len(selected)} tasks", flush=True)
    for t in out["tasks"]:
        if not t["selected"]:
            why = []
            if t["gold_len_median"] is not None and t["gold_len_median"] > 25:
                why.append(f"median={t['gold_len_median']}>25")
            if t["replay_scores"] and any(s != 100 for s in t["replay_scores"]):
                why.append(f"replay={t['replay_scores']}")
            if t["errors"]:
                why.append(f"errors={[e['error'] for e in t['errors']]}")
            print(f"  {t['task_name']}: excluded ({'; '.join(why) or 'no data'})", flush=True)
    print(f"task_selection -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
