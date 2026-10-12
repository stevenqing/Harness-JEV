"""ScienceWorld held-in / held-out split (SCIWORLD_SPEC.md §3.2).

From the tasks selected by ``select_tasks.py``, build 64 held-in episodes (from
*train* variations) and 64 held-out episodes (from *test* variations).  Each
selected task gets an as-equal-as-possible share; the remainder goes to the
tasks with the most variations of that split.  Variations are taken by
increasing index (no random sampling).  Deterministic.

Run inside the py3.8 ``ee-sciworld`` env::

    ee-sciworld/bin/python -m recipe.agent_evolver.envs.sciworld.splits
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SELECTION_PATH = os.path.join(HERE, "task_selection.json")
SPLITS_PATH = os.path.join(HERE, "splits.json")

HELDIN_N = 64
HELDOUT_N = 64


def _episode_id(task_name: str, variation: int) -> str:
    return f"{task_name}_{variation}"


def _distribute(tasks: List[str], counts: Dict[str, List[int]], total: int) -> Dict[str, int]:
    """Evenly split ``total`` episodes across ``tasks``; remainder to the tasks
    with the most variations (ties broken by name for determinism).

    A task whose allocation exceeds its available variations is capped, and the
    shortfall is redistributed to the tasks with the most spare variations, so
    the total always reaches ``total``."""
    n = len(tasks)
    base, rem = divmod(total, n)
    order = sorted(tasks, key=lambda t: (-len(counts[t]), t))
    alloc = {t: min(base + (1 if t in order[:rem] else 0), len(counts[t])) for t in tasks}
    shortfall = total - sum(alloc.values())
    i = 0
    while shortfall > 0 and i < n * total:
        t = order[i % n]
        if alloc[t] < len(counts[t]):
            alloc[t] += 1
            shortfall -= 1
        i += 1
    return alloc


def main() -> None:
    selection = json.load(open(SELECTION_PATH, encoding="utf-8"))
    selected = [t["task_name"] for t in selection["tasks"] if t["selected"]]

    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv(envStepLimit=100)
    train_var: Dict[str, List[int]] = {}
    test_var: Dict[str, List[int]] = {}
    try:
        for t in selected:
            # get_variations_*() is per-task on the Java side; load variation 0
            # for each task first.
            env.load(t, 0, selection["simplification"])
            train_var[t] = sorted(env.get_variations_train())
            test_var[t] = sorted(env.get_variations_test())
    finally:
        env.close()

    heldin_alloc = _distribute(selected, train_var, HELDIN_N)
    heldout_alloc = _distribute(selected, test_var, HELDOUT_N)

    heldin: List[str] = []
    heldout: List[str] = []
    episodes: Dict[str, Dict] = {}
    for t in selected:
        for v in train_var[t][: heldin_alloc[t]]:
            eid = _episode_id(t, v)
            heldin.append(eid)
            episodes[eid] = {"task_name": t, "variation": v, "split": "heldin"}
        for v in test_var[t][: heldout_alloc[t]]:
            eid = _episode_id(t, v)
            heldout.append(eid)
            episodes[eid] = {"task_name": t, "variation": v, "split": "heldout"}

    heldin.sort()
    heldout.sort()

    out = {"heldin": heldin, "heldout": heldout, "episodes": episodes}
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"selected tasks: {len(selected)}")
    print(f"heldin: {len(heldin)} episodes")
    print(f"heldout: {len(heldout)} episodes")
    for t in selected:
        print(f"  {t}: heldin={heldin_alloc[t]} (train {len(train_var[t])}) "
              f"heldout={heldout_alloc[t]} (test {len(test_var[t])})")
    print(f"splits -> {SPLITS_PATH}")


if __name__ == "__main__":
    main()
