"""Failure labelling for ScienceWorld (SCIWORLD_SPEC.md §8.1).

Reads only the trajectory, the per-step env info, and the environment structure
(goal-progress string / gold sequence); no model calls.  Produces a per-step
leaf label and a per-episode main label.

Step labels (first match wins):

  invalid_action  env returned "No known action matches that input."
  ambiguous       this model action triggered an ambiguity menu (auto-answered 0)
  wrong_focus     action starts with "focus on" and the score dropped to -100
  progress        score strictly higher than the previous step
  no_progress     everything else

Episode labels (priority, first match wins):

  success / wrong_focus / invalid_dominant / loop / stalled_partial /
  stalled_zero / timeout_partial

Also parses ``get_goal_progress()`` into ordered / unordered subgoal lists and
records the name of the first uncompleted *ordered* subgoal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

_INVALID_MSG = "No known action matches that input."


def parse_goal_progress(text: str) -> tuple[List[Dict], List[Dict]]:
    """→ (ordered, unordered) subgoal lists.

    Each subgoal is ``{index, completed, type, name}``.  ``name`` is the
    human-readable goal description (e.g. "focus on substance"); ``type`` is the
    ScienceWorld goal class (e.g. "GoalFind").  The raw output has two sections
    — "Sequential Subgoals" (→ ordered) and "Unordered and Optional Subgoals"
    (→ unordered, the optional goals are folded in; 1.2.3 does not separate
    them).
    """
    ordered: List[Dict] = []
    unordered: List[Dict] = []
    section: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Sequential Subgoals"):
            section = "ordered"
            continue
        if stripped.startswith("Unordered"):
            section = "unordered"
            continue
        if stripped.startswith("Completed keys"):
            section = None
            continue
        if section is None or not stripped or stripped.startswith("---"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        item = {
            "index": idx,
            "completed": parts[1].strip().lower() == "true",
            "type": parts[2].strip(),
            "name": " ".join(p.strip() for p in parts[3:]).strip(),
        }
        (ordered if section == "ordered" else unordered).append(item)
    return ordered, unordered


def first_unmet_subgoal(ordered: List[Dict]) -> Optional[str]:
    for sg in ordered:
        if not sg["completed"]:
            return sg["name"]
    return None


def step_label(action: str, env_message: str, prev_score: int, score: int,
               auto_disambiguated: bool) -> str:
    if _INVALID_MSG in env_message:
        return "invalid_action"
    if auto_disambiguated:
        return "ambiguous"
    if action.startswith("focus on") and score < 0:
        return "wrong_focus"
    if score > prev_score:
        return "progress"
    return "no_progress"


def _distinct_last_n(values: List[str], n: int) -> int:
    return len(set(values[-n:])) if values else 0


def label_trajectory(
    steps: List[Dict],
    goal_progress: str,
    success: bool,
) -> Dict:
    """steps: list of dicts with action / env_message / score /
    auto_disambiguated / invalid (parse-fail)."""
    ordered, unordered = parse_goal_progress(goal_progress)

    labels: List[str] = []
    prev = 0
    for s in steps:
        score = int(s.get("score", 0))
        labels.append(step_label(
            s.get("action", ""), s.get("env_message", ""), prev, score,
            bool(s.get("auto_disambiguated")),
        ))
        prev = score

    n_steps = len(steps) or 1
    n_invalid_action = labels.count("invalid_action")
    n_parse_fail = sum(1 for s in steps if s.get("invalid"))
    scores = [int(s.get("score", 0)) for s in steps]
    final_score = scores[-1] if scores else 0
    actions = [s.get("action", "") for s in steps]

    # episode label by priority
    if success:
        episode = "success"
    elif "wrong_focus" in labels:
        episode = "wrong_focus"
    elif n_invalid_action + n_parse_fail > n_steps / 2:
        episode = "invalid_dominant"
    elif len(actions) >= 6 and _distinct_last_n(actions, 6) <= 2:
        episode = "loop"
    elif final_score > 0 and len(scores) >= 10 and len(set(scores[-10:])) == 1:
        episode = "stalled_partial"
    elif final_score == 0:
        episode = "stalled_zero"
    else:
        episode = "timeout_partial"

    return {
        "step_labels": labels,
        "episode_label": episode,
        "first_unmet_subgoal": first_unmet_subgoal(ordered),
        "subgoals": {"ordered": ordered, "unordered": unordered},
    }
