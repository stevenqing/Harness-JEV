"""Failure labelling for TextCraft (TEXTCRAFT_SPEC.md §7).

Reads only the trajectory, the task's manifest row, and the environment
structure (no model calls). Produces a per-step leaf label and a per-episode
main label. Leaf labels:

  format / get_craftable / get_tag / get_unknown_form / missing_intermediate /
  missing_base / wrong_count / hallucinated_recipe / craft_invalid /
  crafted_distractor / ok
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .vendor.utils import item_id_to_str

_CRAFT_RE = re.compile(r"^craft\s+(\d+)\s+(.+?)\s+using\s+(.+)$")
_GET_RE = re.compile(r"^get\s+(\d+)\s+(.+)$")
_CRAFTED_RE = re.compile(r"^Crafted\s+\d+\s+(minecraft:\S+)")


def _to_item_id(human: str) -> str:
    return "minecraft:" + human.replace(" ", "_")


def parse_craft(action: str):
    m = _CRAFT_RE.match(action)
    if not m:
        return None
    out_human = m.group(2).strip()
    ings = []
    for part in m.group(3).split(","):
        part = part.strip()
        im = re.match(r"(\d+)\s+(.+)", part)
        if im:
            ings.append((im.group(2).strip(), int(im.group(1))))
    return {"output": out_human, "ingredients": ings}


def _gold_output_ids(row: Dict) -> set:
    return {r["output"] for r in row["gold_recipes"]}


def _known_recipe_ids(row: Dict) -> List[Dict]:
    return row["gold_recipes"] + row["distractor_recipes"]


def _matches_known(parsed, row: Dict) -> bool:
    if parsed is None:
        return False
    want_out = parsed["output"]
    want_ings = Counter({i: c for i, c in parsed["ingredients"]})
    for r in _known_recipe_ids(row):
        if item_id_to_str(r["output"]) != want_out:
            continue
        have_ings = Counter({item_id_to_str(i["item"]): i["count"] for i in r["ingredients"]})
        if have_ings == want_ings:
            return True
    return False


def step_label(tree, row: Dict, action: str, env_message: str, inventory: Dict) -> str:
    msg = env_message
    if msg.startswith("Could not execute") or msg.startswith("Wrong item format"):
        return "format"

    if msg.startswith("Could not find enough items to craft"):
        parsed = parse_craft(action)
        if not _matches_known(parsed, row):
            return "hallucinated_recipe"
        gold_out = _gold_output_ids(row)
        missing_intermediate = False
        missing_base = False
        wrong_count = False
        for item_human, count in parsed["ingredients"]:
            item_id = _to_item_id(item_human)
            have = inventory.get(item_id, 0)
            if have == 0:
                if item_id in gold_out:
                    missing_intermediate = True
                else:
                    missing_base = True
            elif have < count:
                wrong_count = True
        if missing_intermediate:
            return "missing_intermediate"
        if missing_base:
            return "missing_base"
        if wrong_count:
            return "wrong_count"
        return "craft_missing"

    if msg.startswith("Could not find a valid recipe for"):
        return "craft_invalid"

    if msg.startswith("Could not find"):
        gm = _GET_RE.match(action)
        item_id = _to_item_id(gm.group(2)) if gm else ""
        if item_id and item_id in _gold_output_ids(row):
            return "get_craftable"
        if item_id and tree.is_tag(item_id):
            return "get_tag"
        return "get_unknown_form"

    if msg.startswith("Crafted"):
        m = _CRAFTED_RE.match(msg)
        if m and m.group(1) in _gold_output_ids(row):
            return "ok"
        return "crafted_distractor"

    return "ok"


def label_trajectory(
    tree, row: Dict, steps: List[Dict], success: bool
) -> Dict:
    """steps: list of dicts with keys action / env_message / inventory."""
    labels: List[str] = []
    attempted_goal = False
    goal_human = row["goal"]
    for s in steps:
        lab = step_label(tree, row, s["action"], s["env_message"], s.get("inventory", {}))
        labels.append(lab)
        if parse_craft(s["action"]):
            p = parse_craft(s["action"])
            if p["output"] == goal_human:
                attempted_goal = True

    label_counts = Counter(labels)

    # episode label by priority
    if success:
        episode = "success"
    elif len(labels) and label_counts["format"] * 2 > len(labels):
        episode = "format_dominant"
    elif len(labels) >= 6 and len({s["action"] for s in steps[-6:]}) <= 2:
        episode = "loop"
    elif not attempted_goal:
        episode = "never_attempted_goal"
    else:
        last_fail = None
        for lab in reversed(labels):
            if lab != "ok":
                last_fail = lab
                break
        if last_fail in ("missing_intermediate", "get_craftable"):
            episode = "stuck_intermediate"
        elif last_fail in ("wrong_count", "craft_invalid"):
            episode = "stuck_count"
        else:
            episode = "timeout_other"

    # progress: fraction of gold-tree intermediates present in final inventory
    gold_out = _gold_output_ids(row)
    intermediates = gold_out - {row["goal_id"]}
    final_inv = steps[-1].get("inventory", {}) if steps else {}
    if intermediates:
        progress = sum(1 for i in intermediates if i in final_inv) / len(intermediates)
    else:
        progress = 1.0

    return {
        "step_labels": labels,
        "episode_label": episode,
        "label_counts": dict(label_counts),
        "progress": progress,
    }
