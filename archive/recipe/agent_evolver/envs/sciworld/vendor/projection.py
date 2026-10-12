"""Text action projection for the native ScienceWorld command interface."""

import json
import re
from typing import List, Optional, Tuple


def _extract_action(text: str) -> Optional[str]:
    text = str(text).strip()
    tagged = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    action_line = re.search(r"(?:^|\n)\s*Action\s*:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if action_line:
        return action_line.group(1).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("action"), str):
            return value["action"].strip()
    except (TypeError, ValueError):
        pass
    return text


def scienceworld_projection(
    actions: List[str], action_pools: Optional[List[List[str]]] = None
) -> Tuple[List[str], List[int]]:
    projected = []
    valids = []
    for index, raw_action in enumerate(actions):
        action = _extract_action(raw_action)
        action = re.sub(r"\s+", " ", action or "").strip().rstrip(".")
        admissible = action_pools[index] if action_pools is not None else []
        valid = int(bool(action) and (not admissible or action in admissible))
        projected.append(action or "look around")
        valids.append(valid)
    return projected, valids
