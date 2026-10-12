"""Fixed-format trajectory rendering for TextCraft (TEXTCRAFT_SPEC.md §8).

``render_for_reader(traj) -> str`` produces a harness-independent view of a
trajectory (goal, recipe list, per-step action / raw env return / inventory,
final result) for the kev reader. It must NOT contain the agent-facing prompt,
so it stays stable as the harness prompt evolves. The same function signature is
used by ScienceWorld (envs/sciworld/render.py).
"""

from __future__ import annotations

from typing import Dict, List

from .vendor.utils import item_id_to_str


def _fmt_inventory(inventory: Dict) -> str:
    if not inventory:
        return "(empty)"
    parts = [f"{item_id_to_str(k)} x{v}" for k, v in sorted(inventory.items())]
    return ", ".join(parts)


def render_for_reader(record: Dict) -> str:
    lines: List[str] = []
    lines.append(f"Goal: craft {record['goal']}")
    lines.append("Recipes:")
    for r in record["commands_text"].split("\n"):
        lines.append(f"- {r}")
    steps: List[Dict] = record.get("steps", [])
    for i, s in enumerate(steps, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {s.get('action', '')}")
        lines.append(f"  Env: {s.get('env_message', '')}")
        lines.append(f"  Inventory: {_fmt_inventory(s.get('inventory', {}))}")
    result = "success" if record.get("success") else "failed"
    lines.append(f"Result: {result} (steps={len(steps)})")
    return "\n".join(lines)
