"""Fixed-format trajectory rendering for ScienceWorld (SCIWORLD_SPEC.md §8.2).

``render_for_reader(traj) -> str`` produces a harness-independent view of a
trajectory for the kev reader: the task description, each step's action + raw
environment return + score, and the final result.  It must NOT contain the
agent-facing prompt, so it stays stable as the harness prompt evolves.  The same
signature is used by TextCraft (envs/textcraft/render.py).
"""

from __future__ import annotations

from typing import Dict, List


def render_for_reader(record: Dict) -> str:
    lines: List[str] = []
    lines.append(f"Task: {record.get('goal', '')}")
    lines.append(f"task_name={record.get('task_name', '')} variation={record.get('variation', '')}")
    steps: List[Dict] = record.get("trajectory") or []
    for i, s in enumerate(steps, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {s.get('action', '')}")
        lines.append(f"  Env: {s.get('env_message', '')}")
        lines.append(f"  Score: {s.get('score', '')}")
    result = "success" if record.get("success") else "failed"
    lines.append(f"Result: {result} (score={record.get('reward', 0.0):.3f}, steps={len(steps)})")
    return "\n".join(lines)
