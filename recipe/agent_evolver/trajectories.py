# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Per-task trajectory ``.md`` writer for the text-protocol evolver.

The record shape is the one returned by ``agents.run_alfworld`` /
``agents.run_webshop``: ``task_id``, ``success``, ``reward``, ``steps``,
``goal``, ``trajectory`` (list of ``{step, prompt, obs, adm, raw, action,
invalid}``) and ``final_obs``.

Frontmatter carries the ground-truth outcome the meta-agent's generic read
pattern relies on (``eval_passed`` / ``eval_score`` / ``eval_reason``) plus the
behaviour tier (steps / messages / invalid-action count).  The body renders the
step-by-step model output → extracted action so the meta-agent can read the
*why*.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _scalar(v: Any) -> str:
    if v is None:
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    s = str(v).replace("\n", " ").strip()
    return json.dumps(s, ensure_ascii=False)


def _derive(record: dict, max_steps: int) -> tuple[str, str]:
    success = bool(record.get("success"))
    steps = int(record.get("steps") or 0)
    if success:
        return "done", "won"
    if steps >= max_steps:
        return "max_steps", "incomplete"
    return "done", "failed"


def render_frontmatter(record: dict, max_steps: int) -> str:
    reward = float(record.get("reward") or 0.0)
    eval_passed = bool(record.get("success"))
    exit_reason, eval_reason = _derive(record, max_steps)
    steps = int(record.get("steps") or 0)
    traj = record.get("trajectory") or []
    n_invalid = sum(1 for s in traj if s.get("invalid"))

    fields: list[tuple[str, Any]] = [
        ("task_id", record.get("task_id")),
        ("goal", record.get("goal") or ""),
        ("exit_reason", exit_reason),
        ("steps", steps),
        ("num_messages", steps),  # one model call per env step
        ("invalid_actions", n_invalid),
        ("reward", reward),
        ("eval_passed", eval_passed),
        ("eval_score", reward),
        ("eval_reason", eval_reason),
    ]

    if not eval_passed:
        fields.append(("final_obs", record.get("final_obs") or ""))

    # WebShop: the env computes the reward decomposition (``reward_info``) only at
    # buy time; persist it so the progress-diagnosis oracle can be recovered
    # post-hoc from the ``.md`` (absent on ALFWorld / TextCraft / ScienceWorld).
    if "reward_info" in record:
        fields.append(("reward_info", record.get("reward_info")))

    for k, v in (record.get("frontmatter_extra") or {}).items():
        fields.append((k, v))

    lines = ["---"]
    for k, v in fields:
        lines.append(f"{k}: {_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


def render_body(record: dict) -> str:
    lines: list[str] = []
    lines.append("## Task")
    lines.append(str(record.get("goal") or ""))
    lines.append("")
    lines.append("## Result")
    lines.append(f"success={bool(record.get('success'))}  steps={record.get('steps')}  "
                 f"reward={record.get('reward')}")
    lines.append("")

    traj = record.get("trajectory") or []
    if not traj:
        lines.append("(no trajectory steps recorded)")
        return "\n".join(lines)

    lines.append("## Execution Steps")
    for step in traj:
        lines.append(f"\n### Step {step.get('step')}")
        lines.append(f"**Observation**: {step.get('obs', '')}")
        lines.append("")
        lines.append("**Model output (raw)**:")
        lines.append("```")
        lines.append(step.get("raw", ""))
        lines.append("```")
        lines.append(f"**Extracted action**: `{step.get('action', '')}`  "
                     f"(invalid={bool(step.get('invalid'))})")
        # Benchmark-specific step info (TextCraft / ScienceWorld) — rendered only
        # when present so ALFWorld / WebShop bodies are unchanged.
        if "raw_action" in step:
            lines.append(f"**Raw action**: `{step.get('raw_action', '')}`")
        if "env_message" in step:
            lines.append(f"**Env return**: {step.get('env_message', '')}")
        if "inventory" in step:
            lines.append(f"**Inventory**: {_scalar(step.get('inventory', {}))}")
        if "in_admissible" in step:
            lines.append(f"**in_admissible**: {bool(step.get('in_admissible'))}")
        if "score" in step:
            lines.append(f"**Score**: {step.get('score', '')}")
        if "in_valid" in step:
            lines.append(f"**in_valid**: {bool(step.get('in_valid'))}")
        if "auto_disambiguated" in step:
            lines.append(f"**auto_disambiguated**: {bool(step.get('auto_disambiguated'))}")
    return "\n".join(lines)


def write_task_trajectory(traj_dir: Path, record: dict, max_steps: int) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    fm = render_frontmatter(record, max_steps)
    body = render_body(record)
    (traj_dir / f"{record['task_id']}.md").write_text(f"{fm}\n\n{body.lstrip()}", encoding="utf-8")
