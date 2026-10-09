# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Evolvable loop processors for the agent ReAct loop.

Each processor is a small, parameterizable rule that steers a weak 4B model
away from its most common failure modes (loops, revisiting already-tried
actions, misreading long observations).

These are *not* model weights and *not* prompt text — they are deterministic
rules applied to the running episode state (history + admissible actions) before
the model is queried.

History entry shape throughout this module: ``(obs: str, action: str, valid: bool)``
where ``valid`` is False for actions that were not in the admissible set.
"""
from __future__ import annotations

from .spec import HISTORY_LENGTH

# Default parameters for each processor. ``processors`` on HarnessSpec is a
# sparse dict keyed by processor name; missing keys fall back to these.
PROCESSOR_DEFAULTS: dict[str, dict] = {
    "loop_breaker": {"k": 3},
    "history_format": {"mode": "raw", "n": HISTORY_LENGTH, "obs_clip": 150},
    "tried_action_mask": {"enabled": False},
}


def _cfg(processors: dict, name: str) -> dict:
    cfg = dict(PROCESSOR_DEFAULTS.get(name, {}))
    cfg.update(processors.get(name) or {})
    return cfg


# ─── history_format ─────────────────────────────────────────────────────────

def render_history(history: list, processors: dict) -> tuple[str, int]:
    """Return ``(action_history_string, shown_count)`` for the prompt.

    ``mode`` selects what the agent sees about the recent past:
      - "raw":         full observation text (the Skill1 default),
      - "summary":     each observation truncated to ``obs_clip`` chars,
      - "failed-only": only steps whose action was invalid/forced,
      - "successful-only": only steps whose action was valid.
    """
    cfg = _cfg(processors, "history_format")
    mode = cfg.get("mode", "raw")
    n = int(cfg.get("n", HISTORY_LENGTH))
    clip = int(cfg.get("obs_clip", 150))

    recent = history[-n:] if n > 0 else history
    if mode == "failed-only":
        recent = [h for h in recent if not h[2]]
    elif mode == "successful-only":
        recent = [h for h in recent if h[2]]

    start = len(history) - len(recent)
    lines = []
    for j, (obs, act, _valid) in enumerate(recent):
        nstep = start + j + 1
        obs = obs if mode != "summary" else _truncate(obs, clip)
        lines.append(f"[Observation {nstep}: '{obs}', Action {nstep}: '{act}']")
    return "\n".join(lines), len(recent)


def _truncate(text: str, n: int) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "…"


# ─── action masking ─────────────────────────────────────────────────────────

def mask_admissible(history: list, admissible: list[str], processors: dict) -> list[str]:
    """Filter the admissible-action list before it is shown to the model.

    - ``tried_action_mask`` (enabled=true): drop actions that were already tried
      this episode (prevents revisiting).
    - ``loop_breaker``: drop the single action that repeats for the last ``k``
      steps (breaks a hard loop).

    Never masks down to an empty list — if every action would be masked, return
    the original list so the loop can still make progress.
    """
    result = [a for a in admissible]
    tried = [h[1] for h in history]

    if _cfg(processors, "tried_action_mask").get("enabled"):
        result = [a for a in result if a not in set(tried)]

    lb = _cfg(processors, "loop_breaker")
    k = int(lb.get("k", 3))
    if k > 0 and len(tried) >= k and len(set(tried[-k:])) == 1:
        repeat = tried[-1]
        result = [a for a in result if a != repeat]

    return result if result else list(admissible)
