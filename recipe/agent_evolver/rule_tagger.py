# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Deterministic rule taggers (visibility spec A2).

Each rule reads *only* the trajectory text (the same input kev reads) and emits
the same closed-set label as A1.  It is the dumb baseline kev must beat where its
signal is real (ALFWorld — the rule is target-agnostic) and can only tie where
the mode is state-readable (gridgames — the rule reads ``eval_reason`` + the
action sequence, i.e. exactly what kev's rendered state carries, so kev is held
to the same information ceiling).
"""
from __future__ import annotations

from .failure_labels import has_loop, parse_agent_action

# verbs that count as "made progress past acquisition" for the ALF rule.  Deliberately
# target-agnostic: the rule does not know which object the goal asks for.
_ALF_PROGRESS_VERBS = {"put", "clean", "heat", "cool", "slice", "use", "toggle"}


def alfworld_rule(actions: list[str]) -> str:
    """``find`` = no ``take`` at all; ``acquire`` = ``take`` but no post-take
    progress verb; ``other`` = ``take`` + a progress verb.

    The blindness to *which* object the goal wants is exactly where kev's
    semantic read should beat this rule (it also under-splits the ``acquire`` /
    ``other`` boundary on two-object tasks — a known weakness).
    """
    parsed = [parse_agent_action(a) for a in actions]
    took = any(v == "take" for v, _ in parsed)
    if not took:
        return "find"
    progressed = any(v in _ALF_PROGRESS_VERBS for v, _ in parsed)
    return "acquire" if not progressed else "other"


def gridgames_rule(game: str, eval_reason: str, actions: list[str | None]) -> str:
    """Text-only rule for gridgames — the same input ceiling as kev.

    Reads ``eval_reason`` + the action sequence (exactly what ``_render_state``
    hands kev) and maps it onto the 4-way A1 label:

    * ``failed`` -> ``fell_in_hole`` (only FrozenLake dies mid-episode; Sokoban
      never sets ``done`` except on success/budget, so ``dead_push`` is the one
      mode the trajectory text does *not* carry — the rule folds it into
      ``budget``, and kev is held to that same ceiling);
    * otherwise budget-out -> ``loop`` if the actions collapse, else ``budget``.
    """
    acts = [a for a in actions if a is not None]
    if (eval_reason or "").strip().lower() == "failed":
        return "fell_in_hole"
    return "loop" if has_loop(acts) else "budget"
