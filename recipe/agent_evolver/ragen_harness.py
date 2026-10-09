# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""RAGen-style harness text + parser for FrozenLake / Sokoban.

The one thing RAGen's FrozenLake/Sokoban envs do that our bracket-based R0 does
not is a *protocol*: ``<think>`` / ``<answer>`` tags, "no extra text", a short
response budget, thinking OFF, and a ``fullmatch`` parser over the ``<answer>``
payload.  This module exposes that protocol as (a) an evolvable system prompt
and (b) a fixed parser, so ``MetaAgent.evolve()`` can evolve the RAGen-style
harness exactly the way it evolves the bracket harness.

The parser is fixed (spec section 5): extract the ``<answer>`` payload and
``fullmatch`` it against one direction word; anything else is "invalid"
(LevelEnv treats ``None`` as a no-op / parse_fail — RAGen's "invalid action").
"""
from __future__ import annotations

import re

ACTIONS = ["up", "down", "left", "right"]

FORMAT_DIRECTIVE = (
    "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> "
    "with no extra text. Strictly follow this format. "
    "Your answer must be exactly one of: up, down, left, right."
)


def system_message(game: str, budget: str, size: str) -> str:
    """Game rules + symbols, RAGen-style (no bracket hint).  ``budget``/``size``
    are format *values* so callers can pass the literal ``"{budget}"``/``"{size}"``
    to leave the placeholders intact for the rollout's ``str.replace``."""
    if game == "frozenlake":
        return (
            "You are playing FrozenLake on a {size}x{size} grid of ice. "
            "Each turn move up, down, left or right to walk the player to the goal.\n"
            "Symbols: P = player, F = ice (safe), H = hole (fall in and lose), G = goal (win).\n"
            "Reach G without stepping on H. You have {budget} moves.\n"
            "Your move must be exactly one of: up, down, left, right."
        ).format(size=size, budget=budget)
    return (
        "You are playing Sokoban. Push every box onto a target to win.\n"
        "Symbols: # = wall, _ = floor, O = target, P = player, S = player standing on a target, "
        "B = box, X = box on a target.\n"
        "Move up, down, left or right. A move into a wall, off the board, or pushing a box into a "
        "wall or another box does nothing.\n"
        "You have {budget} moves. Your move must be exactly one of: up, down, left, right."
    ).format(budget=budget)


def baseline_system_prompt(game: str) -> str:
    """RAGen-style baseline system-prompt template (``{size}``/``{budget}`` intact)."""
    return system_message(game, "{budget}", "{size}") + "\n\n" + FORMAT_DIRECTIVE


# ── fixed parser (spec section 5) ─────────────────────────────────────────────

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_DIRECTION = re.compile(r"^\s*(up|down|left|right)\s*$", re.IGNORECASE)


def parse_action(reply: str) -> str | None:
    """Extract the ``<answer>`` payload and ``fullmatch`` one direction word.

    ``None`` = invalid (RAGen's "invalid action"; LevelEnv no-op).  Mirrors
    RAGen ``extract_action``: first ``<answer>`` wins, else the whole reply."""
    m = _ANSWER.search(reply)
    payload = m.group(1).strip() if m else reply.strip()
    mm = _DIRECTION.fullmatch(payload)
    return mm.group(1).lower() if mm else None
