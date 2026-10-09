# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""R0 for FrozenLake / Sokoban: prompt text, board renderer, action parser.

R0 is the *starting* harness.  It draws its own boards (one symbol per cell) and
does not use ``LevelEnv.observation()["board_text"]``.  The prompt text, the
renderer and the history window are the R0 values of the evolvable harness;
the parser below is fixed for every round and every harness (spec section 5).

``messages(obs, history)`` returns exactly two messages: a system message (game
rules + symbols + move budget) and one user message (board + moves left + the
last three moves with their results).  No earlier turn is sent as chat history.
"""
from __future__ import annotations

import re

ACTIONS = ["up", "down", "left", "right"]

_HISTORY_WINDOW = 3  # "the last three moves of the episode with their results"


# ── system messages ──────────────────────────────────────────────────────────

def system_message(game: str, budget: int, size: int) -> str:
    if game == "frozenlake":
        return (
            "You are playing FrozenLake on a {size}x{size} grid of ice. "
            "Each turn move up, down, left or right to walk the player to the goal.\n"
            "Symbols: P = player, F = ice (safe), H = hole (fall in and lose), G = goal (win).\n"
            "Reach G without stepping on H. You have {budget} moves.\n"
            "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. "
            "Once you've finished your reasoning, present your chosen move within <action> </action> tags (one of up, down, left, right)."
        ).format(size=size, budget=budget)
    return (
        "You are playing Sokoban. Push every box onto a target to win.\n"
        "Symbols: # = wall, _ = floor, O = target, P = player, S = player standing on a target, "
        "B = box, X = box on a target.\n"
        "Move up, down, left or right. A move into a wall, off the board, or pushing a box into a "
        "wall or another box does nothing.\n"
        "You have {budget} moves. "
        "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. "
        "Once you've finished your reasoning, present your chosen move within <action> </action> tags (one of up, down, left, right)."
    ).format(budget=budget)


# ── board renderer (R0 draws its own board) ─────────────────────────────────

def render_board(obs: dict) -> str:
    if obs["game"] == "frozenlake":
        return _render_frozenlake(obs)
    return _render_sokoban(obs)


def _render_frozenlake(obs: dict) -> str:
    size = obs["rows"]
    grid = [["F"] * size for _ in range(size)]
    for (r, c) in obs.get("holes", []):
        grid[r][c] = "H"
    gr, gc = obs["goal"]
    grid[gr][gc] = "G"
    player = obs["player"]
    if player is not None:
        pr, pc = player
        grid[pr][pc] = "P"
    return "\n".join("".join(row) for row in grid)


def _render_sokoban(obs: dict) -> str:
    fixed = obs["fixed"]
    grid = [list(r) for r in fixed]
    targets = {(r, c) for r, row in enumerate(fixed) for c, ch in enumerate(row) if ch == "O"}
    boxes = set(tuple(b) for b in obs.get("boxes", []))
    for (r, c) in boxes:
        grid[r][c] = "X" if (r, c) in targets else "B"
    pr, pc = obs["player"]
    grid[pr][pc] = "S" if (pr, pc) in targets else "P"  # keep target visible
    return "\n".join("".join(row) for row in grid)


# ── messages ────────────────────────────────────────────────────────────────

def messages(obs: dict, history: list) -> list[dict]:
    """Two messages: system + user.  ``history`` = list of ``(action, result)``."""
    game = obs["game"]
    system = system_message(game, obs["budget"], obs["rows"])
    board = render_board(obs)

    lines = ["Board:", board, "", f"Moves left: {obs['moves_left']}"]
    if history:
        lines.append("Last moves:")
        for action, result in history[-_HISTORY_WINDOW:]:
            a = action if action is not None else "(unreadable)"
            lines.append(f"  {a} -> {result}")
    user = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── action parser (fixed for every round and every harness) ─────────────────

_BRACKETED = re.compile(r"\[(up|down|left|right)\]", re.IGNORECASE)
_ACTION_TAG = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_DIR_WORD = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)


def _after_think(reply: str) -> str:
    """Keep only the text after the last ``</think>``; an unclosed ``<think>``
    drops the rest of the reply (spec section 5, step 1)."""
    close = reply.rfind("</think>")
    if close != -1:
        return reply[close + len("</think>"):]
    open_ = reply.find("<think>")
    if open_ != -1:
        return reply[:open_]
    return reply


def parse_action(reply: str) -> str | None:
    """Parse a reply into a direction, or ``None`` when unreadable.

    Primary protocol (mirrors ALFWorld ``extract_action``): the chosen move is
    inside ``<action>…</action>``, so only that payload is read — the
    ```` reasoning is never scanned for a move.  Falls back to the legacy
    whole-reply bracket / lone-word scan when there is no ``<action>`` tag.
    """
    m = _ACTION_TAG.search(reply)
    if m:
        payload = m.group(1).strip()
        bracketed = [x.group(1).lower() for x in _BRACKETED.finditer(payload)]
        if bracketed:
            return bracketed[-1]
        dm = _DIR_WORD.search(payload)
        return dm.group(1).lower() if dm else None

    # Legacy fallback (no <action> tag): R0's original bracket / lone-word rules.
    text = _after_think(reply)
    bracketed = [x.group(1).lower() for x in _BRACKETED.finditer(text)]
    if bracketed:
        return bracketed[0] if len(set(bracketed)) == 1 else None
    stripped = text.strip()
    for punct in (".", ",", "!", "?"):
        if stripped.endswith(punct):
            stripped = stripped[:-1].strip()
    word = stripped.lower()
    return word if word in ACTIONS else None
