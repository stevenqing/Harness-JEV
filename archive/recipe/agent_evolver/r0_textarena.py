# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""TextArena-aligned R0 variant for FrozenLake / Sokoban (experiment).

This is **not** the spec's R0 (``r0.py``).  It is an experimental harness that
swaps R0's prompt, board glyphs and action parser for TextArena's own
(https://github.com/TextArena/TextArena, ``textarena/envs/{FrozenLake,Sokoban}``),
to test whether the R0 baseline's collapse is caused by R0's custom symbols /
wording rather than by the models or the decoding.

Differences from ``r0.py`` (spec R0):

* FrozenLake ice is a space ``' '`` (not ``F``) and the board is TextArena's
  ruled table (``+---+`` cells), as in ``FrozenLakeEnv._render_board``.
* Sokoban box-off-target is ``X`` and box-on-target is ``√`` — R0 used ``B`` /
  ``X``, i.e. ``X`` had the *opposite* meaning.  Player on a target is ``S``.
  The board is space-separated, as in ``SokobanEnv.create_board_str``.
* The parser also accepts ``[w]/[a]/[s]/[d]`` (mapped to up/left/down/right),
  matching TextArena's ``step`` parser.  This is the one change to the "fixed"
  section-5 parser, recorded as a deviation for this experiment.
* The system message is TextArena's ``player_prompt.intro`` text verbatim, plus
  one added line stating the move budget (TextArena uses an implicit turn limit;
  this env ends the episode at ``budget``, so the model is told the number).

Everything else — two messages (system + user), the last-three-move history
window, the seed scheme, the decoding — is unchanged from R0 so the comparison
is a one-variable prompt swap.
"""
from __future__ import annotations

import re

ACTIONS = ["up", "down", "left", "right"]
_ALIAS = {"w": "up", "a": "left", "s": "down", "d": "right"}

_HISTORY_WINDOW = 3  # same as r0.py

# TextArena ``player_prompt.intro`` (FrozenLake/locales/en.json).  One line is
# changed: TextArena's "Navigate from the start (top-left) to the goal
# (bottom-right)" is replaced because this manifest randomizes start/goal, so
# the literal claim would mislead the model.  Everything else is verbatim.
_FL_INTRO = (
    "Welcome to Frozen Lake!\n\n"
    "You are represented by 'P' on the grid.\n"
    "Grid symbols:\n"
    "  ' ' = Frozen surface (safe to walk on)\n"
    "  'H' = Hole (fall in and lose!)\n"
    "  'G' = Goal (reach this to win!)\n"
    "  'P' = Your current position\n\n"
    "Available actions: up, down, left, right (or w, a, s, d)\n"
    "Type your action as: [up], [down], [left], [right] or [w], [a], [s], [d]\n\n"
    "Objective: Navigate from your current position ('P') to the goal ('G') "
    "without falling into any holes!"
)

# TextArena ``player_prompt.intro`` (Sokoban/locales/en.json), verbatim.
_SK_INTRO = (
    "You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets.\n"
    "When you are right next to a box, you can push it by moving in the same direction.\n"
    "You cannot push a box through a wall, and you cannot pull a box.\n"
    "On the board, objects are represented as:\n"
    "  - The player (you) appears as 'P'\n"
    "  - Walls are represented with '#'\n"
    "  - Boxes are marked as 'X'\n"
    "  - Empty goals are shown with a 'O'\n"
    "  - Boxes on goals are visualized with '√'\n"
    "You can also use [w] for up, [a] for left, [s] for down, and [d] for right."
)


def system_message(game: str, budget: int, size: int) -> str:
    """TextArena intro + one added budget line."""
    if game == "frozenlake":
        return _FL_INTRO + f"\n\nYou have {budget} moves."
    return _SK_INTRO + f"\n\nYou have {budget} moves."


# ── board renderer (TextArena formats) ───────────────────────────────────────

def render_board(obs: dict) -> str:
    if obs["game"] == "frozenlake":
        return _render_frozenlake(obs)
    return _render_sokoban(obs)


def _render_frozenlake(obs: dict) -> str:
    """TextArena ``FrozenLakeEnv._render_board``: a ruled ``+---+`` table,
    ``' '`` = ice, ``H`` = hole, ``G`` = goal, ``P`` = player."""
    size = obs["rows"]
    grid = [[" "] * size for _ in range(size)]
    for (r, c) in obs.get("holes", []):
        grid[r][c] = "H"
    gr, gc = obs["goal"]
    grid[gr][gc] = "G"
    player = obs["player"]
    if player is not None:
        pr, pc = player
        grid[pr][pc] = "P/G" if (pr, pc) == (gr, gc) else "P"

    cell_width = 3
    hline = "+" + "+".join(["-" * (cell_width + 2) for _ in range(size)]) + "+"
    lines = [hline]
    for r in range(size):
        cells = [f" {grid[r][c]:^{cell_width}} " for c in range(size)]
        lines.append("|" + "|".join(cells) + "|")
        lines.append(hline)
    return "\n".join(lines)


def _render_sokoban(obs: dict) -> str:
    """TextArena ``SokobanEnv.create_board_str``: space-separated glyphs
    ``#`` wall, ``_`` floor, ``O`` empty target, ``X`` box, ``√`` box-on-target,
    ``P`` player, ``S`` player-on-target."""
    fixed = obs["fixed"]
    targets = {(r, c) for r, row in enumerate(fixed) for c, ch in enumerate(row) if ch == "O"}
    grid = [list(r) for r in fixed]  # '#' / '_' / 'O'
    boxes = set(tuple(b) for b in obs.get("boxes", []))
    pr, pc = obs["player"]
    for (r, c) in boxes:
        grid[r][c] = "√" if (r, c) in targets else "X"
    grid[pr][pc] = "S" if (pr, pc) in targets else "P"
    return "\n".join(" ".join(row) for row in grid)


# ── messages ────────────────────────────────────────────────────────────────

def messages(obs: dict, history: list) -> list[dict]:
    """Two messages: system + user.  Same structure as r0.py — only the intro,
    the glyphs and the parser differ."""
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


# ── action parser (R0's 3 steps + TextArena's wasd aliases) ─────────────────

_BRACKETED = re.compile(r"\[(up|down|left|right|w|a|s|d)\]", re.IGNORECASE)


def _normalize(token: str) -> str:
    return _ALIAS.get(token, token)


def _after_think(reply: str) -> str:
    """Keep only the text after the last ``</think>``; an unclosed ``<think>``
    drops the rest of the reply (identical to r0.py)."""
    close = reply.rfind("</think>")
    if close != -1:
        return reply[close + len("</think>"):]
    open_ = reply.find("<think>")
    if open_ != -1:
        return reply[:open_]
    return reply


def parse_action(reply: str) -> str | None:
    """Parse a reply into a direction, or ``None`` when unreadable."""
    text = _after_think(reply)

    # Step 2: bracketed moves must all name the same direction (after wasd map).
    bracketed = [_normalize(m.group(1).lower()) for m in _BRACKETED.finditer(text)]
    if bracketed:
        return bracketed[0] if len(set(bracketed)) == 1 else None

    # Step 3: without a bracketed move, only a lone direction word (or wasd key).
    stripped = text.strip()
    for punct in (".", ",", "!", "?"):
        if stripped.endswith(punct):
            stripped = stripped[:-1].strip()
    word = _normalize(stripped.lower())
    return word if word in ACTIONS else None
