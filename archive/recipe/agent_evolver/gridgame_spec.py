# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Evolvable harness spec for FrozenLake / Sokoban (gridgames).

Mirrors ``spec.py``'s ``HarnessSpec`` for the ALF/WS text loop.  The gridgame
loop is R0 (spec section 5): one model call per move, exactly two messages
(system + user), a fixed action parser.  The evolvable values are:

* ``system_prompt`` — the game rules + symbol text (R0's ``system_message``
  template, ``{size}``/``{budget}`` placeholders intact so the rollout can
  substitute them with ``str.replace`` without disturbing text the meta-agent
  may have rewritten).
* ``guidance``      — a hint appended to the system message every turn (the
  primary lever, mirroring the ALF/WS ``{skills}`` slot).
* ``history_window`` — how many recent moves are shown (R0 ``_HISTORY_WINDOW``).

The board renderer and the action parser are **frozen** for this first Phase C
version — the renderer is ``r0.render_board`` and the parser is ``r0.parse_action``
(spec section 5 pins the parser for every round and every harness).

Model weights are never modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .r0 import system_message


@dataclass
class GridgameSpec:
    """The evolvable gridgame harness artifact."""

    system_prompt: str
    guidance: str = ""  # optional hint appended to the system message each turn
    history_window: int = 3  # recent moves shown to the model (r0._HISTORY_WINDOW)

    def to_dict(self) -> dict:
        return asdict(self)


# ── baseline system prompts (R0's exact templates) ───────────────────────────
#
# R0's ``system_message`` calls ``str.format`` on the template, so passing the
# literal strings ``"{size}"`` / ``"{budget}"`` as the format *arguments* yields
# the template back with its ``{size}`` / ``{budget}`` placeholders untouched.
# That keeps the baseline byte-identical to R0 while leaving the placeholders
# available for the rollout's ``str.replace`` (see ``gridgame_evolve.py``).

_FL_SYSTEM_TEMPLATE = system_message("frozenlake", "{budget}", "{size}")
_SK_SYSTEM_TEMPLATE = system_message("sokoban", "{budget}", "{size}")


def baseline_system_prompt(game: str) -> str:
    """Return the R0 baseline system-prompt template for ``game``.

    ``game`` is ``"frozenlake"`` or ``"sokoban"``; any other value falls back to
    the Sokoban template (same as R0's ``system_message``).
    """
    return _FL_SYSTEM_TEMPLATE if game == "frozenlake" else _SK_SYSTEM_TEMPLATE
