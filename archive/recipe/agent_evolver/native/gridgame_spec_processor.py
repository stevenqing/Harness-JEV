# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Carrier processor for the gridgame evolvable spec.

The gridgame loop (R0, spec section 5) is driven by our own rollout code
(``recipe/agent_evolver/gridgame_evolve.py``), **not** the native runloop.  The
``HarnessConfig`` therefore exists only to *carry* the evolvable spec so
``MetaAgent.evolve()`` can read and edit it as YAML; the native runloop never
meaningfully executes this config (the replay gate only needs a runnable,
non-crashing pipeline).

All three evolvable fields live on this single no-op processor:

* ``system_prompt``  — game rules + symbol text (``{size}``/``{budget}`` kept).
* ``guidance``       — per-turn hint appended to the system message.
* ``history_window`` — recent moves shown (``r0._HISTORY_WINDOW``).

NOTE: this module is loaded by the processor loader via ``file://…::Class``
(synthetic module name), so it MUST only use absolute imports.
"""
from __future__ import annotations

from harnessx.core.processor import MultiHookProcessor


class GridgameSpecProcessor(MultiHookProcessor):
    """No-op native-runloop processor that carries the gridgame spec.

    The three flattened fields mirror ``recipe.agent_evolver.gridgame_spec.GridgameSpec``.
    """

    def __init__(
        self,
        system_prompt: str = "",
        guidance: str = "",
        history_window: int = 3,
    ) -> None:
        self.system_prompt = system_prompt
        self.guidance = guidance
        self.history_window = int(history_window)
