# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Carrier processor for the text-loop evolvable spec.

The text protocol (Skill1 ``<think>/<action>`` ReAct loop, see
``agents.run_alfworld`` / ``agents.run_webshop``) is driven by our own rollout
code, **not** the native runloop.  The ``HarnessConfig`` therefore exists only
to *carry* the evolvable spec so ``MetaAgent.evolve()`` can read and edit it as
YAML; the native runloop never meaningfully executes this config (the replay
gate only needs a runnable, non-crashing pipeline).

The spec is split across two processors (see ``config.py``):

* ``SystemPromptProcessor`` + ``StaticSystemPromptBuilder`` → the system prompt
  (``prompt:`` field).
* ``TextEvolverSpecProcessor`` (this class) → the per-turn ``guidance`` string
  (the ``{skills}`` slot) and the flattened loop-processor params.

NOTE: this module is loaded by the processor loader via ``file://…::Class``
(synthetic module name), so it MUST only use absolute imports.
"""
from __future__ import annotations

from harnessx.core.processor import MultiHookProcessor


class TextEvolverSpecProcessor(MultiHookProcessor):
    """No-op native-runloop processor that carries the text-loop params.

    The flattened fields mirror ``HarnessSpec.processors`` (see
    ``recipe.agent_evolver.processors.PROCESSOR_DEFAULTS``):

    * ``loop_breaker_k``        ↔ ``loop_breaker.k``
    * ``history_mode`` / ``history_n`` / ``history_obs_clip``
                                ↔ ``history_format.{mode,n,obs_clip}``
    * ``tried_action_mask``     ↔ ``tried_action_mask.enabled``
    """

    def __init__(
        self,
        guidance: str = "",
        loop_breaker_k: int = 3,
        history_mode: str = "raw",
        history_n: int = 2,
        history_obs_clip: int = 150,
        tried_action_mask: bool = False,
    ) -> None:
        self.guidance = guidance
        self.loop_breaker_k = int(loop_breaker_k)
        self.history_mode = history_mode
        self.history_n = int(history_n)
        self.history_obs_clip = int(history_obs_clip)
        self.tried_action_mask = bool(tried_action_mask)
