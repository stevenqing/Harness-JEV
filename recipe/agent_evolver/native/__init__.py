# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Config encoding for the text-protocol evolver.

The agent is Qwen3.5-4B driven by the Skill1 ``<think>/<action>`` ReAct loop
(``recipe.agent_evolver.agents``), not native tool-calling.  The evolvable
artifact is a ``HarnessConfig`` that *carries* the text-loop spec (system
prompt + guidance + loop-processor params) so ``MetaAgent.evolve()`` can edit
it as YAML.  See ``config.py`` for the encoding and ``spec_from_config``.

Model weights are never modified.
"""
