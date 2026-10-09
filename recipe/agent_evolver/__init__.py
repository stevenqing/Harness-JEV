# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""agent_evolver: harness-evolution loop for qwen3.5-4b on ALFWorld/WebShop.

Only the harness (system prompt + guidance + loop params) evolves; model weights
are never modified or trained.

Entry point is ``recipe.agent_evolver.run`` (``--benchmark {alfworld,webshop}``),
a text-protocol evolver: the agent is the Skill1 ``<think>/<action>`` ReAct loop
(``agents.run_alfworld`` / ``agents.run_webshop``), and each round
``MetaAgent.evolve()`` edits a ``HarnessConfig`` that *carries* the text-loop
spec. See ``native/`` for the config encoding and
``skills/{alfworld,webshop}-playbook/`` for the meta-agent's benchmark guidance.
"""
