# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Defaults and the evolvable HarnessSpec for agent_evolver.

The "harness" being evolved is the system prompt + a per-turn guidance string.
Model weights are never touched.

Generation / env defaults below follow the visibility-spec S1 decoding
(thinking on, max_tokens 4096, one temperature) so ALFWorld/WebShop share the
same decoding as gridgames.  Flip ``EVOLVER_AGENT_ENABLE_THINKING`` / raise
``EVOLVER_AGENT_MAX_TOKENS`` only for a deliberate decoding ablation; the S1
report always records the exact values used.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# Agent model = the weak model we're making competent (never trained).
# vLLM serves this model under the bare id "Qwen3.5-4B" (no provider prefix).
AGENT_MODEL = os.environ.get("EVOLVER_AGENT_MODEL", "Qwen3.5-4B")
AGENT_API_BASE = os.environ.get("EVOLVER_AGENT_API_BASE", "http://127.0.0.1:8200/v1")

# Skill1 / verl-agent base generation settings. temperature 0.4 (NOT 0.0),
# top_p 0.8 is the Qwen recommended value.
AGENT_TEMPERATURE = float(os.environ.get("EVOLVER_AGENT_TEMPERATURE", "0.4"))
AGENT_TOP_P = float(os.environ.get("EVOLVER_AGENT_TOP_P", "0.8"))
AGENT_MAX_TOKENS = int(os.environ.get("EVOLVER_AGENT_MAX_TOKENS", "4096"))
# S1: thinking ON by default (gridgames already runs this way).  The
# <think>/<action> tags in agents.py still parse correctly because
# extract_action only looks for <action>...</action>; native thinking is
# simply additional text ahead of it.
AGENT_ENABLE_THINKING = os.environ.get("EVOLVER_AGENT_ENABLE_THINKING", "1") == "1"

# Qwen3.5 default system prompt (matches the verl-agent / Skill1 rollout).
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Env / loop settings aligned to the reference configs.
HISTORY_LENGTH = int(os.environ.get("EVOLVER_HISTORY_LENGTH", "2"))  # last N (obs, action) pairs
ALF_MAX_STEPS = int(os.environ.get("EVOLVER_ALF_MAX_STEPS", "50"))
WS_MAX_STEPS = int(os.environ.get("EVOLVER_WS_MAX_STEPS", "15"))


@dataclass
class HarnessSpec:
    """The evolvable harness artifact.

    ``processors`` is a sparse dict of loop processors the harness can enable
    and tune (see processors.py). Keys are processor names ("loop_breaker",
    "history_format", "tried_action_mask"); values are parameter dicts. Missing
    keys mean "use the default / disabled".
    """

    system_prompt: str
    guidance: str = ""  # optional hint injected into the prompt each turn
    processors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
