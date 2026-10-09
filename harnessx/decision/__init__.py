# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""``harnessx.decision`` — a backend-agnostic interface to System-1 discrete-decision models.

A "jevlike" model (kev, semif, anyjev, …) answers closed-set decisions about a
state in a single forward pass — no generation, only a logit readout over the
candidates.  This package is the general contract both call sites share:

- the **meta-evolve pre-analyzer** (failure-mode bucketing, capability-gap
  triage, lever priors), and
- the **runtime processors** (model routing, judging, sycophancy, loop
  detection).

Only two question shapes exist — **判断** (yes/no) and **选项** (argmax over a
closed candidate set).  Continuous scoring is deliberately absent: the decision
taxonomy is discrete by design.  Each backend adaptor translates that canonical
vocabulary into its native API (kev's ``noul``/``choice``, semif's first-token
logit read, …), so swapping a backend never touches the callers.
"""

from .types import ChooseQ, Decision, DiscreteDecisionModel, JudgeQ, Question
from .kev import KevBackend
from .llm import LlmDecisionBackend
from .semif import SemifBackend
from .registry import get_decision_model, register

__all__ = [
    "ChooseQ",
    "Decision",
    "DiscreteDecisionModel",
    "JudgeQ",
    "KevBackend",
    "LlmDecisionBackend",
    "SemifBackend",
    "Question",
    "get_decision_model",
    "register",
]
