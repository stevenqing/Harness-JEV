# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Backend registry: name → :class:`DiscreteDecisionModel` factory.

Mirrors the providers pattern elsewhere in harnessx: a backend is registered by
name and instantiated through one entry point.  ``kev`` and ``meta`` ship
registered; every caller uses :func:`get_decision_model` unchanged.
"""

from __future__ import annotations

from typing import Any

from .kev import KevBackend
from .llm import LlmDecisionBackend
from .types import DiscreteDecisionModel

_REGISTRY: dict[str, type] = {
    "kev": KevBackend,
    "meta": LlmDecisionBackend,
}


def get_decision_model(name: str, **kwargs: Any) -> DiscreteDecisionModel:
    """Instantiate a backend by name.  Unknown names raise with the available set."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown decision model {name!r}; registered: {sorted(_REGISTRY)}") from None
    return cls(**kwargs)
