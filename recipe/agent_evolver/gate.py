# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Regression gate: accept/revert an evolved config against the historical best.

Copied verbatim from ``recipe/tau2_evolver/run.py:_score_and_gate`` (the shared
orchestrator contract).  ``best`` is ``(reward, cost, config, round_idx)`` or
``None``; ``cost_weight`` is unused by default (0.0) but kept for parity.
"""
from __future__ import annotations

from typing import Any


def score_and_gate(
    *,
    round_reward: float,
    round_cost: float,
    round_idx: int,
    round_config: Any,
    best: tuple[float, float, Any, int] | None,
    tolerance: float,
    cost_weight: float,
) -> tuple[str, str, tuple | None, Any | None]:
    """Gate evolved config against historical best.

    Returns ``(decision, reason, new_best, reverted_config_or_None)``.
    ``decision`` is ``"accept"`` | ``"reject"``.
    """
    adjusted = round_reward - cost_weight * max(round_cost - (best[1] if best else 0.0), 0.0)

    if best is None:
        new_best = (round_reward, round_cost, round_config, round_idx)
        return "accept", "first round — establishing baseline", new_best, None

    best_reward, best_cost, best_cfg, best_idx = best
    best_adjusted = best_reward - cost_weight * 0.0

    if adjusted >= best_adjusted - tolerance:
        new_best = (round_reward, round_cost, round_config, round_idx) if round_reward > best_reward else best
        return "accept", f"reward {round_reward:.4f} ≥ best {best_reward:.4f} − tol {tolerance}", new_best, None
    reason = f"reward {round_reward:.4f} < best {best_reward:.4f} − tol {tolerance} → revert to R{best_idx}"
    return "reject", reason, best, best_cfg
