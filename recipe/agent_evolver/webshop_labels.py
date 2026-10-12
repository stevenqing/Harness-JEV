# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Programmatic WebShop failure labels (§4.2 of ``specs/kev_ranker_step0c.md``).

A WebShop reward decomposes into four components, all computed by the env at
buy time (``web_agent_site.engine.goal.get_reward``):

- ``r_type``   (0/1) — purchased product's type matches the goal type.
- ``r_att``    (0..1) — fraction of goal attributes matched.
- ``r_option`` (0..1) — fraction of goal options matched (``None`` if no options).
- ``r_price``  (0/1) — price within the goal's upper bound (``None`` if no bound).

The env server (``benchmarks/webshop/env_server.py``) now returns that
decomposition as ``reward_info`` on the ``/step`` that terminates the episode
(buy).  The labels are decided in the order below, first match wins — exactly
the table in spec §4.2.  ``reward == 1`` is a success and gets no label.

Algorithm (also documented for the user in ``webshop_labels.md``):

    1. success (reward >= 0.999)                        -> None (no failure)
    2. no ``search[...]`` action in the trajectory      -> ``no_search``
    3. searched but no buy (no ``reward_info``)         -> ``not_bought``
    4. bought, ``r_type`` == 0                          -> ``wrong_type``
    5. bought, type ok, ``r_att`` < 1                   -> ``attr_mismatch``
    6. bought, type+attr ok, ``r_option`` < 1           -> ``option_mismatch``
    7. bought, first three ok, ``r_price`` == 0         -> ``price_over``
    8. (defensive) everything matched but reward < 1    -> ``other``
"""
from __future__ import annotations

from typing import Any

_FLOAT_EPS = 1e-6


def _is_full(value: Any) -> bool:
    """True when a 0..1 component counts as 'fully matched' (1.0, or absent)."""
    return value is None or (isinstance(value, (int, float)) and value >= 1.0 - _FLOAT_EPS)


def ws_label(record: dict) -> str | None:
    """Six-way programmatic label for one failed WebShop trajectory.

    ``record`` is a single ``run_webshop`` result dict (``task_id``, ``success``,
    ``reward``, ``reward_info``, ``trajectory``, …).  A successful episode
    (``reward >= 0.999``) is not a failure and returns ``None``.
    """
    if record.get("success") or float(record.get("reward") or 0.0) >= 0.999:
        return None

    actions = [
        str(t.get("action") or "") for t in (record.get("trajectory") or [])
    ]
    searched = any(a.startswith("search[") for a in actions)
    if not searched:
        return "no_search"

    info = record.get("reward_info") or {}
    if not info:
        return "not_bought"  # searched but never bought within the step budget

    if int(info.get("r_type") or 0) == 0:
        return "wrong_type"
    if not _is_full(info.get("r_att")):
        return "attr_mismatch"
    if not _is_full(info.get("r_option")):
        return "option_mismatch"
    if info.get("r_price") is not None and int(info.get("r_price") or 0) == 0:
        return "price_over"
    return "other"  # defensive: all components matched but reward < 1 (shouldn't happen)


def ws_labels_from_records(records: list[dict]) -> dict[int, str]:
    """``{task_id: label}`` for the failed tasks in one rollout's records."""
    out: dict[int, str] = {}
    for r in records:
        label = ws_label(r)
        if label is not None:
            out[r["task_id"]] = label
    return out
