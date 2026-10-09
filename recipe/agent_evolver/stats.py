# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Bootstrap statistics for the visibility report (spec S3).

Every reported number has 3 seeds and a 95% percentile bootstrap over the
sampling unit — a *level* for gridgames, a *task* for ALFWorld/WebShop — with
10,000 draws.  A difference between two arms is called real only when the
interval of the difference excludes 0 (spec S3 last sentence).

The functions here are pure math over per-unit scalars; the caller decides what
a unit is and how it is averaged (across seeds / episodes) before calling in.
"""
from __future__ import annotations

import random
from typing import Sequence

BOOTSTRAP_DRAWS = 10_000
# Distinct from the Phase-B baseline report seed (20261002) so the two reports
# do not silently share a resampling stream.
BOOTSTRAP_SEED = 20261007


def bootstrap_ci(
    per_unit: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` — 95% percentile bootstrap over units.

    ``per_unit`` is one scalar per sampling unit (already averaged over seeds
    and episodes within that unit).  Units are resampled with replacement,
    their mean recorded, and lo/hi are the 2.5 / 97.5 percentiles of the
    ``draws`` means.  Empty input returns NaNs.
    """
    units = list(per_unit)
    if not units:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(units)
    est = [sum(rng.choices(units, k=n)) / n for _ in range(draws)]
    est.sort()
    lo = est[int(0.025 * draws)]
    hi = est[int(0.975 * draws) - 1]
    return sum(units) / n, lo, hi


def bootstrap_diff(
    a_per_unit: Sequence[float],
    b_per_unit: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, bool]:
    """Return ``(mean_diff, lo, hi, real)`` for ``a − b``, paired over units.

    The two arms are aligned unit-by-unit (same levels/tasks), so indices are
    resampled, not values — the bootstrap of a *paired* difference.  ``real``
    is True iff the 95% interval excludes 0 (spec S3).
    """
    a = list(a_per_unit)
    b = list(b_per_unit)
    if len(a) != len(b):
        raise ValueError(f"paired arms must have equal units: {len(a)} vs {len(b)}")
    if not a:
        return float("nan"), float("nan"), float("nan"), False
    rng = random.Random(seed)
    n = len(a)
    idx = range(n)
    est = [sum(a[i] - b[i] for i in rng.choices(idx, k=n)) / n for _ in range(draws)]
    est.sort()
    lo = est[int(0.025 * draws)]
    hi = est[int(0.975 * draws) - 1]
    mean_diff = (sum(a) - sum(b)) / n
    real = not (lo <= 0.0 <= hi)
    return mean_diff, lo, hi, real


def per_unit_mean(
    episodes: Sequence[dict],
    *,
    unit_key: str = "level_id",
    value_key: str = "success",
) -> dict[str, float]:
    """Group ``episodes`` by ``unit_key`` and return ``unit -> mean(value_key)``.

    This is the single step that collapses 3 seeds × episodes into one scalar
    per sampling unit before ``bootstrap_ci`` / ``bootstrap_diff``.
    """
    acc: dict[str, list[float]] = {}
    for ep in episodes:
        acc.setdefault(ep[unit_key], []).append(float(ep[value_key]))
    return {u: sum(v) / len(v) for u, v in acc.items()}
