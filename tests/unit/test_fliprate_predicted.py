# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Smoke/unit tests for the task-flip-rate ``predicted_affected`` replacement.

Covers spec §3.1/§3.2 (journal back-fill uses ``fliprate_predicted`` = top-k of
the parent round's failures by historical flip rate, while the meta's original
``predicted_affected`` is kept verbatim) and §3.3 (gate reward weighting).

Zero rollout / zero training / zero API calls — only on-disk fixtures plus the
precomputed ``data/predicted_affected_ablation/task_fliprate.json`` table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harnessx.meta_harness.journal import (
    JournalEntrySpec,
    append_entry,
    compute_attribution,
    fill_gating,
    read_entries,
)

# Import the module under test. run.py inserts the project root onto sys.path
# and imports litellm at module scope, but it performs no rollout at import.
from recipe.agent_evolver import run as evolver_run

REPO = Path(__file__).resolve().parents[2]
FLIPRATE_JSON = REPO / "data" / "predicted_affected_ablation" / "task_fliprate.json"


# ─── §3.1: top-k selection by flip rate ────────────────────────────────────


def test_compute_fliprate_predicted_topk_and_stratum_mean_fallback() -> None:
    fliprate_map = {"t0": 0.9, "t1": 0.1, "t2": 0.5, "t3": 0.8, "t4": 0.3}
    failed = {"t0", "t1", "t2", "t3", "t9"}  # t9 has no history → stratum_mean

    # k=2 → the two highest flip rates (t0=0.9, t3=0.8).
    assert evolver_run.compute_fliprate_predicted(failed, fliprate_map, 0.4, 2) == ["t0", "t3"]

    # k clamps to the number of failed tasks (t4 is not in the failed set).
    assert evolver_run.compute_fliprate_predicted(failed, fliprate_map, 0.4, 99) == [
        "t0", "t3", "t2", "t9", "t1",
    ]

    # t9 falls back to stratum_mean=0.6 → beats t1 (0.1) and t4 (0.3).
    assert evolver_run.compute_fliprate_predicted({"t9", "t1", "t4"}, fliprate_map, 0.6, 1) == ["t9"]

    assert evolver_run.compute_fliprate_predicted(set(), fliprate_map, 0.4, 8) == []
    assert evolver_run.compute_fliprate_predicted({"t0"}, fliprate_map, 0.4, 0) == []


def test_model_size_extraction_and_stratum_resolution() -> None:
    assert evolver_run._extract_model_size_b("Qwen3.5-4B") == "4b"
    assert evolver_run._extract_model_size_b("Qwen3.5-2b") == "2b"
    assert evolver_run._extract_model_size_b("some-0.8B-ckpt") == "0.8b"
    assert evolver_run._extract_model_size_b("no-size-token") is None

    table = {
        "alfworld_4b": {"stratum_mean": 0.3, "0": {"flip_rate": 0.5}},
        "alfworld_2b": {"stratum_mean": 0.1, "0": {"flip_rate": 0.2}},
    }
    # Exact key.
    assert evolver_run._resolve_fliprate_stratum(table, "alfworld", "Qwen3.5-4B")[0] == "alfworld_4b"
    assert evolver_run._resolve_fliprate_stratum(table, "alfworld", "Qwen3.5-2B")[0] == "alfworld_2b"
    # Size token present but no exact key → benchmark-prefix fallback (first match).
    key, stratum = evolver_run._resolve_fliprate_stratum(table, "alfworld", "Qwen3.5-0.8B")
    assert key == "alfworld_4b" and stratum["stratum_mean"] == 0.3
    # Size token un-extractable → benchmark-prefix fallback.
    assert evolver_run._resolve_fliprate_stratum(table, "alfworld", "mystery")[0] == "alfworld_4b"
    # No benchmark match at all.
    assert evolver_run._resolve_fliprate_stratum(table, "textcraft", "Qwen3.5-4B") == (None, {})

    assert evolver_run._stratum_fliprate_map(table["alfworld_4b"]) == {"0": 0.5}


# ─── §3.1/§3.2: back-fill uses fliprate_predicted, keeps meta original ─────


def test_backfill_uses_fliprate_predicted_and_preserves_meta_original(tmp_path: Path) -> None:
    journal_path = tmp_path / "learnings.md"

    # Meta wrote predicted_affected as ints (ALFWorld convention).
    append_entry(
        journal_path,
        JournalEntrySpec(
            round=1,
            label="fliprate smoke",
            hypothesis_id="h_smoke",
            levers=["instruction"],
            predicted_affected=[14, 24, 52],
            prose="### Why\nsmoke\n",
        ),
    )

    entry = read_entries(journal_path)[0]
    meta_predicted = entry.predicted_affected  # str-normalized ["14", "24", "52"]
    assert meta_predicted == ["14", "24", "52"]

    k = len(meta_predicted) if meta_predicted else 8
    assert k == 3

    # Handwritten two-round per-task pass sets (parent round = round 0).
    prev_passed = {"24"}  # 24 was already passing before → out of the flip pool
    prev_failed = {"14", "24", "52", "17"}  # src 失败 (parent failures)
    cur_passed = {"14"}  # 14 flipped this round
    fliprate_map = {"14": 0.9, "24": 0.1, "52": 0.4, "17": 0.7}
    stratum_mean = 0.3

    # fliprate_predicted = top-k of parent failures by flip rate.
    predicted = evolver_run.compute_fliprate_predicted(prev_failed, fliprate_map, stratum_mean, k)
    # 14 (0.9), 17 (0.7), 52 (0.4) — 24 (0.1) is dropped.
    assert predicted == ["14", "17", "52"]

    attribution = compute_attribution(
        predicted,
        passed_now=cur_passed,
        passed_before=prev_passed,
        appeared_now={"14", "17", "52", "24"},
        appeared_before={"14", "17", "52", "24"},
    )
    # 14 flipped, 17/52 stayed failing (regression of 24 is outside the prediction).
    assert attribution == {"14": "flipped", "17": "still_F", "52": "still_F"}

    # regressed_unpredicted = tasks that regressed but weren't predicted.
    predicted_set = set(predicted)
    regressed_unpredicted = sorted(
        (prev_passed & {"14", "17", "52", "24"} & {"14", "17", "52", "24"})
        - cur_passed - predicted_set
    )
    assert regressed_unpredicted == ["24"]

    extra_fm = {
        "regressed_unpredicted": regressed_unpredicted,
        "fliprate_predicted": predicted,
    }
    if "predicted_affected" in entry.frontmatter:
        extra_fm["predicted_affected"] = entry.frontmatter["predicted_affected"]

    ok = fill_gating(journal_path, 1, "accepted", attribution, extra_frontmatter=extra_fm)
    assert ok is True

    reread = read_entries(journal_path)[0]
    assert reread.gating_attribution == attribution
    assert reread.frontmatter["fliprate_predicted"] == ["14", "17", "52"]
    # Meta's original predicted_affected preserved verbatim (ints, not strings).
    assert reread.frontmatter["predicted_affected"] == [14, 24, 52]
    assert reread.frontmatter["regressed_unpredicted"] == ["24"]


# ─── §3.3: gate reward weighting ───────────────────────────────────────────


def test_weighted_fliprate_reward() -> None:
    records = [
        {"task_id": "a", "reward": 1.0},
        {"task_id": "b", "reward": 0.0},
        {"task_id": "c", "reward": 0.5},
    ]
    fliprate_map = {"a": 0.0, "b": 1.0, "c": 0.5}
    # weights: a=1.0, b=0.0, c=0.5 → (1.0*1.0 + 0.0*0.0 + 0.5*0.5) / 1.5 = 1.25/1.5
    got = evolver_run._weighted_fliprate_reward(records, fliprate_map, 0.3)
    assert got == pytest.approx((1.0 + 0.25) / 1.5)

    # stratum_mean fallback for unknown tasks: all three get 0.3 → w=0.7 each.
    got2 = evolver_run._weighted_fliprate_reward(records, {}, 0.3)
    assert got2 == pytest.approx(sum(r["reward"] for r in records) / len(records))

    # All-zero weights fall back to plain mean.
    all_one = [{"task_id": "a", "reward": 1.0}, {"task_id": "b", "reward": 0.0}]
    got3 = evolver_run._weighted_fliprate_reward(all_one, {"a": 1.0, "b": 1.0}, 0.3)
    assert got3 == pytest.approx(0.5)

    assert evolver_run._weighted_fliprate_reward([], {}, 0.3) == 0.0


# ─── Real-data wiring (skipped when the on-disk artefacts are absent) ──────


@pytest.mark.skipif(not FLIPRATE_JSON.is_file(), reason="fliprate table not present")
def test_real_fliprate_table_wiring() -> None:
    table = evolver_run._load_task_fliprate_table(FLIPRATE_JSON)
    assert set(table) == {"alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"}

    key, stratum = evolver_run._resolve_fliprate_stratum(table, "alfworld", "Qwen3.5-4B")
    assert key == "alfworld_4b"
    fliprate_map = evolver_run._stratum_fliprate_map(stratum)
    assert len(fliprate_map) == 64
    assert 0.0 <= min(fliprate_map.values()) <= max(fliprate_map.values()) <= 1.0

    # top-k over a handwritten failed set ranks by the real flip rates.
    failed = {"0", "1", "2", "3", "4", "5"}
    top3 = evolver_run.compute_fliprate_predicted(failed, fliprate_map, stratum["stratum_mean"], 3)
    assert len(top3) == 3
    rates = [fliprate_map[t] for t in top3]
    assert rates == sorted(rates, reverse=True)
