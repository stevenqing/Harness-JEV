# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Part A — frozen kev lens drift check across evolution rounds (zero rollout).

SPEC ``kev co-evolution`` part A.  For every existing ALFWorld-4B evolution run,
for each round's failure trajectories under that round's config, score the frozen
kev (Mode A find/acquire noul questions) and compute the oracle ground truth
(find/acquire/other from ``traj_data.json`` ``plan.high_pddl``, the a1a2 method).
Report per-round accuracy / Cohen's kappa / kev & ground-truth distributions with
run-cluster bootstrap CIs, the R0-R1 vs R4-R5 accuracy difference (drift test),
and per-class recall over rounds.  Deliverable: ``data/kev_coevo/A_drift/report.md``
plus the per-trajectory predictions and labels.

Zero rollout: reads existing ``runs/evolve/*/R*/trajectories/*.md`` and the frozen
kev sidecar only.

G0 (``--g0``): reproduce the a1a2 oracle labels for the reference R0 dirs
(``alfheldin-{0.8b,2b}-v1/R0``) and assert they match ``/tmp/a1a2_json/*_R0.json``
label-by-label.  This validates the tid→task mapping (``gamefile_order``) and the
``alfworld_label`` wiring, independent of the 4B analysis.

    .venv/bin/python -m recipe.agent_evolver.a_drift --g0
    .venv/bin/python -m recipe.agent_evolver.a_drift
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from harnessx.meta_harness.decision_prior import analyze_trajectories

from .decision_prior import _parse_trajectory, _render_state
from .failure_labels import alfworld_label, gamefile_order, load_alf_task
from .reviewer import ALF_MODES, ALF_QUESTIONS, accuracy, cohens_kappa, tag_from_scores

_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "evolve"
_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "kev_coevo" / "A_drift"

_ALF_HELDIN_ENV = "http://127.0.0.1:18082"
_KEV_URL = "http://127.0.0.1:8090"

# SPEC A.2: run-cluster bootstrap, 1000 draws.
_BOOTSTRAP_DRAWS = 1000
_BOOTSTRAP_SEED = 20261012

# SPEC G0 reference: a1a2 per-run R0 tables (stored labels) → the dirs they label.
_G0_REFERENCES = [
    ("alf_0.8b-v1_R0.json", "alfheldin-0.8b-v1"),
    ("alf_2b-v1_R0.json", "alfheldin-2b-v1"),
]
_G0_JSON_DIR = Path("/tmp/a1a2_json")


# ── run discovery ─────────────────────────────────────────────────────────────

def _comparison(path: Path) -> dict | None:
    cj = path / "comparison.json"
    if not cj.is_file():
        return None
    try:
        return json.loads(cj.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def discover_runs(min_rounds: int = 6) -> list[tuple[str, list[int]]]:
    """Return ``(run_name, [round_idx, …])`` for ALFWorld-4B heldin evolution runs.

    Only multi-round heldin 4B runs (the evolution loop); heldout one-shot probes,
    smoke tests, 2B/0.8B, WebShop/TextCraft/ScienceWorld and gridgames are out.
    """
    out: list[tuple[str, list[int]]] = []
    for d in sorted(_RUNS_DIR.glob("*/")):
        c = _comparison(d)
        if not c:
            continue
        if c.get("benchmark") != "alfworld":
            continue
        if c.get("agent_model") != "Qwen3.5-4B":
            continue
        if c.get("split") != "heldin":
            continue
        rounds = sorted(
            int(p.name[1:])
            for p in d.glob("R*")
            if p.is_dir() and p.name[1:].isdigit() and (p / "trajectories").is_dir()
        )
        if len(rounds) >= min_rounds:
            out.append((d.name, rounds))
    return out


def distinct_config_hashes(run_name: str) -> int:
    """Number of byte-distinct ``config.yaml`` across a run's rounds.

    A run with a single distinct config never evolved (meta produced no change /
    the gate reverted every round), so its R1..R5 trajectories are just R0 repeats
    and it cannot contribute to a *drift* measurement — it has no distribution
    shift to drift on.
    """
    hashes: set[str] = set()
    for p in (_RUNS_DIR / run_name).glob("R*/config.yaml"):
        try:
            hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
        except OSError:
            pass
    return len(hashes)


# ── oracle ground truth (a1a2 method, cached) ────────────────────────────────

def build_tasks(env_url: str, seed: int, num_tasks: int = 64) -> dict[str, dict]:
    """``traj_id (game_idx) → traj_data.json`` dict, via the seed-ordered gamefiles."""
    order = gamefile_order(env_url, seed)
    return {str(i): load_alf_task(order[i]) for i in range(min(num_tasks, len(order)))}


def _labels_for(fields_by_id: dict[str, dict], tasks: dict[str, dict]) -> dict[str, str]:
    return {
        tid: alfworld_label([a for a, _ in f["pairs"]], tasks[tid])
        for tid, f in fields_by_id.items()
    }


# ── one round: labels + kev scores ───────────────────────────────────────────

async def _score_round(traj_dir: Path, kev_url: str) -> dict[str, dict[str, float]]:
    """kev noul scores per traj_id: ``{mode: p(true)}`` for find/acquire."""
    rows, _elapsed = await analyze_trajectories(
        trajectories_dir=traj_dir,
        questions=ALF_QUESTIONS,
        parse_trajectory=_parse_trajectory,
        render_state=_render_state,
        model="kev",
        model_kwargs={"base_url": kev_url},
        max_trajectories=100_000,
    )
    return {
        r.traj_id: {m: r.decisions[i].probabilities.get("true", 0.0) for i, m in enumerate(ALF_MODES)}
        for r in rows
    }


def _parse_failures(traj_dir: Path) -> dict[str, dict]:
    files = sorted(
        traj_dir.glob("*.md"),
        key=lambda p: (int(p.stem) if p.stem.isdigit() else 1e9, p.stem),
    )
    out: dict[str, dict] = {}
    for p in files:
        f = _parse_trajectory(p)
        if f is not None:
            out[p.stem] = f
    return out


# ── bootstrap ────────────────────────────────────────────────────────────────

def cluster_bootstrap(
    stat_fn: Callable[[list[str]], float],
    run_names: list[str],
    *,
    draws: int = _BOOTSTRAP_DRAWS,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Run-cluster bootstrap: resample runs (with replacement), recompute
    ``stat_fn(sample)`` → ``(lo, hi, n_finite)``.  Non-finite draws (e.g. a rare
    class with zero gt instances in a resample) are dropped from the percentiles.
    """
    rng = random.Random(seed)
    n = len(run_names)
    est = []
    for _ in range(draws):
        sample = [rng.choice(run_names) for _ in range(n)]
        v = stat_fn(sample)
        if math.isfinite(v):
            est.append(v)
    est.sort()
    if not est:
        return float("nan"), float("nan"), 0
    lo = est[int(0.025 * len(est))]
    hi = est[int(0.975 * len(est)) - 1]
    return lo, hi, len(est)


# ── G0: label reproduction vs A1A2 stored tables ─────────────────────────────

def g0_check() -> int:
    """Reproduce a1a2 oracle labels on the reference R0 dirs; compare label-by-label."""
    print("# G0 — ground-truth label reproduction vs A1A2 stored tables\n")
    tasks = build_tasks(_ALF_HELDIN_ENV, 1234)
    all_ok = True
    results = {}
    for json_name, run_name in _G0_REFERENCES:
        ref_path = _G0_JSON_DIR / json_name
        if not ref_path.is_file():
            print(f"FAIL: reference table missing {ref_path}")
            all_ok = False
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_labels = {r["traj_id"]: r["label"] for r in ref["per_trajectory"]}

        traj_dir = _RUNS_DIR / run_name / "R0" / "trajectories"
        fields = _parse_failures(traj_dir)
        got = _labels_for(fields, tasks)

        common = sorted(set(ref_labels) & set(got))
        only_ref = sorted(set(ref_labels) - set(got))
        only_got = sorted(set(got) - set(ref_labels))
        mismatch = [(t, ref_labels[t], got[t]) for t in common if ref_labels[t] != got[t]]

        ok = not only_ref and not only_got and not mismatch
        all_ok &= ok
        results[run_name] = {
            "n_ref": len(ref_labels),
            "n_got": len(got),
            "n_common": len(common),
            "only_ref": only_ref,
            "only_got": only_got,
            "n_mismatch": len(mismatch),
            "mismatch": mismatch[:20],
        }
        status = "PASS" if ok else "FAIL"
        print(f"{run_name}: {status}  ref={len(ref_labels)} got={len(got)} "
              f"common={len(common)} mismatch={len(mismatch)}")
        for t, a, b in mismatch[:10]:
            print(f"   traj {t}: ref={a} got={b}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "g0_check.json").write_text(
        json.dumps({"passed": all_ok, "runs": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nG0 {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


# ── main drift analysis ──────────────────────────────────────────────────────

def _recall(records: list[dict], cls: str) -> float:
    denom = [r for r in records if r["label"] == cls]
    if not denom:
        return float("nan")
    return sum(r["kev_tag"] == cls for r in denom) / len(denom)


def _dist(records: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(r[key] for r in records))


async def _collect(
    run_names: list[tuple[str, list[int]]], tasks: dict[str, dict], kev_url: str
) -> list[dict]:
    """Score every round of every run; returns per-trajectory records (cached)."""
    cache_path = _OUT_DIR / "kev_scores_cache.json"
    cache: dict[str, dict[str, float]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    records: list[dict] = []
    for run_name, rounds in run_names:
        run_dir = _RUNS_DIR / run_name
        for rnd in rounds:
            traj_dir = run_dir / f"R{rnd}" / "trajectories"
            fields = _parse_failures(traj_dir)
            labels = _labels_for(fields, tasks)

            key_prefix = f"{run_name}|{rnd}|"
            if any(key_prefix + tid not in cache for tid in labels):
                scores = await _score_round(traj_dir, kev_url)
                for tid, s in scores.items():
                    cache[key_prefix + tid] = s
                cache_path.write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8"
                )

            for tid, label in labels.items():
                s = cache.get(key_prefix + tid)
                if s is None:
                    continue
                records.append(
                    {
                        "run": run_name,
                        "round": rnd,
                        "traj_id": tid,
                        "label": label,
                        "kev_tag": tag_from_scores(s, ALF_MODES),
                        "kev_scores": {m: round(s.get(m, 0.0), 4) for m in ALF_MODES},
                    }
                )
            print(f"  scored {run_name} R{rnd}: {len(labels)} failures", flush=True)
    return records


def _fmt(mean: float, lo: float, hi: float) -> str:
    if not math.isfinite(mean):
        return "—"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", action="store_true", help="run the G0 label-reproduction check only")
    ap.add_argument("--min-rounds", type=int, default=6)
    ap.add_argument("--include-stalled", action="store_true",
                    help="include runs whose config never changed across rounds (default: exclude)")
    ap.add_argument("--env-url", default=_ALF_HELDIN_ENV)
    ap.add_argument("--kev-url", default=_KEV_URL)
    args = ap.parse_args()

    if args.g0:
        return g0_check()

    run_names = discover_runs(args.min_rounds)
    if not run_names:
        print(f"no ALFWorld-4B heldin runs with >= {args.min_rounds} rounds under {_RUNS_DIR}")
        return 1

    stalled = [(n, distinct_config_hashes(n)) for n, _ in run_names if distinct_config_hashes(n) == 1]
    if not args.include_stalled and stalled:
        run_names = [(n, r) for n, r in run_names if distinct_config_hashes(n) >= 2]
        print(f"# excluded {len(stalled)} stalled run(s) (single config hash, no evolution):")
        for n, h in stalled:
            print(f"   {n} (distinct configs={h})")

    print(f"# Part A — ALFWorld-4B drift check over {len(run_names)} runs")
    for name, rounds in run_names:
        print(f"  {name}: rounds {rounds[0]}..{rounds[-1]} (configs={distinct_config_hashes(name)})")

    tasks = build_tasks(args.env_url, 1234)
    records = asyncio.run(_collect(run_names, tasks, args.kev_url))
    print(f"\ncollected {len(records)} failed-trajectory records")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUT_DIR / "per_trajectory.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_round: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_round[r["round"]].append(r)
    rounds = sorted(by_round)
    run_list = sorted({r["run"] for r in records})
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_run[r["run"]].append(r)

    def _pool(rounds_sel: set[int], sample_runs: list[str] | None = None) -> list[dict]:
        """Records in ``rounds_sel``.  With ``sample_runs``, respect bootstrap
        multiplicity (each repeated run contributes its trajectories again)."""
        if sample_runs is None:
            return [x for x in records if x["round"] in rounds_sel]
        out: list[dict] = []
        for rn in sample_runs:
            for x in by_run[rn]:
                if x["round"] in rounds_sel:
                    out.append(x)
        return out

    def _labels_tags(pool: list[dict]) -> tuple[list[str], list[str]]:
        return [r["label"] for r in pool], [r["kev_tag"] for r in pool]

    def _acc(pool: list[dict]) -> float:
        return accuracy(*_labels_tags(pool))

    def _kappa(pool: list[dict]) -> float:
        return cohens_kappa(*_labels_tags(pool))

    # per-round accuracy + kappa with cluster-bootstrap CI
    lines = ["# A_drift — frozen kev lens drift (ALFWorld 4B, zero rollout)", ""]
    lines.append(f"runs: {len(run_list)}  rounds: {rounds[0]}..{rounds[-1]}  "
                 f"failed trajectories: {len(records)}")
    lines.append(f"cluster bootstrap: run-level, {_BOOTSTRAP_DRAWS} draws, seed={_BOOTSTRAP_SEED}")
    lines.append("")
    lines.append("## Per-round accuracy / kappa / distributions")
    lines.append("")
    lines.append("| round | n_failed | accuracy | Cohen's κ | gt find/acquire/other | kev find/acquire/other |")
    lines.append("|---|---|---|---|---|---|")

    round_stats: dict[int, dict] = {}
    for rnd in rounds:
        pool = by_round[rnd]
        acc = _acc(pool)
        kappa = _kappa(pool)
        acc_lo, acc_hi, _ = cluster_bootstrap(
            lambda s, rnd=rnd: _acc(_pool({rnd}, s)), run_list
        )
        kappa_lo, kappa_hi, _ = cluster_bootstrap(
            lambda s, rnd=rnd: _kappa(_pool({rnd}, s)), run_list
        )
        gt = _dist(pool, "label")
        kv = _dist(pool, "kev_tag")
        round_stats[rnd] = {
            "n_failed": len(pool),
            "accuracy": acc,
            "accuracy_lo": acc_lo,
            "accuracy_hi": acc_hi,
            "kappa": kappa,
            "kappa_lo": kappa_lo,
            "kappa_hi": kappa_hi,
            "gt": gt,
            "kev": kv,
        }
        gt_s = f"{gt.get('find',0)}/{gt.get('acquire',0)}/{gt.get('other',0)}"
        kv_s = f"{kv.get('find',0)}/{kv.get('acquire',0)}/{kv.get('other',0)}"
        lines.append(
            f"| {rnd} | {len(pool)} | {_fmt(acc, acc_lo, acc_hi)} | "
            f"{_fmt(kappa, kappa_lo, kappa_hi)} | {gt_s} | {kv_s} |"
        )
    lines.append("")

    # drift test: acc(R4-R5) - acc(R0-R1)
    early_pool = _pool({0, 1})
    late_pool = _pool({4, 5})
    acc_early = _acc(early_pool)
    acc_late = _acc(late_pool)
    diff = acc_late - acc_early
    early_lo, early_hi, _ = cluster_bootstrap(lambda s: _acc(_pool({0, 1}, s)), run_list)
    late_lo, late_hi, _ = cluster_bootstrap(lambda s: _acc(_pool({4, 5}, s)), run_list)
    diff_lo, diff_hi, _ = cluster_bootstrap(
        lambda s: _acc(_pool({4, 5}, s)) - _acc(_pool({0, 1}, s)), run_list
    )
    drift = diff < 0 and diff_hi < 0  # CI entirely below 0 → late significantly lower

    lines.append("## Drift test (A.3)")
    lines.append("")
    lines.append(f"accuracy R0-R1: {_fmt(acc_early, early_lo, early_hi)}")
    lines.append(f"accuracy R4-R5: {_fmt(acc_late, late_lo, late_hi)}")
    lines.append(f"difference (R4-R5 − R0-R1): {_fmt(diff, diff_lo, diff_hi)}")
    lines.append("")
    if drift:
        lines.append("**Judgment: drift EXISTS** — R4-R5 accuracy is lower than R0-R1 and the CI excludes 0. B.3 lens training enters the plan.")
    elif diff < 0 and diff_hi >= 0:
        lines.append("**Judgment: no confirmed drift** — the difference is negative but its CI contains 0. Lens stays frozen.")
    else:
        lines.append("**Judgment: no drift** — R4-R5 accuracy is not lower than R0-R1. Lens stays frozen.")
    lines.append("")

    # per-class recall over rounds
    lines.append("## Per-class recall over rounds (label → kev recall)")
    lines.append("")
    classes = ["find", "acquire", "other"]
    lines.append("| class | " + " | ".join(f"R{rnd}" for rnd in rounds) + " |")
    lines.append("|---|" + "|".join(["---"] * len(rounds)) + "|")
    for cls in classes:
        row = []
        for rnd in rounds:
            rec = _recall(by_round[rnd], cls)
            lo, hi, _ = cluster_bootstrap(
                lambda s, rnd=rnd, cls=cls: _recall(_pool({rnd}, s), cls), run_list
            )
            row.append(_fmt(rec, lo, hi))
        lines.append(f"| {cls} | " + " | ".join(row) + " |")
    lines.append("")

    # find→acquire shift (the key question of A.2)
    lines.append("## Find→acquire shift")
    lines.append("")
    g0 = _dist(by_round[0], "label")
    gl = _dist(by_round[rounds[-1]], "label")
    lines.append(f"ground-truth label distribution R0: {g0}")
    lines.append(f"ground-truth label distribution R{rounds[-1]}: {gl}")
    a0 = _recall(by_round[0], "acquire")
    al = _recall(by_round[rounds[-1]], "acquire")
    a0_lo, a0_hi, _ = cluster_bootstrap(lambda s: _recall(_pool({0}, s), "acquire"), run_list)
    al_lo, al_hi, _ = cluster_bootstrap(
        lambda s: _recall(_pool({rounds[-1]}, s), "acquire"), run_list
    )
    lines.append(f"acquire recall R0: {_fmt(a0, a0_lo, a0_hi)}")
    lines.append(f"acquire recall R{rounds[-1]}: {_fmt(al, al_lo, al_hi)}")
    lines.append("")

    # scope + notes
    lines.append("## Scope / notes")
    lines.append("")
    lines.append("- Only ALFWorld-4B **heldin** evolution runs (R=6). Excluded: heldout one-shot")
    lines.append("  probes (single round, no drift), smoke tests, 2B/0.8B, WebShop/TextCraft/ScienceWorld,")
    lines.append("  the 5 R=3 heldin runs (flipab_meta/rate, heldin-evolve-v1/v2, quality_llm — no R4/R5),")
    lines.append("  and stalled runs whose config never changed across rounds (6 tj-* runs, flat pass_rate")
    lines.append("  0.328 — their R1..R5 are R0 repeats, so they have no distribution shift to drift on;")
    lines.append("  re-run with `--include-stalled` to add them back).")
    lines.append("- WebShop: not performed (ALFWorld-only).  Old WebShop trajectories carry ``goal`` +")
    lines.append("  ``final_obs`` but no ``reward_info`` (the env-server reward decomposition only exists for")
    lines.append("  the new step0c loop), so §4.2 labels would need a manual goal/product-page parser — a")
    lines.append("  separate follow-up, not part of the ALFWorld drift question.")
    lines.append("- kev scored once per trajectory with the currently deployed frozen sidecar (deterministic).")
    lines.append("")

    (_OUT_DIR / "round_stats.json").write_text(
        json.dumps({str(k): v for k, v in round_stats.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (_OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n# Per-round (point estimates)")
    for rnd in rounds:
        s = round_stats[rnd]
        print(f"  R{rnd}: n={s['n_failed']} acc={s['accuracy']:.3f} kappa={s['kappa']:.3f} "
              f"gt={s['gt']} kev={s['kev']}")
    print(f"\ndrift: acc(R4-R5)={acc_late:.3f} acc(R0-R1)={acc_early:.3f} "
          f"diff={diff:.3f} [{diff_lo:.3f}, {diff_hi:.3f}] drift={drift}")
    print(f"wrote {_OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
