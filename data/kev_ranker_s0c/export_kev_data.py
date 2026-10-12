# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Export the layered-candidates loop journals to kev-ranker training data.

Spec: ``specs/kev_ranker_step0c.md`` §7.  Reads every ``rounds.jsonl`` under the
given run dirs, joins each (candidate × task) cell against the parent's
evaluation and the per-round ``failure_types.json``, and writes
``kev_data/tuples.parquet`` plus ``kev_data/premises.md`` (the §5 repair table +
§6 candidate-diff recomputed from the exported cells).

Usage::

    python export_kev_data.py --run-dirs runs/evolve/<tag> [--run-dirs ...] \
        --out kev_data/tuples.parquet --premises kev_data/premises.md

The script is pure data plumbing — it never calls the agent, kev, or the
meta-LLM.  ``kev_answer`` is recovered from each round's ``failure_types.json``
(which stores the full per-question decision distribution, not just the argmax
tag), so no re-scoring happens here.  Only dependency is ``pyarrow`` (for
parquet), imported lazily.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Columns of tuples.parquet, in order (§7 table).
COLUMNS = [
    "env", "arm", "seed", "round",
    "parent_hash", "candidate_hash",
    "layer", "diff",
    "task_id",
    "parent_traj", "parent_pass_mean", "cand_pass_mean",
    "failure_type", "kev_answer",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _diff_dicts(parent: dict, child: dict, prefix: str = "") -> dict[str, list]:
    """Field-level structured diff (no rationale — both sides are config dicts).

    Returns ``{field_path: [parent_value, child_value]}`` for every leaf that
    changed.  Nested dicts recurse; lists of dicts recurse by index; other
    lists and leaf scalars compare by equality.
    """
    out: dict[str, list] = {}
    keys = set(parent) | set(child)
    for k in sorted(keys):
        path = f"{prefix}.{k}" if prefix else k
        pv = parent.get(k, "<absent>")
        cv = child.get(k, "<absent>")
        if isinstance(pv, dict) and isinstance(cv, dict):
            out.update(_diff_dicts(pv, cv, path))
        elif (
            isinstance(pv, list)
            and isinstance(cv, list)
            and pv
            and cv
            and all(isinstance(x, dict) for x in pv)
            and all(isinstance(x, dict) for x in cv)
        ):
            # list of config blocks (e.g. ``processors[]``) — diff element-wise.
            for i in range(max(len(pv), len(cv))):
                ep = pv[i] if i < len(pv) else "<absent>"
                ec = cv[i] if i < len(cv) else "<absent>"
                if isinstance(ep, dict) and isinstance(ec, dict):
                    out.update(_diff_dicts(ep, ec, f"{path}.{i}"))
                elif ep != ec:
                    out[f"{path}.{i}"] = [ep, ec]
        elif pv != cv:
            out[path] = [pv, cv]
    return out


def _task_means(repeats: list[list[dict]], task_ids: list[int]) -> dict[int, dict]:
    """``{task_id: {pass_mean, reward_mean, traj_path}}`` from a candidate's repeats.

    ``repeats`` = one list per rep; each rep = one dict per task (``_records_summary``).
    ``traj_path`` is taken from the first rep that has one (rep0 — §7).
    """
    by_task: dict[int, dict] = {}
    if not repeats:
        return by_task
    for tid in task_ids:
        passes: list[float] = []
        rewards: list[float] = []
        traj = None
        for rep in repeats:
            row = next((r for r in rep if r.get("task_id") == tid), None)
            if row is None:
                continue
            passes.append(1.0 if row.get("pass") else 0.0)
            rewards.append(float(row.get("reward") or 0.0))
            if traj is None:
                traj = row.get("traj_path")
        if not passes:
            continue
        by_task[tid] = {
            "pass_mean": sum(passes) / len(passes),
            "reward_mean": sum(rewards) / len(rewards),
            "traj_path": traj,
        }
    return by_task


def _collect_task_ids(rounds: list[dict]) -> list[int]:
    ids: set[int] = set()
    for rec in rounds:
        for rep in rec.get("repeats") or []:
            ids.update(r["task_id"] for r in rep)
        for cand in rec.get("candidates") or []:
            for rep in cand.get("repeats") or []:
                ids.update(r["task_id"] for r in rep)
        for rep in rec.get("heldout") or []:
            ids.update(r["task_id"] for r in rep)
    return sorted(ids)


def _load_failure_types(run_dir: Path, round_idx: int) -> dict:
    path = run_dir / f"R{round_idx}" / "failure_types.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ── main export ──────────────────────────────────────────────────────────────

def export_run(
    run_dir: Path,
    *,
    env: str,
    arm: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Emit one row per (candidate × task) for a single run's layered journal."""
    rounds = _read_jsonl(run_dir / "rounds.jsonl")
    if not rounds:
        return []

    # Parent lineage: hash -> the parent's per-task evaluation.  R0's base config
    # is the parent for round 1; an accepted candidate becomes the parent for the
    # next round.
    parent_evals: dict[str, dict[int, dict]] = {}
    task_ids = _collect_task_ids(rounds)

    rows: list[dict[str, Any]] = []
    for rec in rounds:
        rnd = rec["round"]
        parent_hash = rec["parent_hash"]
        ft = _load_failure_types(run_dir, rnd)
        names = ft.get("names", [])
        per_traj = {t["traj_id"]: t for t in ft.get("per_trajectory", [])}

        if rnd == 0:
            # R0 is the base config: record its eval as the round-1 parent.
            parent_evals[parent_hash] = _task_means(rec.get("repeats") or [], task_ids)
            continue

        parent_tasks = parent_evals.get(parent_hash, {})
        parent_cfg = rec.get("parent_config")  # persisted in the journal (added for §7)
        for cand in rec.get("candidates") or []:
            cand_tasks = _task_means(cand.get("repeats") or [], task_ids)
            diff = _diff_dicts(parent_cfg or {}, cand.get("config") or {})
            for tid in task_ids:
                pt = parent_tasks.get(tid, {})
                ct = cand_tasks.get(tid, {})
                ft_row = per_traj.get(str(tid), {})
                rows.append({
                    "env": env,
                    "arm": arm,
                    "seed": seed,
                    "round": rnd,
                    "parent_hash": parent_hash,
                    "candidate_hash": cand.get("config_hash"),
                    "layer": cand.get("layer"),
                    "diff": json.dumps(diff, ensure_ascii=False, sort_keys=True),
                    "task_id": tid,
                    "parent_traj": pt.get("traj_path"),
                    "parent_pass_mean": pt.get("pass_mean"),
                    "cand_pass_mean": ct.get("pass_mean"),
                    "failure_type": ft_row.get("label"),
                    "kev_answer": json.dumps(
                        {"names": names, "decisions": ft_row.get("kev_decisions")},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                })

        # If a candidate was accepted, it is the parent for the next round.
        acc = rec.get("accepted")
        if acc is not None and 0 <= acc < len(rec.get("candidates") or []):
            cand = rec["candidates"][acc]
            parent_evals[cand.get("config_hash")] = _task_means(cand.get("repeats") or [], task_ids)

    return rows


# ── premises (§7 tail) ───────────────────────────────────────────────────────

def _write_premises(rows: list[dict], out_path: Path) -> None:
    """Recompute the step0b §5 repair table + §6 candidate diff from the cells."""
    out_path = Path(out_path)
    lines: list[str] = []
    lines.append("# kev-ranker premises (recomputed from tuples.parquet)\n")
    pairs = {(r["parent_hash"], r["candidate_hash"]) for r in rows}
    lines.append(f"cells={len(rows)}  pairs={len(pairs)}\n")

    # §6 candidate diff: per (env, arm, seed, round) candidate-group spread.
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["env"], r["arm"], r["seed"], r["round"])
        groups[key][r["candidate_hash"]].append(r["cand_pass_mean"])

    lines.append("## §6 candidate score diff (held-in mean)\n")
    lines.append("| env | arm | seed | round | candidates | max-min | random-vs-best loss |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for (env, arm, seed, rnd), by_hash in sorted(groups.items(), key=lambda kv: str(kv[0])):
        means = {h: sum(v) / len(v) for h, v in by_hash.items()}
        if len(means) < 2:
            continue
        vals = list(means.values())
        spread = max(vals) - min(vals)
        rand_loss = max(vals) - (sum(vals) / len(vals))
        lines.append(f"| {env} | {arm} | {seed} | {rnd} | {len(means)} | {spread:.4f} | {rand_loss:.4f} |\n")

    # §5 repair table: rows = failure_type, cols = layer, cells = flip rate (n).
    lines.append("\n## §5 repair table (failure_type × layer → flip rate)\n")
    flip_rows = [r for r in rows if r["failure_type"] is not None and r["layer"] is not None]
    if flip_rows:
        ft_layers = sorted({r["layer"] for r in flip_rows})
        ft_types = sorted({r["failure_type"] for r in flip_rows})
        lines.append("| failure_type | " + " | ".join(ft_layers) + " |\n")
        lines.append("|---|" + "---|" * len(ft_layers) + "\n")
        for ft in ft_types:
            cells = []
            for lay in ft_layers:
                sub = [r for r in flip_rows if r["failure_type"] == ft and r["layer"] == lay]
                if not sub:
                    cells.append("—")
                    continue
                flip = sum(r["cand_pass_mean"] - r["parent_pass_mean"] for r in sub) / len(sub)
                cells.append(f"{flip:+.3f} (n={len(sub)})")
            lines.append(f"| {ft} | " + " | ".join(cells) + " |\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_parquet(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyarrow is required to write tuples.parquet; run this in an env that "
            f"has it (e.g. ee-agentgym). ({exc})"
        ) from exc
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        pa.field("env", pa.string()),
        pa.field("arm", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("round", pa.int64()),
        pa.field("parent_hash", pa.string()),
        pa.field("candidate_hash", pa.string()),
        pa.field("layer", pa.string()),
        pa.field("diff", pa.string()),
        pa.field("task_id", pa.int64()),
        pa.field("parent_traj", pa.string()),
        pa.field("parent_pass_mean", pa.float64()),
        pa.field("cand_pass_mean", pa.float64()),
        pa.field("failure_type", pa.string()),
        pa.field("kev_answer", pa.string()),
    ]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def _parse_tag(tag: str) -> tuple[str, str, int]:
    """Infer (env, arm, seed) from a run tag like ``alfworld_kev_s1234_heldin``."""
    parts = tag.split("_")
    env = "alfworld" if "alf" in tag else ("webshop" if "web" in tag or "shop" in tag else "?")
    arm = "kev" if "kev" in parts else ("const" if "const" in parts else ("llm" if "llm" in parts else "?"))
    seed = 0
    for part in parts:
        if part.startswith("s") and part[1:].isdigit():
            seed = int(part[1:])
    return env, arm, seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True, help="run dirs under runs/evolve/")
    p.add_argument("--out", default="kev_data/tuples.parquet")
    p.add_argument("--premises", default="kev_data/premises.md")
    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        env, arm, seed = _parse_tag(run_dir.name)
        all_rows.extend(export_run(run_dir, env=env, arm=arm, seed=seed))

    _write_parquet(all_rows, Path(args.out))
    _write_premises(all_rows, Path(args.premises))
    print(f"exported {len(all_rows)} cells → {args.out}")
    print(f"premises → {args.premises}")


if __name__ == "__main__":
    main()
