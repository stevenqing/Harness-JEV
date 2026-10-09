# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Gate G1 for the gridgames adapter (spec section 7 A3).

Plays the four stub models through the full HarnessX pipeline — i.e. through
``gridgame_agent.run_gridgame``, the same loop real models use — on the gate
split of all 8 cells, and checks the four parts of spec section 7 A3.

    python -m recipe.agent_evolver.g1_gridgames              # run all four parts
    python -m recipe.agent_evolver.g1_gridgames compare FILE
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .gridgame_agent import (
    StubProvider,
    _BUNDLE_DIR,
    gate_levels,
    load_levels,
    load_tiers,
    run_gridgame,
)

# The bundle's stubs (sys.path already carries the bundle dir after the import above).
import stubs  # noqa: E402

_CELL_TOLERANCE = {"L4": 0.03, "L8": 0.02, "L16": 0.01, "L32": 0.01}
_GOLDEN = _BUNDLE_DIR / "g1d_golden.jsonl"


def _cell_key(level):
    return (level["game"], level["tier"])


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


async def g1a(levels, tiers, ep_per_level=16, seed=1234):
    """NullStub over the gate split → per-cell stats (spec G1a)."""
    stub = stubs.NullStub(seed=seed)
    model = StubProvider(stub)
    cells = {}
    for lvl in gate_levels(levels):
        key = _cell_key(lvl)
        acc = cells.setdefault(key, {"success": [], "dead": [], "score": [], "parse_fail": 0, "moves": 0})
        for _ in range(ep_per_level):
            rep = (await run_gridgame(model, [lvl]))[0]["report"]
            acc["success"].append(rep["success"])
            acc["dead"].append(rep["dead"])
            acc["score"].append(rep["progress_end"])
            acc["parse_fail"] += rep["parse_fail"]
            acc["moves"] += rep["moves"]
    out = {}
    for key, acc in cells.items():
        out[key] = {
            "success": _mean(acc["success"]),
            "dead": _mean(acc["dead"]),
            "score": _mean(acc["score"]),
            "parse_fail": acc["parse_fail"],
            "moves": acc["moves"],
        }
    return out, stub.calls


def check_g1a(stats, tiers):
    floors = tiers["floors"]["gate"]
    ok = True
    msgs = []
    for key, s in stats.items():
        game, tier = key
        f = floors[game][tier]
        tol = _CELL_TOLERANCE[tier]
        checks = [
            ("success", abs(s["success"] - f["uniform"]) <= tol),
            ("dead", abs(s["dead"] - f["uniform_dead"]) <= 0.06),
            ("score", abs(s["score"] - f["uniform_progress"]) <= 0.04),
            ("parse_fail", s["parse_fail"] == 0),
        ]
        for name, passed in checks:
            if not passed:
                ok = False
                msgs.append(f"{game}/{tier} {name}={s[name]} vs floor "
                            f"{f.get('uniform' if name == 'success' else 'uniform_dead' if name == 'dead' else 'uniform_progress')}")
    return ok, msgs


async def g1b(levels):
    stub = stubs.SilentStub()
    model = StubProvider(stub)
    bad = []
    moves_total = 0
    for lvl in gate_levels(levels):
        rep = (await run_gridgame(model, [lvl]))[0]["report"]
        moves_total += rep["moves"]
        budget = lvl["budget"]
        if not (rep["moves"] == budget and rep["parse_fail"] == budget
                and rep["success"] == 0 and rep["progress_end"] == 0
                and rep["dead"] == 0 and rep["blocked_moves"] == 0):
            bad.append(lvl["id"])
    return (len(bad) == 0 and stub.calls == moves_total), bad, stub.calls, moves_total


async def g1c(levels):
    stub = stubs.SolverStub()
    model = StubProvider(stub)
    bad = []
    moves_total = 0
    for lvl in gate_levels(levels):
        rep = (await run_gridgame(model, [lvl]))[0]["report"]
        moves_total += rep["moves"]
        if not (rep["success"] == 1 and rep["moves"] == lvl["opt_steps"]):
            bad.append(lvl["id"])
    return (len(bad) == 0 and stub.calls == moves_total), bad, stub.calls, moves_total


async def g1d_run(levels, out_path=None):
    stub = stubs.ScriptStub()
    model = StubProvider(stub)
    reports = []
    moves_total = 0
    for lvl in gate_levels(levels):
        rep = (await run_gridgame(model, [lvl]))[0]["report"]
        reports.append(rep)
        moves_total += rep["moves"]
    if out_path:
        Path(out_path).write_text("\n".join(json.dumps(r) for r in reports) + "\n", encoding="utf-8")
    return reports, stub.calls, moves_total


def compare(file_path, golden_path=None):
    golden = Path(golden_path) if golden_path else _GOLDEN
    got = [json.loads(line) for line in Path(file_path).read_text().splitlines() if line.strip()]
    want = [json.loads(line) for line in golden.read_text().splitlines() if line.strip()]
    if got == want:
        print(f"G1d PASS {len(want)} episodes identical to the golden record")
        return 0
    print(f"G1d FAIL: {len(got)} vs {len(want)} golden lines differ")
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            print(f"  first diff at episode {i}:")
            for k in a:
                if a[k] != b.get(k):
                    print(f"    {k}: got={a[k]!r} want={b.get(k)!r}")
            break
    return 1


async def run_all(levels, tiers, seed, out_dir=None):
    ok_overall = True

    print("== G1a (NullStub, 16/level) ==")
    stats, calls = await g1a(levels, tiers, seed=seed)
    ok, msgs = check_g1a(stats, tiers)
    moves = sum(s["moves"] for s in stats.values())
    for key in sorted(stats):
        print(f"  {key[0]}/{key[1]} success={stats[key]['success']:.4f} "
              f"dead={stats[key]['dead']:.4f} score={stats[key]['score']:.4f} "
              f"parse_fail={stats[key]['parse_fail']}")
    print(f"  G1a {'PASS' if ok else 'FAIL'}  calls={calls} moves={moves} equal={calls == moves}")
    for m in msgs:
        print("    -", m)
    ok_overall &= ok and (calls == moves)

    print("== G1b (SilentStub) ==")
    okb, bad, calls, moves = await g1b(levels)
    print(f"  G1b {'PASS' if okb else 'FAIL'}  calls={calls} moves={moves} equal={calls == moves}  bad={bad[:5]}")
    ok_overall &= okb

    print("== G1c (SolverStub) ==")
    okc, bad, calls, moves = await g1c(levels)
    print(f"  G1c {'PASS' if okc else 'FAIL'}  calls={calls} moves={moves} equal={calls == moves}  bad={bad[:5]}")
    ok_overall &= okc

    print("== G1d (ScriptStub) ==")
    out_path = Path(out_dir) / "g1d_harness.jsonl" if out_dir else Path("/tmp/g1d_harness.jsonl")
    reports, calls, moves = await g1d_run(levels, out_path=out_path)
    print(f"  produced {len(reports)} reports  calls={calls} moves={moves} equal={calls == moves}")
    rc = compare(str(out_path))
    ok_overall &= (calls == moves) and (rc == 0)

    print("== G1 OVERALL:", "PASS" if ok_overall else "FAIL", "==")
    return 0 if ok_overall else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", nargs="?", default="run", choices=["run", "compare"])
    p.add_argument("arg", nargs="?", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)

    levels = load_levels()
    tiers = load_tiers()
    if a.cmd == "compare":
        return compare(a.arg)
    return asyncio.run(run_all(levels, tiers, a.seed, a.out_dir))


if __name__ == "__main__":
    sys.exit(main())
