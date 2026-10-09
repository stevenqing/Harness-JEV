# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Phase B report generator for FrozenLake / Sokoban (spec section 9).

Recomputes every number of ``REPORT_gridgames_baseline.md`` from the
per-episode JSONL written by ``baseline_gridgames.py`` together with
``levels.jsonl`` and ``tiers.json``.  This script *is* the recomputation the
spec asks for ("Every number in the report must be recomputable from that file
together with levels.jsonl and tiers.json. Include the script that does it.").

    python -m recipe.agent_evolver.report_gridgames \
        --episodes runs/agent_evolver/gridgames/baseline/episodes.jsonl \
        --out runs/agent_evolver/gridgames/baseline/REPORT_gridgames_baseline.md

Reads the G0 / G1 gate output files (``--g0-out`` / ``--g1-out``) if present and
embeds them verbatim in section 1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import r0

_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR", "/mnt/llmshared-ssd-hd/shishuqing/gridgames")).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
import ta_env  # noqa: E402

GAMES = ["frozenlake", "sokoban"]
TIERS = ["L4", "L8", "L16", "L32"]
MODEL_ORDER = ["Qwen3.5-4B", "Qwen3.5-2B", "Qwen3.5-0.8B"]

# Bootstrap settings (spec section 9): 10,000 draws, stated seed.
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261002

LIVE_DELTA = 0.10  # success - floors.gate.uniform >= 0.10 → live (spec 7 B4)


def load_levels():
    levels = [json.loads(l) for l in (_BUNDLE_DIR / "levels.jsonl").read_text().splitlines() if l.strip()]
    by_id = {}
    for i, lvl in enumerate(levels):
        lvl["_line"] = i
        by_id[lvl["id"]] = lvl
    return levels, by_id


def load_tiers():
    return json.loads((_BUNDLE_DIR / "tiers.json").read_text())


def load_episodes(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def gate_levels_of_cell(levels, game, tier):
    return [lvl for lvl in levels if lvl["split"] == "gate" and lvl["game"] == game and lvl["tier"] == tier]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_success(ep_by_level, n_levels, rng):
    """One bootstrap estimate of success under level-level resampling."""
    ids = list(ep_by_level.keys())
    # resample n_levels levels with replacement (use the cell's gate level ids)
    level_pool = sorted(ep_by_level.keys())
    if not level_pool:
        return float("nan")
    succ = tot = 0
    for _ in range(n_levels):
        lid = rng.choice(level_pool)
        eps = ep_by_level[lid]
        succ += sum(e["report"]["success"] for e in eps)
        tot += len(eps)
    return succ / tot if tot else float("nan")


def bootstrap_ci(ep_by_level, n_levels, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    rng = random.Random(seed)
    est = []
    for _ in range(draws):
        est.append(_bootstrap_success(ep_by_level, n_levels, rng))
    est = sorted(x for x in est if x == x)  # drop NaN
    if not est:
        return float("nan"), float("nan"), float("nan")
    lo = est[int(0.025 * len(est))]
    hi = est[int(0.975 * len(est)) - 1]
    point = sum(est) / len(est)
    return point, lo, hi


def build_cells(levels, by_id, episodes):
    """Group non-left-out episodes by (model, game, tier); also count left_out."""
    gate_ids = {lvl["id"] for lvl in levels if lvl["split"] == "gate"}
    cells = defaultdict(lambda: {"eps": [], "left_out": 0, "ep_by_level": defaultdict(list)})
    for ep in episodes:
        if ep["level_id"] not in gate_ids:
            continue
        key = (ep["model"], ep["game"], ep["tier"])
        if ep.get("left_out"):
            cells[key]["left_out"] += 1
        else:
            cells[key]["eps"].append(ep)
            cells[key]["ep_by_level"][ep["level_id"]].append(ep)
    return cells


def cell_stats(model, game, tier, cell, floors):
    eps = cell["eps"]
    ep_by_level = cell["ep_by_level"]
    n_levels = len(gate_levels_of_cell(load_levels()[0], game, tier))
    succ = mean([e["report"]["success"] for e in eps])
    _, lo, hi = bootstrap_ci(ep_by_level, n_levels)
    f = floors[game][tier]
    total_moves = sum(e["report"]["moves"] for e in eps)
    total_pf = sum(e["report"]["parse_fail"] for e in eps)
    total_blocked = sum(e["report"]["blocked_moves"] for e in eps)
    solved = [e for e in eps if e["report"]["success"]]
    return {
        "model": model, "game": game, "tier": tier,
        "n_levels": n_levels,
        "n_episodes": len(eps),
        "left_out": cell["left_out"],
        "success": succ,
        "success_lo": lo, "success_hi": hi,
        "uniform": f["uniform"],
        "success_minus_uniform": succ - f["uniform"],
        "progress_end": mean([e["report"]["progress_end"] for e in eps]),
        "uniform_progress": f["uniform_progress"],
        "dead": mean([e["report"]["dead"] for e in eps]),
        "parse_fail_per_move": total_pf / total_moves if total_moves else float("nan"),
        "blocked_per_move": total_blocked / total_moves if total_moves else float("nan"),
        "mean_moves": mean([e["report"]["moves"] for e in eps]),
        "solved_excess": mean([e["report"]["moves"] - e["report"]["opt_steps"] for e in solved]),
        "n_solved": len(solved),
    }


def direct_detour(model, tier, cells, by_id):
    """FL success for direct vs detour levels.

    ``detour`` in the manifest is ``opt_steps - manhattan`` (spec §3.7), an
    integer ≥ 0; a *detour level* is one with ``detour > 0``, a *direct* level
    one with ``detour == 0`` (spec §3.3).
    """
    eps = cells[(model, "frozenlake", tier)]["eps"]
    direct = [e for e in eps if by_id[e["level_id"]]["detour"] == 0]
    detour = [e for e in eps if by_id[e["level_id"]]["detour"] > 0]
    return {
        "direct": mean([e["report"]["success"] for e in direct]),
        "direct_n": len(direct),
        "detour": mean([e["report"]["success"] for e in detour]),
        "detour_n": len(detour),
    }


def top_unreadable(model, cells):
    counter = Counter()
    for (m, g, t), cell in cells.items():
        if m != model:
            continue
        for ep in cell["eps"]:
            for mv in ep["moves"]:
                if mv["parser_output"] is None:
                    counter[mv["raw"].strip()] += 1
    return counter.most_common(10)


def pooled_parse_fail(model, cells):
    pf = 0
    moves = 0
    for (m, g, t), cell in cells.items():
        if m != model:
            continue
        for ep in cell["eps"]:
            pf += ep["report"]["parse_fail"]
            moves += ep["report"]["moves"]
    return pf / moves if moves else float("nan")


def working_tier(success_minus_uniform):
    """(working tier, flag) — highest live tier; flag lower non-live tiers."""
    live = {t: success_minus_uniform.get(t, -1.0) >= LIVE_DELTA for t in TIERS}
    working = None
    for t in TIERS:
        if live[t]:
            working = t
    flags = [t for t in TIERS if t != working and not live[t] and working is not None]
    return working, flags


# ── two R0 message examples (spec 9.2) ───────────────────────────────────────

def r0_examples(levels):
    fl = next(l for l in levels if l["game"] == "frozenlake" and l["split"] == "gate" and l["tier"] == "L4")
    sk = next(l for l in levels if l["game"] == "sokoban" and l["split"] == "gate" and l["tier"] == "L4")
    out = {}
    for tag, lvl in (("frozenlake", fl), ("sokoban", sk)):
        env = ta_env.LevelEnv(lvl)
        msgs = r0.messages(env.observation(), [])
        out[tag] = msgs
    return out


def fmt_pct(x):
    return "n/a" if (isinstance(x, float) and x != x) else f"{100 * x:.2f}%"


def fmt_float(x, nd=4):
    return "n/a" if (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", default="runs/agent_evolver/gridgames/baseline/episodes.jsonl")
    p.add_argument("--out", default="runs/agent_evolver/gridgames/baseline/REPORT_gridgames_baseline.md")
    p.add_argument("--g0-out", default="/tmp/gg_g0_output.txt")
    p.add_argument("--g1-out", default="/tmp/gg_g1_output.txt")
    a = p.parse_args(argv)

    levels, by_id = load_levels()
    tiers = load_tiers()
    floors = tiers["floors"]["gate"]
    episodes = load_episodes(a.episodes)
    cells = build_cells(levels, by_id, episodes)

    # Per-cell stats table (Table B1).
    stats = {}
    for model in MODEL_ORDER:
        for game in GAMES:
            for tier in TIERS:
                stats[(model, game, tier)] = cell_stats(model, game, tier, cells[(model, game, tier)], floors)

    # Working tiers + flags.
    wt = {}
    for model in MODEL_ORDER:
        for game in GAMES:
            smu = {t: stats[(model, game, t)]["success_minus_uniform"] for t in TIERS}
            wt[(model, game)] = working_tier(smu)

    # Assemble markdown.
    md = assemble_report(stats, cells, by_id, tiers, wt, levels, episodes, a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    print(f"[report] wrote {a.out}")
    return 0


def assemble_report(stats, cells, by_id, tiers, wt, levels, episodes, a):
    g0 = Path(a.g0_out).read_text(encoding="utf-8").strip() if Path(a.g0_out).exists() else "(G0 output not found)"
    g1 = Path(a.g1_out).read_text(encoding="utf-8").strip() if Path(a.g1_out).exists() else "(G1 output not found)"

    lines = []
    A = lines.append
    A("# REPORT — FrozenLake / Sokoban R0 baseline (Phase B)")
    A("")
    A("Target: `recipe/agent_evolver`, models Qwen3.5-4B / Qwen3.5-2B / Qwen3.5-0.8B. No weights were trained.")
    A("")
    A(f"Bootstrap seed: **{BOOTSTRAP_SEED}**, **{BOOTSTRAP_DRAWS}** percentile draws, level-level resampling (spec 9).")
    A("")

    # 1. G0 + G1
    A("## 1. Gates G0 and G1")
    A("")
    A("### G0 (`verify_levels.py`)")
    A("```")
    A(g0)
    A("```")
    A("")
    A("### G1 (`g1_gridgames.py`, four stub models through the full pipeline)")
    A("```")
    A(g1)
    A("```")
    A("")

    # 2. Versions, decoding, two R0 messages
    A("## 2. Versions, decoding settings, and the two R0 messages")
    A("")
    A("- vLLM **0.28.0** (`/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm`).")
    A("- Models served: `Qwen3.5-4B` (port 8200/GPU 0), `Qwen3.5-2B` (8201/GPU 1), `Qwen3.5-0.8B` (8202/GPU 2). "
      "All `--dtype bfloat16 --max-model-len 32768 --max-num-seqs 256 --tensor-parallel-size 1 --gpu-memory-utilization 0.85`.")
    A("- Hugging Face revision: **not recorded** — the model directories are local snapshots without `_commit_hash` / "
      "revision metadata (vLLM launched with `revision=None`). Recorded as a deviation in §9.9.")
    A("- Decoding (spec §5): thinking off (`chat_template_kwargs.enable_thinking=false`), `max_tokens 32`, "
      "`temperature 0.4`, `top_p 0.8`, `top_k 20`, per-move `seed` = `100 * episode_seed + move`.")
    A("- `runs/agent_evolver/official/` (the '4B generative baseline' of §5) does not exist on this box, so sampling is "
      "the recipe's own 4B baseline (`spec.py`: temp 0.4 / top_p 0.8), a single non-zero-temperature configuration — "
      "the §5 fallback (0.7 / top_k 20) is not triggered; top_k 20 is still pinned. Deviation noted in §9.9.")
    A("")
    ex = r0_examples(levels)
    for game in ("frozenlake", "sokoban"):
        A(f"### R0 messages — one full example, {game}")
        A("")
        for m in ex[game]:
            A(f"**{m['role']}**:")
            A("```")
            A(m["content"])
            A("```")
            A("")
        A("")

    # 3. Table B1
    A("## 3. Table B1 — per model, game, tier")
    A("")
    A("| model | game | tier | eps | left_out | success [95% CI] | uniform | succ−uniform | "
      "mean progress_end | uniform_progress | dead | parse_fail/move | blocked/move | mean moves | solved excess moves |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODEL_ORDER:
        for game in GAMES:
            for tier in TIERS:
                s = stats[(model, game, tier)]
                ci = f"{fmt_pct(s['success'])} [{fmt_pct(s['success_lo'])}, {fmt_pct(s['success_hi'])}]"
                A(f"| {model} | {game} | {tier} | {s['n_episodes']} | {s['left_out']} | {ci} | "
                  f"{fmt_float(s['uniform'])} | {fmt_float(s['success_minus_uniform'])} | "
                  f"{fmt_float(s['progress_end'])} | {fmt_float(s['uniform_progress'])} | "
                  f"{fmt_float(s['dead'])} | {fmt_float(s['parse_fail_per_move'])} | {fmt_float(s['blocked_per_move'])} | "
                  f"{fmt_float(s['mean_moves'], 1)} | {fmt_float(s['solved_excess'], 2)} |")
    A("")
    A("*`success [95% CI]` is the percentile bootstrap over levels. `parse_fail/move` and `blocked/move` are totals "
      "over the cell (spec §9). `solved excess moves` = mean(moves − opt_steps) over solved episodes only.*")
    A("")

    # 4. Table B2
    A("## 4. Table B2 — FrozenLake success, direct vs detour")
    A("")
    A("| model | tier | direct success | direct n | detour success | detour n |")
    A("|---|---|---|---|---|---|")
    for model in MODEL_ORDER:
        for tier in TIERS:
            dd = direct_detour(model, tier, cells, by_id)
            A(f"| {model} | {tier} | {fmt_pct(dd['direct'])} | {dd['direct_n']} | "
              f"{fmt_pct(dd['detour'])} | {dd['detour_n']} |")
    A("")
    A("*L4 has only 2 detour levels (16 episodes), so its detour entry is a count, not a reliable rate (spec §9.4).*")
    A("")

    # 5. Working tiers
    A("## 5. Working tier per model per game")
    A("")
    A("| model | game | L4 succ−floor | L8 | L16 | L32 | working tier | flagged lower non-live |")
    A("|---|---|---|---|---|---|---|---|")
    for model in MODEL_ORDER:
        for game in GAMES:
            smu = [fmt_float(stats[(model, game, t)]["success_minus_uniform"], 3) for t in TIERS]
            working, flags = wt[(model, game)]
            wstr = working if working else "**none**"
            fstr = ", ".join(flags) if flags else "—"
            A(f"| {model} | {game} | {smu[0]} | {smu[1]} | {smu[2]} | {smu[3]} | {wstr} | {fstr} |")
    A("")
    A("*Live = success − `floors.gate.uniform` ≥ 0.10 (spec §7 B4). Working tier = highest live tier.*")
    A("")

    # 6. Top-10 unreadable replies
    A("## 6. Ten most frequent unreadable replies per model")
    A("")
    for model in MODEL_ORDER:
        A(f"### {model}")
        A("```")
        for raw, n in top_unreadable(model, cells):
            A(f"{n:6d}  {raw!r}")
        if not top_unreadable(model, cells):
            A("(none — no unreadable replies)")
        A("```")
        A("")

    # 7. Predictions
    A("## 7. Predictions (section 8) — verdicts")
    A("")
    A(build_prediction_text(stats, cells, by_id, wt))
    A("")

    # 8. What the data do not support
    A("## 8. What the data do not support")
    A("")
    A(_what_data_do_not_support(stats, wt))
    A("")

    # 9. Deviations
    A("## 9. Deviations from this spec")
    A("")
    A(_deviations())
    A("")

    return "\n".join(lines)


def build_prediction_text(stats, cells, by_id, wt):
    out = []
    A = out.append
    # P1
    viol1 = []
    for model in MODEL_ORDER:
        for game in GAMES:
            for a_, b in zip(TIERS, TIERS[1:]):
                sa = stats[(model, game, a_)]["success"]
                sb = stats[(model, game, b)]["success"]
                if sb - sa > 0.05:
                    viol1.append(f"{model}/{game} {a_}→{b}: {fmt_pct(sa)} → {fmt_pct(sb)} (+{fmt_pct(sb - sa)})")
    A(f"**P1** (success never rises tier-to-tier): **{'WRONG' if viol1 else 'HOLD'}**")
    if viol1:
        for v in viol1:
            A(f"  - {v}")
    A("")
    # P2
    viol2 = []
    for game in GAMES:
        for tier in ["L4", "L8"]:
            for big, small in (("Qwen3.5-4B", "Qwen3.5-2B"), ("Qwen3.5-2B", "Qwen3.5-0.8B"), ("Qwen3.5-4B", "Qwen3.5-0.8B")):
                sb = stats[(small, game, tier)]["success"]
                sg = stats[(big, game, tier)]["success"]
                if sb - sg > 0.05:
                    viol2.append(f"{game}/{tier}: {small} ({fmt_pct(sb)}) > {big} ({fmt_pct(sg)})")
    A(f"**P2** (L4/L8 success ordered 4B ≥ 2B ≥ 0.8B): **{'WRONG' if viol2 else 'HOLD'}**")
    if viol2:
        for v in viol2:
            A(f"  - {v}")
    A("")
    # P3
    sk_l4_08 = stats[("Qwen3.5-0.8B", "sokoban", "L4")]["success_minus_uniform"]
    A(f"**P3** (Sokoban L4 not live for 0.8B): **{'WRONG (it is live)' if sk_l4_08 >= LIVE_DELTA else 'HOLD'}** "
      f"— success − floor = {sk_l4_08:.4f} (live threshold {LIVE_DELTA}).")
    A("")
    # P4
    hit = [(m, g, stats[(m, g, "L32")]["success"]) for m in MODEL_ORDER for g in GAMES if stats[(m, g, "L32")]["success"] >= 0.05]
    A(f"**P4** (no model reaches 0.05 success in L32): **{'WRONG' if hit else 'HOLD'}**")
    if hit:
        for m, g, s in hit:
            A(f"  - {m}/{g} L32 success = {fmt_pct(s)}")
    A("")
    # P5
    dd = direct_detour("Qwen3.5-4B", "L8", cells, by_id)
    gap = dd["direct"] - dd["detour"]
    A(f"**P5** (4B FL L8 direct ≥ detour + 0.10): **{'HOLD' if gap >= 0.10 else 'WRONG'}** "
      f"— direct {fmt_pct(dd['direct'])} vs detour {fmt_pct(dd['detour'])} (gap {fmt_pct(gap)}).")
    A("")
    # P6
    any_bad = False
    rows = []
    for model in MODEL_ORDER:
        pf = pooled_parse_fail(model, cells)
        if pf >= 0.05:
            any_bad = True
        rows.append(f"  - {model}: pooled parse_fail/move = {fmt_float(pf)}")
    A(f"**P6** (parse_fail/move < 0.05 for every model): **{'WRONG' if any_bad else 'HOLD'}**")
    A("\n".join(rows))
    return "\n".join(out)


def _what_data_do_not_support(stats, wt):
    out = []
    A = out.append
    A("This section states what cannot be concluded from these numbers, per spec §9.8.")
    A("")
    A("- **No tier-to-tier *monotonicity* is asserted as a model property.** Each cell is a distinct level pool; "
      "differences across tiers are differences across those pools, not a within-model difficulty curve.")
    A("- **Detour vs direct is only a clean comparison in L8** (spec §3.3): L4 has 2 detour levels, and L16/L32 "
      "detour levels are longer on average, so a raw direct/detour gap there confounds length with detour.")
    A("- **Working tier is a threshold on the gate split only.** It is not a claim about the test split, which is "
      "played once per final harness in Phase C and is not touched here.")
    A("- **`success` is reported but is never the selection signal** (spec §6.6); mean `progress_end` is. Both are "
      "listed; nothing here picks a harness.")
    A("- **These are post-trained Qwen3.5 models under one fixed prompt (R0), one decoding, 4 episodes/level.** "
      "They are not a statement about what these models can do with a different prompt, more episodes, or training.")
    A("- **A live tier for a model does not mean the *model* is 'good at' that game** — only that its success clears "
      "the uniform-random floor by ≥ 0.10 on the gate split. A tier can be live with a low absolute success rate.")
    return "\n".join(out)


def _deviations():
    return (
        "- **Bundle was rebuilt from scratch** (the `gridgames/` directory and its `SHA256SUMS`/`levels.jsonl` hashes "
        "pinned in the spec did not exist on this box). The spec's frozen `SHA256SUMS` "
        "`cf9688bb4bbd3300c1c5d5c20fd48447127dd30c33861ffdcb8559b065e0e3ea` and `levels.jsonl` "
        "`5f869824af8bec64d8000eca8dbdd6b6b418103fed351ba5f4e0419a1035e0ea` are therefore unreproducible; the actual "
        "bundle `SHA256SUMS` is `738aaaa165a130c3992b6ecc2a1a98659320cf0201bbb475d7eaa1b8d0f2961d` and G0 was run "
        "against that value. Consequently the `floors.gate` numbers in `tiers.json` differ slightly from the "
        "illustrative §3.4 table (e.g. FL L4 `uniform` 0.0569 here vs 0.0634 in the spec); all report floors are read "
        "from the actual `tiers.json`."
        "\n- **`runs/agent_evolver/official/` does not exist**, so the §5 '4B generative baseline' decoding could not be "
        "read; sampling was taken from the recipe's `spec.py` baseline (temp 0.4 / top_p 0.8) with `top_k 20` and "
        "`max_tokens 32`, thinking off."
        "\n- **Hugging Face revision not recorded**: the model directories are local snapshots with no `_commit_hash` / "
        "revision metadata, and vLLM reports `revision=None`. No revision string can be stated."
        "\n- **G1a `NullStub` seed**: the gate script uses its default seed 1234; the spec leaves the stub seed free."
        "\n- **Meta-agent file access cannot be restricted on this box** (spec §6.1): the bundle directory is outside "
        "the repo tree and nothing in Phase A/B feeds it to the meta-agent, but there is no OS-level guarantee that the "
        "meta-agent process could not read it. Phase C (evolution) is not run under this spec."
    )


if __name__ == "__main__":
    sys.exit(main())
