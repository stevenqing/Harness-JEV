#!/usr/bin/env python3
"""§4 offline verification: task historical flip-rate vs meta predicted_affected.

Zero rollout, zero training, zero API calls.  Pure read-only statistics over
the on-disk Step-0b artefacts.

Reuses Step-0b's loader (analyze.load_cells / analyze.pass_mean) so the
pass/fail semantics are identical to B1:
  - ALFWorld: pass = mean of up to 3 binary replicates, threshold 0.5
  - WebShop : pass = mean of up to 3 raw-reward replicates, threshold 0.5
  - a cell exactly at 0.5 is dropped (ties carry no flip signal), same as B1.

flip_rate[(stratum, task_id)] = (# parent-fail -> candidate-pass) /
                                (# parent-fail) over ALL included adjacent
                                parent->candidate pairs (spec §4.1, full sample).
No history -> the stratum's overall flip mean.

Evaluation unit = a parent->candidate pair restricted to parent-FAILED tasks
(parent-passed tasks are regress candidates, out of scope per spec §4.2).
True label y = 1 if candidate pass (flip), 0 if candidate fail (stay-F).

Writes:
  task_fliprate.json   per-task flip rates + stratum means
  metrics.json         per-stratum metrics + per-pair records
  report.md            §4 report (with the GO/NO-GO verdict)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV/data/kev_ranker_s0b")
import analyze  # noqa: E402  (reuse Step-0b loader)

OUT = Path("/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV/data/predicted_affected_ablation")
SEED = 20261011
B = 1000
STRATA = ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]
FIXED_KS = [4, 8, 16]

TASK_IDS = analyze.TASK_IDS  # {"alfworld": 0..63, "webshop": 1500..1563}


def roc(y, score):
    """Per-pair AUC; None if degenerate (single class or single score value)."""
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    if len(set(y)) < 2 or len(set(score)) < 2:
        return None
    # sklearn roc_auc_score is imported by analyze; use it for exact parity with B1
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def bootstrap_mean(run_vals, seed=SEED, nb=B):
    """Cluster bootstrap (by run) of the mean. run_vals: {run: [float,...]}."""
    runs = sorted(run_vals)
    all_vals = [v for r in runs for v in run_vals[r]]
    if not all_vals:
        return None, None, None
    mean = float(np.mean(all_vals))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(nb):
        sampled = rng.choice(runs, size=len(runs), replace=True)
        vals = [v for r in sampled for v in run_vals[r]]
        if vals:
            boot.append(float(np.mean(vals)))
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return mean, lo, hi


def main() -> None:
    pairs = analyze.load_jsonl(analyze.OUT / "real_pairs.jsonl")
    cells = analyze.load_cells()

    # ── step 1: per-task flip rate (full sample) ──────────────────────────────
    num = defaultdict(int)
    den = defaultdict(int)
    for row in pairs:
        stratum = row["stratum"]
        bench = stratum.split("_")[0]
        for tid in TASK_IDS[bench]:
            pcell = cells.get((stratum, row["run"], row["parent_round"], tid))
            ccell = cells.get((stratum, row["run"], row["round"], tid))
            if pcell is None or ccell is None:
                continue
            pp = analyze.pass_mean(pcell, bench)
            cp = analyze.pass_mean(ccell, bench)
            if pp is None or cp is None or abs(pp - 0.5) < 1e-9:
                continue
            if pp < 0.5:  # parent failed
                den[(stratum, tid)] += 1
                if cp > 0.5:  # candidate passed -> flip
                    num[(stratum, tid)] += 1

    stratum_mean = {}
    for stratum in STRATA:
        s_num = sum(n for (s, _), n in num.items() if s == stratum)
        s_den = sum(d for (s, _), d in den.items() if s == stratum)
        stratum_mean[stratum] = (s_num / s_den) if s_den else 0.0

    flip_rate = {}
    for (stratum, tid), d in den.items():
        flip_rate[(stratum, tid)] = num[(stratum, tid)] / d if d else stratum_mean[stratum]

    def fr_of(stratum, tid):
        return flip_rate.get((stratum, tid), stratum_mean[stratum])

    # ── step 2: per-pair evaluation ───────────────────────────────────────────
    pair_records = []
    for row in pairs:
        stratum = row["stratum"]
        bench = stratum.split("_")[0]
        pa = set(row["predicted_affected"] or [])
        tasks = []
        for tid in TASK_IDS[bench]:
            pcell = cells.get((stratum, row["run"], row["parent_round"], tid))
            ccell = cells.get((stratum, row["run"], row["round"], tid))
            if pcell is None or ccell is None:
                continue
            pp = analyze.pass_mean(pcell, bench)
            cp = analyze.pass_mean(ccell, bench)
            if pp is None or cp is None or abs(pp - 0.5) < 1e-9:
                continue
            if pp >= 0.5:  # parent passed -> out of scope (regress problem)
                continue
            y = 1.0 if cp > 0.5 else 0.0
            tasks.append({
                "tid": tid,
                "y": y,
                "meta": 1.0 if tid in pa else 0.0,
                "fr": fr_of(stratum, tid),
            })
        if not tasks:
            continue

        ys = [t["y"] for t in tasks]
        n_flips = int(sum(ys))
        k_meta = int(sum(t["meta"] for t in tasks))  # |predicted_affected ∩ parent-failed|

        rec = {
            "stratum": stratum, "run": row["run"], "round": row["round"],
            "parent_round": row["parent_round"],
            "n_failed": len(tasks), "n_flips": n_flips, "k_meta": k_meta,
        }
        rec["meta_auc"] = roc(ys, [t["meta"] for t in tasks])
        rec["fr_auc"] = roc(ys, [t["fr"] for t in tasks])

        # aux precision/recall at k = |pa ∩ parent-failed|
        if k_meta > 0:
            meta_flips = int(sum(t["y"] for t in tasks if t["meta"] == 1.0))
            rec["meta_prec_at_k"] = meta_flips / k_meta
            rec["meta_rec_at_k"] = (meta_flips / n_flips) if n_flips else None
            order = sorted(tasks, key=lambda t: (-t["fr"], t["tid"]))[:k_meta]
            fr_flips = int(sum(t["y"] for t in order))
            rec["fr_prec_at_k"] = fr_flips / k_meta
            rec["fr_rec_at_k"] = (fr_flips / n_flips) if n_flips else None
        else:
            rec["meta_prec_at_k"] = rec["meta_rec_at_k"] = None
            rec["fr_prec_at_k"] = rec["fr_rec_at_k"] = None

        # fixed-k sensitivity (fliprate only): require at least k failed tasks,
        # then predict exactly top-k by flip_rate.
        order = sorted(tasks, key=lambda t: (-t["fr"], t["tid"]))
        for k in FIXED_KS:
            if len(order) >= k:
                top = order[:k]
                fl = int(sum(t["y"] for t in top))
                rec[f"fr_prec_k{k}"] = fl / k
                rec[f"fr_rec_k{k}"] = (fl / n_flips) if n_flips else None
            else:
                rec[f"fr_prec_k{k}"] = rec[f"fr_rec_k{k}"] = None
        pair_records.append(rec)

    # ── step 3: aggregate per stratum ─────────────────────────────────────────
    metrics = {"seed": SEED, "bootstrap_b": B, "strata": {}}
    for stratum in STRATA:
        prs = [p for p in pair_records if p["stratum"] == stratum]
        runs = sorted({p["run"] for p in prs})

        def agg_metric(key, *, none_ok=False):
            g = defaultdict(list)
            for p in prs:
                v = p.get(key)
                if v is not None:
                    g[p["run"]].append(v)
            m, lo, hi = bootstrap_mean(g)
            return {"mean": m, "ci95": (lo, hi), "n_pairs": sum(len(v) for v in g.values())}

        out = {
            "n_runs": len(runs),
            "n_pairs": len(prs),
            "meta_auc": agg_metric("meta_auc"),
            "fliprate_auc": agg_metric("fr_auc"),
            "meta_prec_at_k": agg_metric("meta_prec_at_k"),
            "meta_rec_at_k": agg_metric("meta_rec_at_k"),
            "fliprate_prec_at_k": agg_metric("fr_prec_at_k"),
            "fliprate_rec_at_k": agg_metric("fr_rec_at_k"),
        }
        # paired diff (only pairs where both AUCs are defined)
        diff_g = defaultdict(list)
        for p in prs:
            if p.get("meta_auc") is not None and p.get("fr_auc") is not None:
                diff_g[p["run"]].append(p["fr_auc"] - p["meta_auc"])
        dm, dlo, dhi = bootstrap_mean(diff_g)
        out["auc_diff"] = {"mean": dm, "ci95": (dlo, dhi),
                           "n_pairs": sum(len(v) for v in diff_g.values())}
        # paired diff for precision@k / recall@k
        for key in ("prec_at_k", "rec_at_k"):
            dg = defaultdict(list)
            for p in prs:
                a, b = p.get(f"meta_{key}"), p.get(f"fr_{key}")
                if a is not None and b is not None:
                    dg[p["run"]].append(b - a)
            dm, dlo, dhi = bootstrap_mean(dg)
            out[f"{key}_diff"] = {"mean": dm, "ci95": (dlo, dhi),
                                  "n_pairs": sum(len(v) for v in dg.values())}
        # fixed-k sensitivity
        sens = {}
        for k in FIXED_KS:
            sens[str(k)] = {
                "precision": agg_metric(f"fr_prec_k{k}"),
                "recall": agg_metric(f"fr_rec_k{k}"),
            }
        out["k_sensitivity"] = sens
        metrics["strata"][stratum] = out

    # ── leave-one-run-out flip-rate (leakage / B1 parity check) ───────────────
    loo_metrics = {}
    for stratum in STRATA:
        prs = [p for p in pair_records if p["stratum"] == stratum]
        runs = sorted({p["run"] for p in prs})
        held = {}
        for h in runs:
            # train flip rates from pairs of other runs
            tnum, tden = defaultdict(int), defaultdict(int)
            for p in prs:
                if p["run"] == h:
                    continue
                bench = stratum.split("_")[0]
                for tid in TASK_IDS[bench]:
                    pcell = cells.get((stratum, p["run"], p["parent_round"], tid))
                    ccell = cells.get((stratum, p["run"], p["round"], tid))
                    if pcell is None or ccell is None:
                        continue
                    pp = analyze.pass_mean(pcell, bench)
                    cp = analyze.pass_mean(ccell, bench)
                    if pp is None or cp is None or abs(pp - 0.5) < 1e-9:
                        continue
                    if pp < 0.5:
                        tden[tid] += 1
                        if cp > 0.5:
                            tnum[tid] += 1
            t_mean = sum(tnum.values()) / sum(tden.values()) if sum(tden.values()) else 0.0
            aucs = []
            for p in prs:
                if p["run"] != h:
                    continue
                bench = stratum.split("_")[0]
                ys, frs = [], []
                for tid in TASK_IDS[bench]:
                    pcell = cells.get((stratum, p["run"], p["parent_round"], tid))
                    ccell = cells.get((stratum, p["run"], p["round"], tid))
                    if pcell is None or ccell is None:
                        continue
                    pp = analyze.pass_mean(pcell, bench)
                    cp = analyze.pass_mean(ccell, bench)
                    if pp is None or cp is None or abs(pp - 0.5) < 1e-9:
                        continue
                    if pp >= 0.5:
                        continue
                    ys.append(1.0 if cp > 0.5 else 0.0)
                    frs.append(tnum[tid] / tden[tid] if tden[tid] else t_mean)
                a = roc(ys, frs)
                if a is not None:
                    aucs.append(a)
            if aucs:
                held[h] = float(np.mean(aucs))
        if held:
            loo_metrics[stratum] = {
                "auc_loo": float(np.mean(list(held.values()))),
                "n_held_out_runs": len(held),
                "per_run": {k: round(v, 4) for k, v in held.items()},
            }
        else:
            loo_metrics[stratum] = {"auc_loo": None, "n_held_out_runs": 0}
    metrics["loo_robustness"] = loo_metrics

    # ── write task_fliprate.json ──────────────────────────────────────────────
    tfj = {}
    for stratum in STRATA:
        d = {"stratum_mean": round(stratum_mean[stratum], 6)}
        for tid in TASK_IDS[stratum.split("_")[0]]:
            key = (stratum, tid)
            d[str(tid)] = {
                "flip_rate": round(fr_of(stratum, tid), 6),
                "n_flips": num.get(key, 0),
                "n_parent_fails": den.get(key, 0),
            }
        tfj[stratum] = d
    (OUT / "task_fliprate.json").write_text(
        json.dumps(tfj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── write metrics.json ────────────────────────────────────────────────────
    metrics["pair_records"] = pair_records
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── verdict ───────────────────────────────────────────────────────────────
    def passes(stratum):
        s = metrics["strata"][stratum]
        d = s["auc_diff"]
        return (d["mean"] is not None and d["ci95"][0] is not None
                and d["mean"] >= 0 and d["ci95"][0] > 0)
    go = passes("alfworld_4b") and passes("webshop_4b")
    metrics["verdict"] = {
        "GO": bool(go),
        "alfworld_4b_passes": passes("alfworld_4b"),
        "webshop_4b_passes": passes("webshop_4b"),
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── report.md ─────────────────────────────────────────────────────────────
    write_report(metrics, tfj)

    # console summary
    for stratum in STRATA:
        s = metrics["strata"][stratum]
        print(f"{stratum}: meta_auc={s['meta_auc']['mean']} fliprate_auc={s['fliprate_auc']['mean']} "
              f"diff={s['auc_diff']['mean']} ci={s['auc_diff']['ci95']}")
    print("LOO (B1 parity):", {k: round(v['auc_loo'], 3) if v['auc_loo'] is not None else None
                               for k, v in loo_metrics.items()})
    print("VERDICT:", "GO" if go else "NO-GO", metrics["verdict"])


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_ci(ci):
    if ci is None or ci[0] is None or ci[1] is None:
        return "—"
    return f"({ci[0]:.3f}, {ci[1]:.3f})"


def write_report(metrics, tfj):
    L = []
    L.append("# §4 离线验证 — task 历史翻转率 vs meta `predicted_affected`\n")
    L.append(f"seed={SEED}, cluster bootstrap by run, B={B}\n")
    L.append("口径：pass/fail = Step-0b B1（ALF 三复本均值、WS 三复本奖励均值，阈值 0.5）；"
             "flip = parent 失败 → candidate 通过。只评估 parent 失败的 task。\n")

    v = metrics["verdict"]
    verdict = "GO" if v["GO"] else "NO-GO"
    L.append("\n## 1. 验收结论\n")
    L.append(f"**{verdict}**\n")
    L.append(f"- ALFWorld 4B：fliprate vs meta AUC 差值 CI95 下限 {v['alfworld_4b_passes'] and '> 0（通过）' or '≤ 0（不通过）'}\n")
    L.append(f"- WebShop 4B：fliprate vs meta AUC 差值 CI95 下限 {v['webshop_4b_passes'] and '> 0（通过）' or '≤ 0（不通过）'}\n")
    L.append("\n（验收标准：fliprate 版在 ALFWorld 4B 与 WebShop 4B 的 AUC 都 ≥ meta 版，且差值 CI 下限 > 0。）\n")

    L.append("\n## 2. AUC 对比（主指标）\n")
    L.append("| stratum | meta AUC | meta CI95 | fliprate AUC | fliprate CI95 | diff (fr−meta) | diff CI95 | n_pairs(成对) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for stratum in ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]:
        s = metrics["strata"][stratum]
        L.append(f"| {stratum} | {fmt(s['meta_auc']['mean'])} | {fmt_ci(s['meta_auc']['ci95'])} | "
                 f"{fmt(s['fliprate_auc']['mean'])} | {fmt_ci(s['fliprate_auc']['ci95'])} | "
                 f"{fmt(s['auc_diff']['mean'], 3) if s['auc_diff']['mean'] is not None else '—'} | "
                 f"{fmt_ci(s['auc_diff']['ci95'])} | {s['auc_diff']['n_pairs']} |")
    L.append("")

    L.append("\n## 3. precision@k / recall@k（k = 每对 |predicted_affected ∩ parent失败|）\n")
    L.append("| stratum | meta P@k | fliprate P@k | P@k diff | meta R@k | fliprate R@k | R@k diff |")
    L.append("|---|---|---|---|---|---|---|")
    for stratum in ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]:
        s = metrics["strata"][stratum]
        L.append(f"| {stratum} | {fmt(s['meta_prec_at_k']['mean'])} | {fmt(s['fliprate_prec_at_k']['mean'])} | "
                 f"{fmt(s['prec_at_k_diff']['mean'])} | {fmt(s['meta_rec_at_k']['mean'])} | "
                 f"{fmt(s['fliprate_rec_at_k']['mean'])} | {fmt(s['rec_at_k_diff']['mean'])} |")
    L.append("（P@k/R@k 为跨对均值；diff 为 fliprate − meta，成对，只在两版都有定义的 pair 上算。）\n")

    L.append("\n## 4. k 敏感度（fliprate 版 top-k 的 precision/recall，固定 k）\n")
    L.append("| stratum | k | P@k | R@k | n_pairs |")
    L.append("|---|---|---|---|---|")
    for stratum in ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]:
        for k in [4, 8, 16]:
            e = metrics["strata"][stratum]["k_sensitivity"][str(k)]
            L.append(f"| {stratum} | {k} | {fmt(e['precision']['mean'])} | {fmt(e['recall']['mean'])} | "
                     f"{e['precision']['n_pairs']} |")
    L.append("")

    L.append("\n## 5. 样本量\n")
    L.append("| stratum | n_runs | n_pairs | meta AUC 可用对 | fliprate AUC 可用对 | 成对 diff 可用对 |")
    L.append("|---|---|---|---|---|---|")
    for stratum in ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]:
        s = metrics["strata"][stratum]
        L.append(f"| {stratum} | {s['n_runs']} | {s['n_pairs']} | {s['meta_auc']['n_pairs']} | "
                 f"{s['fliprate_auc']['n_pairs']} | {s['auc_diff']['n_pairs']} |")
    L.append("")

    L.append("\n## 6. 口径与注意事项\n")
    L.append("- flip_rate 为**全样本**（所有纳入 run 的相邻 parent→candidate 对）计算，含自包含，"
             "因此 fliprate AUC 是 in-sample 上界（spec §4.1 明确口径）。")
    loo = metrics["loo_robustness"]
    L.append("- leave-one-run-out（LOO）翻转率 AUC 作为泄漏/B1 对齐检查：")
    for stratum in ["alfworld_4b", "alfworld_2b", "webshop_4b", "webshop_2b"]:
        a = loo[stratum]["auc_loo"]
        L.append(f"  - {stratum}: LOO AUC = {fmt(a)}（n={loo[stratum]['n_held_out_runs']} runs）")
    L.append("  - 4B 两 stratum 的 LOO AUC 应对齐 Step-0b B1（0.747 / 0.810）；全样本值高于 LOO 即自包含增益。")
    L.append("- parent 通过（pass_mean ≥ 0.5）的 task 不进评估（回归问题，spec §4 只管 flip）。")
    L.append("- pass_mean 恰等于 0.5 的 cell 丢弃（与 B1 一致）。")
    L.append("\n")

    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
