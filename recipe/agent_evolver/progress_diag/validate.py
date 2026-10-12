# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""进度诊断第一关的验证入口（§4.4 → G3 GO/NO-GO）。**无 meta 调用。**

组合三件事：oracle 卡位 m（``oracle.py``）→ kev 卡位 k（``judge.py``）→ 算
覆盖率 / 卡位准确率 / 分布 gate。指标定义见 ``align.md`` §3/§4。

两种 benchmark 的 oracle 来源不同：

- **ALFWorld**：post-hoc 可恢复（``.md`` + ``env_url`` → ``traj_data.json``）。
- **WebShop**：oracle 需要 buy 时产生的 ``reward_info``，现已随 ``.md`` 前端持久化
  （``trajectories.render_frontmatter`` 落 ``reward_info`` 字段），所以同样 post-hoc
  可恢复（``oracle.resolve_ws_tasks``）。``--ws-labels`` 只在轨迹写于该字段落地之前
  时作为 override 使用。

用法：:

    python -m recipe.agent_evolver.progress_diag.validate alfworld \
        --trajectories-dir runs/evolve/<tag>/R0/trajectories \
        --env-url http://127.0.0.1:18082

    python -m recipe.agent_evolver.progress_diag.validate webshop \
        --trajectories-dir runs/evolve/<tag>/R0/trajectories
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from collections import Counter
from pathlib import Path

from ..decision_prior import _parse_trajectory, _sort_key
from .conditions import alf_conditions, ws_conditions
from .judge import judge_trajectories
from .oracle import alf_milestone, resolve_alf_tasks, resolve_ws_tasks, ws_milestone

# 卡位序号空间：ALF 2 里程碑，WS 6 里程碑（见 align.md §1）。
_MILESTONE_COUNT = {"alfworld": 2, "webshop": 6}


def _norm_dist(vals: list[int], m: int) -> dict[int, float]:
    """归一化的卡位分布（支持 1..m，缺失位补 0，和为 1）。"""
    cnt = Counter(v for v in vals if v is not None)
    total = sum(cnt.values()) or 1
    return {i: cnt.get(i, 0) / total for i in range(1, m + 1)}


def _js_divergence(p: dict[int, float], q: dict[int, float]) -> float:
    """Jensen–Shannon 散度（对称 KL，取值 0..log2）。"""
    keys = sorted(set(p) | set(q))
    pp = [p.get(k, 0.0) for k in keys]
    qq = [q.get(k, 0.0) for k in keys]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

    mix = [(a + b) / 2 for a, b in zip(pp, qq)]
    return 0.5 * (kl(pp, mix) + kl(qq, mix))


def _constant_baseline(ms: list[int]) -> float:
    """常数基线：always 猜最常见的 oracle m（背 base rate，不读轨迹）。"""
    if not ms:
        return 0.0
    return Counter(ms).most_common(1)[0][1] / len(ms)


async def run_validation(
    *,
    benchmark: str,
    trajectories_dir: Path,
    kev_base_url: str,
    env_url: str | None,
    seed: int,
    ws_labels: dict[str, str] | None,
    concurrency: int,
) -> dict:
    """跑完整个第一关，返回带全部指标的报告 dict。"""
    m_count = _MILESTONE_COUNT[benchmark]

    # ── 1. oracle 卡位 m + kev 条件构造 ──────────────────────────────────────
    if benchmark == "alfworld":
        if not env_url:
            raise ValueError("alfworld needs --env-url to recover traj_data.json")
        resolved = resolve_alf_tasks(env_url, seed, trajectories_dir)
        milestones: dict[str, int] = {}
        for tid, (task, actions) in resolved.items():
            mm = alf_milestone(task, actions)
            if mm is not None:
                milestones[tid] = mm
        builder = lambda tid: alf_conditions(resolved[tid][0]) if tid in resolved else []  # noqa: E731
    elif benchmark == "webshop":
        # Post-hoc recovery from the persisted reward_info frontmatter field;
        # --ws-labels (live in-memory record) overrides where present, e.g. for
        # trajectories written before the field existed.
        resolved = resolve_ws_tasks(trajectories_dir)
        if ws_labels:
            resolved.update({str(k): v for k, v in ws_labels.items()})
        milestones = {
            tid: m
            for tid, label in resolved.items()
            if (m := ws_milestone(label)) is not None
        }
        builder = lambda _tid: ws_conditions()  # noqa: E731
    else:
        raise NotImplementedError(benchmark)

    # ── 2. kev 逐条判 → 卡位 k ───────────────────────────────────────────────
    judged, judging_s = await judge_trajectories(
        kev_base_url=kev_base_url,
        trajectories_dir=trajectories_dir,
        condition_builder=builder,
        concurrency=concurrency,
    )

    # ── 3. 覆盖 + 对齐 ───────────────────────────────────────────────────────
    n_failed = sum(
        1 for p in sorted(trajectories_dir.glob("*.md"), key=_sort_key)
        if _parse_trajectory(p) is not None
    )
    common = [tid for tid in milestones if tid in judged]
    ms = [milestones[t] for t in common]
    ks = [judged[t]["k"] for t in common]
    coverage = len(common) / n_failed if n_failed else 0.0

    # WebShop 的 not_bought（m=6）是粗粒度：oracle 只知「搜了没买」，kev 的 k
    # 落在 2..6 都是「一致」而非「精确」。精确准确率只在细粒度 m（1..5）上算。
    if benchmark == "webshop":
        fine = [(k, m) for k, m in zip(ks, ms) if m != m_count]
        not_bought = [k for k, m in zip(ks, ms) if m == m_count]
        exact_accuracy = sum(k == m for k, m in fine) / len(fine) if fine else None
        not_bought_consistency = (
            sum(k >= 2 for k in not_bought) / len(not_bought) if not_bought else None
        )
        fine_ms = [m for _, m in fine]
        constant_baseline = _constant_baseline(fine_ms)
    else:
        exact_accuracy = sum(k == m for k, m in zip(ks, ms)) / len(ks) if ks else None
        not_bought_consistency = None
        constant_baseline = _constant_baseline(ms)

    kev_dist = _norm_dist(ks, m_count)
    oracle_dist = _norm_dist(ms, m_count)
    js = _js_divergence(kev_dist, oracle_dist)

    report = {
        "benchmark": benchmark,
        "n_failed": n_failed,
        "n_covered": len(common),
        "coverage": coverage,
        "exact_accuracy": exact_accuracy,
        "constant_baseline": constant_baseline,
        "not_bought_consistency": not_bought_consistency,
        "kev_k_distribution": kev_dist,
        "oracle_m_distribution": oracle_dist,
        "js_divergence": js,
        "judging_seconds": judging_s,
    }
    return report


def _print_report(report: dict) -> None:
    print(f"\n=== progress-diagnosis first gate ({report['benchmark']}) ===")
    print(f"failed trajectories: {report['n_failed']}  covered: {report['n_covered']} "
          f"(coverage {report['coverage']:.3f})")
    acc = report["exact_accuracy"]
    if acc is not None:
        print(f"卡位准确率 P(k==m): {acc:.3f}  (常数基线 {report['constant_baseline']:.3f})")
    if report["not_bought_consistency"] is not None:
        print(f"not_bought 一致性 P(k>=2): {report['not_bought_consistency']:.3f}")
    print("kev   卡位分布:", {i: f"{v:.3f}" for i, v in report["kev_k_distribution"].items()})
    print("oracle 卡位分布:", {i: f"{v:.3f}" for i, v in report["oracle_m_distribution"].items()})
    print(f"JS(kev||oracle) = {report['js_divergence']:.3f}   judging {report['judging_seconds']:.1f}s")
    print("(gate: 准确率 > 常数基线 且 JS(kev||oracle) < 0.2，两 benchmark 统一；见 align.md §4)")


def main() -> None:
    ap = argparse.ArgumentParser(description="进度诊断第一关验证（无 meta）")
    ap.add_argument("benchmark", choices=["alfworld", "webshop"])
    ap.add_argument("--trajectories-dir", required=True, help="R{n}/trajectories 目录")
    ap.add_argument("--kev-base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--env-url", default=None, help="ALFWorld env server（recover traj_data.json）")
    ap.add_argument("--seed", type=int, default=1234, help="ALFWorld 的 env seed（默认 1234）")
    ap.add_argument("--ws-labels", default=None,
                    help="可选：WebShop {task_id: label} JSON，覆盖 post-hoc 从 reward_info 的恢复")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None, help="报告 JSON 输出路径（可选）")
    args = ap.parse_args()

    ws_labels = None
    if args.benchmark == "webshop" and args.ws_labels:
        ws_labels = {str(k): v for k, v in json.loads(Path(args.ws_labels).read_text(encoding="utf-8")).items()}

    report = asyncio.run(
        run_validation(
            benchmark=args.benchmark,
            trajectories_dir=Path(args.trajectories_dir),
            kev_base_url=args.kev_base_url,
            env_url=args.env_url,
            seed=args.seed,
            ws_labels=ws_labels,
            concurrency=args.concurrency,
        )
    )

    _print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
