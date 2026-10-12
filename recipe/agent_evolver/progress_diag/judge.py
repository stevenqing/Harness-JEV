# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""进度诊断第一关的 kev judge：逐轨迹一条 ``decide`` 判全部条件 → 卡位 k。

kev 是一次前向的 logit 读头（``KevBackend``），不生成。每轨迹调用一次
``decide(state, conditions)``，state 是轨迹的紧凑文本（goal + exit/eval +
action 序列，``_render_state``），conditions 是 ``conditions.py`` 给的 noul
``JudgeQ`` 列表。

卡位 k = 第一个 ``p(满足) < 0.5`` 的里程碑序号（1-index）；若全 ≥ 0.5 记
``M+1``（「看起来成功了」，但轨迹实际失败，验证时判为错配）。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

from harnessx.decision import JudgeQ, KevBackend

from ..decision_prior import _parse_trajectory, _render_state, _sort_key


async def judge_trajectories(
    *,
    kev_base_url: str,
    trajectories_dir: Path,
    condition_builder: Callable[[str], list[JudgeQ]],
    concurrency: int = 16,
) -> tuple[dict[str, dict], float]:
    """judge every failed trajectory → ``{traj_id: {"p": [..], "k": int}}``.

    ``condition_builder(traj_id)`` returns that trajectory's conditions (ALFWorld
    injects the target per task; WebShop returns the fixed six).  The kev input
    state is the shared ``_render_state`` (goal + 卡位上下文 + actions).  Returns
    the per-trajectory p-vector + 卡位 k, plus the judging wall-clock.
    """
    files = sorted(trajectories_dir.glob("*.md"), key=_sort_key)
    sem = asyncio.Semaphore(max(1, concurrency))
    results: dict[str, dict] = {}
    started = time.perf_counter()

    async with KevBackend(base_url=kev_base_url) as kev:

        async def one(path: Path) -> None:
            fields = _parse_trajectory(path)
            if fields is None:
                return
            questions = condition_builder(path.stem)
            if not questions:
                return
            state = _render_state(fields)
            async with sem:
                decisions = await kev.decide(state=state, questions=questions)
            p = [d.probabilities.get("true", 0.0) for d in decisions]
            k = next((i + 1 for i, pi in enumerate(p) if pi < 0.5), len(p) + 1)
            results[path.stem] = {"p": p, "k": k}

        await asyncio.gather(*(one(p) for p in files))

    return results, time.perf_counter() - started
