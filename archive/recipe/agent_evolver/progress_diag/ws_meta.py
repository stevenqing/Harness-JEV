# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""WebShop 第一关的 meta 出题器：把 goal 实例化成 6 条具体进度条件。

kev（4B 单次前向判别头）能判「动作存在性」（search / buy）和「值匹配」，但
没能力从自由文本 goal 里解析出 ``type / attribute / option / price`` 的结构化
要求——这正是 WS 第一关卡位准确率掉到常数基线以下的根因（ALF 能 GO 是因为
``alf_conditions`` 早被 expert plan 实例化成「take the mug」这种具体动作）。

这里让 meta（System-2）**只做「出题」**：读 goal，把 WebShop reward 的 4 个
分量（type/attr/option/price）实例化成 kev 能对着 ``selected product`` 详情页
直接判的具体 ``JudgeQ``；``search`` / ``buy`` 两条是动作存在性，固定即可。

分工与第一关设计一致：meta 出题（受控实例化，非自由分解），kev 判分。复用
``decision_prior.induce_lever_predicates`` 的 JSON 诱导模式。

**无泄露红线**（同 ``ws_state.py``）：meta 只读 agent-visible 的 goal 文本，
绝不读 ``reward_info`` / ``final_obs``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from harnessx.core.events import Message
from harnessx.decision import JudgeQ

from ..decision_prior import _parse_trajectory, _sort_key
from ..run import DEFAULT_META_API_BASE, DEFAULT_META_MODEL, _make_provider

_log = logging.getLogger(__name__)

# 两条动作存在性条件：不依赖 goal，固定即可（milestone 1 / 6）。
_SEARCH_Q = JudgeQ(
    "Did the agent issue at least one search[query] action?",
    yes_desc="the agent issued a search[query] action",
    no_desc="the agent never searched",
)
_BUY_Q = JudgeQ(
    "Did the agent complete the purchase by clicking Buy Now?",
    yes_desc="the agent clicked Buy Now",
    no_desc="the agent never clicked Buy Now",
)

_META_SYSTEM = (
    "You are parsing a WebShop shopping-goal instruction into four concrete yes/no "
    "progress questions that map to WebShop's reward components. You output ONLY "
    "JSON; you never see the agent's trajectory, the reward, or any oracle signal."
)

# 4 个 reward 分量 → 具体条件，顺序对齐 milestone 2..5（wrong_type/attr_mismatch/
# option_mismatch/price_over）。关键提醒：type 是产品类别（"nail polish" vs
# "nail-art powder set"，同含 'glitter' 词不算 type 匹配）；option 是可选变体
# （颜色号/尺码）；attribute 是产品的固有属性。
_META_TEMPLATE = """Parse this WebShop shopping-goal instruction into its requirements, then write 4 yes/no progress questions a judge will answer by reading the agent's FINAL selected product detail page (its title and price line).

The 4 questions map to WebShop's reward components in this EXACT order and must keep these meanings:

 q2 — product TYPE: does the selected product's category/type match what the customer asked to buy? (Same filler words like 'glitter' do NOT make the type match — distinguish "nail polish" from "nail-art powder set".)
 q3 — product ATTRIBUTES: does the selected product match the goal's stated attributes — the inherent product properties (material, color family, pattern), if the goal names any?
 q4 — product OPTIONS: does the selected product match the goal's specific option/variant choices — a selectable variant like a color number "04#" or a size, if the goal names any?
 q5 — PRICE: is the selected product's price within the goal's price upper bound, if the goal states one?

For each, "yes" must be anchored to something literally checkable in the product title / price line; "no" is its negation. If the goal does not name a component, phrase the question so "yes" means "nothing required".

Goal instruction:
{goal}

Return ONLY a JSON object mapping q2/q3/q4/q5 to {{"q": "<yes/no question>", "yes": "<what yes means>", "no": "<what no means>"}}. No prose, no markdown fences."""


def _extract_json(text: str | None) -> dict | None:
    """Pull the first JSON object out of a generative reply (fence-tolerant)."""
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def _induce_one(provider: Any, goal: str) -> list[JudgeQ]:
    """One meta call → 6 concrete WS conditions (q1/q6 fixed + q2..q5 induced)."""
    prompt = _META_TEMPLATE.format(goal=goal)
    for attempt in range(4):
        try:
            event = await provider.complete(
                messages=[
                    Message(role="system", content=_META_SYSTEM),
                    Message(role="user", content=prompt),
                ],
                tools=[],
            )
            obj = _extract_json(event.content) or _extract_json(event.thinking)
            if obj:
                qs: list[JudgeQ] = []
                for key in ("q2", "q3", "q4", "q5"):
                    v = obj.get(key)
                    if not isinstance(v, dict) or not all(kk in v for kk in ("q", "yes", "no")):
                        raise ValueError(f"malformed condition {key!r}: {v!r}")
                    qs.append(
                        JudgeQ(
                            instructions=str(v["q"]),
                            yes_desc=str(v["yes"]),
                            no_desc=str(v["no"]),
                        )
                    )
                return [_SEARCH_Q, *qs, _BUY_Q]
        except Exception as exc:  # noqa: BLE001 — retry transient gateway faults / malformed JSON
            _log.warning("ws meta condition induction attempt %d failed: %s", attempt + 1, exc)
        await asyncio.sleep(8 * (attempt + 1))
    raise RuntimeError("meta failed to induce WS conditions after 4 attempts")


async def build_ws_conditions_map(
    provider: Any,
    trajectories_dir: Path,
    concurrency: int = 8,
) -> dict[str, list[JudgeQ]]:
    """``{traj_id: [6 concrete JudgeQ]}`` — meta 出题（每失败轨迹一次），kev 待判。

    goal 从 ``.md`` 前端读回（agent-visible），失败轨迹由 ``_parse_trajectory``
    确定（won 跳过）。
    """
    goals: dict[str, str] = {}
    for p in sorted(trajectories_dir.glob("*.md"), key=_sort_key):
        fields = _parse_trajectory(p)
        if fields is None:
            continue
        goals[p.stem] = fields["goal"]

    sem = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, list[JudgeQ]] = {}

    async def one(tid: str, goal: str) -> None:
        async with sem:
            out[tid] = await _induce_one(provider, goal)

    await asyncio.gather(*(one(t, g) for t, g in goals.items()))
    return out


__all__ = [
    "DEFAULT_META_API_BASE",
    "DEFAULT_META_MODEL",
    "build_ws_conditions_map",
    "_make_provider",
]
