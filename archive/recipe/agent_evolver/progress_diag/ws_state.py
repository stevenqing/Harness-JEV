# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""WebShop 专属的 kev state 构造：补「选中商品详情」证据（方向 A）。

默认共享的 ``_render_state`` 只喂 ``goal`` + action 序列。WS 的 action 是
``click[ASIN]`` 这种不透明 ID，kev 判 type/attr/option/price 条件时看不到
「选中的商品是什么」，所以 WS 第一关的卡位准确率掉到常数基线以下。这里从
``.md`` 的 observation 里提取 agent 最终停留的商品详情页文本（title 含
颜色/尺码 + ``Price:``），补进 state，让 kev 能真的对比「选中 vs goal 要求」。

**无泄露红线**：只喂 agent 视角的观测（goal / action / 商品详情页），绝不喂
``reward_info`` 的 ``r_type``/``r_att``/``r_option``/``r_price`` 或
``category_match``/``query_match``/``title_score`` 等 oracle 结果字段，也不喂
``final_obs``（含 reward 分数）。
"""

from __future__ import annotations

import re
from pathlib import Path

from typing import Any

_OBS_RE = re.compile(r"\*\*Observation\*\*:\s*(.*)$", re.M)
# 详情页特征：搜索结果页是 "Page 1 (Total results)" + "$6.99"（无 "Price:" 标签），
# 详情页是 "Price: $6.99"（有 "Price:" 标签）。用它可靠区分两类 observation。
_DETAIL_MARKER = "Price:"
_MAX_ACTIONS = 900
_MAX_DETAIL = 600


def _extract_selected_product(text: str) -> str:
    """agent 最终停留的商品详情页 observation 文本（无则空串）。"""
    obs = _OBS_RE.findall(text)
    details = [o for o in obs if _DETAIL_MARKER in o]
    return details[-1] if details else ""


def ws_render_state(path: Path, fields: dict[str, Any]) -> str:
    """WS 轨迹 → kev state 文本，在共享字段之外多一行 ``selected product``。

    ``fields`` 是 ``decision_prior._parse_trajectory`` 的结果（非 ``None``）；
    ``path`` 用于读回 observation（``fields`` 不含 observation）。
    """
    acts = "; ".join(f"{a} [invalid]" if inv else a for a, inv in fields["pairs"])
    if len(acts) > _MAX_ACTIONS:
        acts = acts[:_MAX_ACTIONS] + "…"
    detail = _extract_selected_product(path.read_text(encoding="utf-8"))
    if len(detail) > _MAX_DETAIL:
        detail = detail[:_MAX_DETAIL] + "…"

    state = (
        f"goal: {fields['goal']}\n"
        f"exit_reason: {fields['exit_reason']}  eval_reason: {fields['eval_reason']}  steps: {fields['steps']}\n"
        f"actions: {acts}\n"
    )
    if detail:
        state += f"selected product: {detail}\n"
    return state
