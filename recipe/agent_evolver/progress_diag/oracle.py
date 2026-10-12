# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""进度诊断第一关的 oracle：把已有程序化 ground-truth 转成「卡位序号 m」。

复用两个现成模块，不重写：

- ALFWorld：``failure_labels.alfworld_label`` → ``find``/``acquire``/``other``，
  ground truth 来自 ``traj_data.json`` 的 ``plan.high_pddl``（expert plan）。
- WebShop：``webshop_labels.ws_label`` → 六路标签，reward 分解在 buy 时由环境
  算好（``reward_info``）。

卡位序号 m = 第一个未达成的里程碑（1..M）。ALFWorld 的 ``other``（无 pickup /
拿到即完 / 已完成要求）**从验证集剔除**，返回 ``None``。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..decision_prior import _FIELD_RE, _parse_trajectory, _sort_key
from ..failure_labels import alfworld_label, gamefile_order, load_alf_task
from ..webshop_labels import ws_label

# WebShop 六路标签 → 进度卡位序号（见 webshop_oracle.md §2 的重排表）。
WS_MILESTONE: dict[str, int] = {
    "no_search": 1,
    "wrong_type": 2,
    "attr_mismatch": 3,
    "option_mismatch": 4,
    "price_over": 5,
    "not_bought": 6,
}


def resolve_alf_tasks(
    env_url: str, seed: int, trajectories_dir: Path
) -> dict[str, tuple[dict, list[str]]]:
    """``traj_id -> (task, actions)`` for every failed ALFWorld trajectory.

    Recovery mirrors ``failure_labels.gamefile_order``: ``traj_id`` (= game_idx)
    → the seed-ordered gamefile list → ``traj_data.json``.  ``actions`` is the
    extracted action strings (the oracle labels against them), kept here so
    ``validate`` does one parse per file.
    """
    order = gamefile_order(env_url, seed)
    out: dict[str, tuple[dict, list[str]]] = {}
    for p in sorted(trajectories_dir.glob("*.md"), key=_sort_key):
        fields = _parse_trajectory(p)
        if fields is None:
            continue
        try:
            task = load_alf_task(order[int(p.stem)])
        except (ValueError, IndexError, KeyError):
            continue
        out[p.stem] = (task, [a for a, _ in fields["pairs"]])
    return out


def alf_milestone(task: dict, actions: list[str]) -> int | None:
    """ALFWorld 卡位序号：``find`` → 1，``acquire`` → 2，``other`` → None（剔除）。"""
    return {"find": 1, "acquire": 2}.get(alfworld_label(actions, task))


def ws_milestone(label: str | None) -> int | None:
    """WebShop 卡位序号：六路标签 → 1..6；``None``（成功/无标签）→ None。"""
    return WS_MILESTONE.get(label or "")


def resolve_ws_tasks(trajectories_dir: Path) -> dict[str, str]:
    """``{traj_id: label}`` recovered post-hoc from ``.md`` (no live record).

    Reads the persisted ``reward_info`` frontmatter field (added by
    ``trajectories.render_frontmatter``) and the action sequence from the body,
    then replays ``webshop_labels.ws_label``.  Won trajectories are skipped by
    ``_parse_trajectory``.
    """
    out: dict[str, str] = {}
    for p in sorted(trajectories_dir.glob("*.md"), key=_sort_key):
        fields = _parse_trajectory(p)
        if fields is None:
            continue
        text = p.read_text(encoding="utf-8")
        fm = dict(_FIELD_RE.findall(text[: text.find("\n---\n")]))
        info_raw = fm.get("reward_info", "").strip()
        try:
            reward_info = json.loads(info_raw) if info_raw and info_raw != '""' else None
        except json.JSONDecodeError:
            reward_info = None
        label = ws_label({
            "success": False,
            "reward": 0.0,
            "reward_info": reward_info,
            "trajectory": [{"action": a} for a, _ in fields["pairs"]],
        })
        if label is not None:
            out[p.stem] = label
    return out
