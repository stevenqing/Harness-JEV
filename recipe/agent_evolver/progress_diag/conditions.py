# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""进度诊断第一关的固定进度条件（oracle 派生，**无 meta 调用**）。

每条条件是一个 ``JudgeQ``（noul）：kev 逐条判 ``p(满足)``，卡位 k = 第一个
``p < 0.5`` 的序号。条件与 oracle 里程碑落在同一序号空间（见 ``align.md`` §1）。

- ALFWorld：2 条，target / 要求动作从 ``alf_requirement``（expert plan）注入，
  所以条件是 **per-task 具体** 的（「拿到 mug」「把 mug 放进 cabinet」）。
- WebShop：6 条 benchmark 级固定，具体属性/选项/价格由 kev 从 state 里的
  ``goal`` 文本自己读（WebShop goal 串已是「color: mint, size: 7, …」结构）。
"""

from __future__ import annotations

from harnessx.decision import JudgeQ

from ..failure_labels import alf_requirement


def _req_phrase(req_verb: str, req_obj: str) -> str:
    """把 PDDL 要求动作转成 agent 文本里的动词短语（Toggle → use/toggle）。"""
    if req_verb in ("use", "toggle"):
        return f"use/toggle the {req_obj}"
    return f"{req_verb} the {req_obj}"


def alf_conditions(task: dict) -> list[JudgeQ]:
    """2 条 ALFWorld 进度条件（find → acquire）。

    ``alf_requirement`` 返回 ``(None, None, None)`` 时（plan 无 pickup）返回空，
    该 task 由 oracle 归 ``other``、验证时剔除。拿到即完（无要求动作）只有 1 条。
    """
    target, req_verb, req_obj = alf_requirement(task)
    if target is None:
        return []
    qs: list[JudgeQ] = [
        JudgeQ(
            f"Did the agent locate and take the {target} (the goal's target object)?",
            yes_desc=f"the agent issued a take action on the {target}",
            no_desc=f"the agent never took the {target}",
        ),
    ]
    if req_verb is not None:
        phrase = _req_phrase(req_verb, req_obj)
        qs.append(
            JudgeQ(
                f"After taking the {target}, did the agent complete the required final action ({phrase})?",
                yes_desc=f"the agent performed {phrase}",
                no_desc=f"the agent did not perform {phrase}",
            )
        )
    return qs


def ws_conditions() -> list[JudgeQ]:
    """6 条 WebShop 进度条件（搜→类型→属性→选项→价格→购买）。"""
    return [
        JudgeQ(
            "Did the agent issue at least one search[query] action?",
            yes_desc="the agent issued a search[query] action",
            no_desc="the agent never searched",
        ),
        JudgeQ(
            "Did the agent click into a product whose type/category matches the product type requested in the goal?",
            yes_desc="the agent clicked a matching-type product",
            no_desc="the agent clicked a wrong-type product, or no product at all",
        ),
        JudgeQ(
            "Did the agent's selected product match every required attribute named in the goal (color, size, material, ...)?",
            yes_desc="all required attributes matched",
            no_desc="some required attribute did not match",
        ),
        JudgeQ(
            "Did the agent's selected product match every required option named in the goal (specific variant choices), if the goal lists any?",
            yes_desc="all required options matched (or none required)",
            no_desc="some required option did not match",
        ),
        JudgeQ(
            "Is the agent's selected price within the goal's price upper bound, if the goal states one?",
            yes_desc="price within budget (or no budget stated)",
            no_desc="price exceeds the budget",
        ),
        JudgeQ(
            "Did the agent complete the purchase by clicking Buy Now?",
            yes_desc="the agent clicked Buy Now",
            no_desc="the agent never clicked Buy Now",
        ),
    ]


def conditions_for(benchmark: str, task: dict | None = None) -> list[JudgeQ]:
    """按 benchmark 返回条件列表。``task`` 仅 ALFWorld 用（注入 target）。"""
    if benchmark == "alfworld":
        return alf_conditions(task or {})
    if benchmark == "webshop":
        return ws_conditions()
    raise NotImplementedError(f"progress conditions not wired for {benchmark!r}")
