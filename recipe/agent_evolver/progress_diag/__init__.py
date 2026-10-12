# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""进度诊断第一关（无 meta）：kev 从轨迹文本判 agent 走到哪一步，oracle 验真。

四个模块分工见同目录 ``align.md``：``conditions`` 建固定进度条件、``judge`` 让
kev 逐条判得卡位 k、``oracle`` 给真实卡位 m、``validate`` 算覆盖率/准确率/分布。
入口：``python -m recipe.agent_evolver.progress_diag.validate``。
"""
