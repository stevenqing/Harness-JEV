# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Tiny shared text helpers (used across harnessx and recipe/agent_evolver)."""


def clip_text(text: str, n: int) -> str:
    """Collapse whitespace and truncate to ``n`` chars (with a trailing ellipsis)."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"
