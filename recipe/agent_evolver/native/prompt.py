# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Inline system-prompt builder.

The evolvable system-prompt text lives **inline in the config YAML** (the
``prompt:`` field of ``system_builder``), so the meta-agent edits the YAML
directly and the config diff captures the prompt change.  (The meta-agent's
write scope is limited to its evolve ``output_dir`` + ``memo_path``, so a
shared on-disk ``.j2`` template would not be editable.)

NOTE: loaded via ``file://…::StaticSystemPromptBuilder`` (synthetic module
name), so no relative imports here.
"""
from __future__ import annotations


class StaticSystemPromptBuilder:
    """Return a fixed, inline system prompt."""

    def __init__(self, prompt: str = "") -> None:
        self.prompt = prompt

    async def build(self, workspace=None) -> str:  # noqa: ANN001
        return self.prompt
