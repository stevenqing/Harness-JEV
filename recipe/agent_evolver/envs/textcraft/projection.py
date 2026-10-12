"""Action parsing for TextCraft (TEXTCRAFT_SPEC.md §5).

Takes the last <action>…</action> block, then: strip whitespace, collapse runs
of whitespace to a single space, lowercase the leading verb (only get / craft /
inventory), strip trailing full stops. ``extract`` returns (raw_action, action);
an empty action means parse failure (no tag or empty content).
"""

from __future__ import annotations

import re

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_VERBS = ("get", "craft", "inventory")


def normalize(action: str) -> str:
    a = action.strip()
    a = re.sub(r"\s+", " ", a)
    m = re.match(r"([A-Za-z]+)", a)
    if m and m.group(1).lower() in _VERBS:
        a = m.group(1).lower() + a[m.end() :]
    return a.rstrip(".")


def extract(raw: str) -> tuple[str, str]:
    matches = _ACTION_RE.findall(raw)
    if not matches:
        return "", ""
    raw_action = matches[-1]
    return raw_action, normalize(raw_action)
