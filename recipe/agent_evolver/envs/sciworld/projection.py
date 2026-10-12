"""Action parsing for ScienceWorld (SCIWORLD_SPEC.md §6).

Take the text between the last pair of ``<action>...</action>`` tags, strip
leading/trailing whitespace, and collapse runs of whitespace to a single space.
No case conversion and no fuzzy matching — the text is handed to the environment
verbatim (the env's own ``No known action matches that input.`` handles unknown
actions).  No tag, or an empty tag, is a parse failure: ``extract`` returns
``("", "")`` and the runner records a fixed ``Invalid action format.`` step
without calling the environment.

RLVMR's projection also marks CJK-containing outputs invalid; we deliberately do
NOT (SPEC §6).
"""

from __future__ import annotations

import re

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def normalize(action: str) -> str:
    return re.sub(r"\s+", " ", action).strip()


def extract(raw: str) -> tuple[str, str]:
    """Return ``(raw_action, action)``.  ``action == ""`` means parse failure."""
    matches = _ACTION_RE.findall(raw)
    if not matches:
        return "", ""
    raw_action = matches[-1]
    return raw_action, normalize(raw_action)
