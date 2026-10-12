#!/usr/bin/env python3
"""Secret-scan gate for §4 deliverables: thin wrapper over the shared scanner.

Patterns/whitelist/loop live in ``data/secret_scan.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from secret_scan import scan  # noqa: E402

if __name__ == "__main__":
    sys.exit(scan(Path(__file__).resolve().parent, {".md", ".json", ".jsonl", ".py"}, ("secret_scan.py",)))
