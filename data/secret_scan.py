#!/usr/bin/env python3
"""Shared secret-scan gate (used by every ``data/`` deliverable directory).

Patterns + whitelist live here exactly once.  Each deliverable directory keeps a
thin wrapper (``data/<dir>/secret_scan.py``) that picks its own scan root /
extensions / name-exclusions and calls :func:`scan`.

Flags: API tokens/keys, AWS keys, passwords, non-localhost URLs / internal IPs,
hostnames, and raw 10.x / 192.168.x / 172.16-31.x addresses.  Known-safe
placeholders (``<redacted-local-url>``, 127.0.0.1 / localhost as the local vLLM
/ env-server loopback) are whitelisted.

Exit code 0 = clean, 1 = matches found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = {
    "openai-style-key": re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "token-assign": re.compile(r"(?i)\b(token|api[_-]?key|secret|password|passwd)\b\s*[:=]\s*\S+"),
    "internal-ip": re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "non-local-url": re.compile(r"https?://(?!127\.0\.0\.1|localhost)[^\s\"'<>]+"),
    "bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{8,}"),
}

WHITELIST = {"<redacted-local-url>", "http://127.0.0.1", "http://localhost"}


def scan(out: Path, exts: set[str], exclude_substrs: tuple[str, ...] = ()) -> int:
    """Scan every ``*<ext>`` file under ``out``; return 0 clean / 1 hits.

    ``exclude_substrs`` drops files whose *name* contains any of the given
    substrings (used to skip sibling step-0 artifacts and the scanner itself).
    """
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.suffix in exts
        and not any(s in p.name for s in exclude_substrs)
    )
    hits: list[tuple[str, str, str]] = []
    for p in files:
        text = p.read_bytes().decode("utf-8", errors="replace")
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                frag = m.group(0)
                if frag in WHITELIST:
                    continue
                hits.append((str(p), name, frag[:80]))
    if hits:
        print(f"SECRET SCAN: {len(hits)} potential hit(s)")
        for p, name, frag in hits:
            print(f"  {p}: {name}: {frag!r}")
        return 1
    print(f"SECRET SCAN clean ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: secret_scan.py <dir> <ext1,ext2,...> [exclude_substr,...]")
    out = Path(sys.argv[1])
    exts = set(sys.argv[2].split(","))
    exclude = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ()
    sys.exit(scan(out, exts, exclude))
