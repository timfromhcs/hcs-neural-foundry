"""Scan the repository for U+FFFD replacement characters introduced by an encoding mistake, and fix
the known ones by mapping them back to the intended punctuation."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = list(ROOT.glob("**/*.md")) + list(ROOT.glob("**/*.py")) + list(ROOT.glob("**/*.ts")) + list(
    ROOT.glob("**/*.tsx")
)
SKIP = {".venv", "node_modules", "dist", ".git", "artifacts"}

hits: list[tuple[pathlib.Path, int, str]] = []
for path in TARGETS:
    if any(part in SKIP for part in path.parts):
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for number, line in enumerate(text.splitlines(), start=1):
        if "\ufffd" in line:
            hits.append((path, number, line.strip()[:100]))

print(f"{len(hits)} line(s) with U+FFFD")
for path, number, line in hits:
    print(f"  {path.relative_to(ROOT)}:{number}: {line}")
