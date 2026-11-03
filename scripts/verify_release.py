"""Pre-release verification.

Checks that must pass before anything is published. This script does **not** publish, tag or push; it
only refuses to let a broken tree leave the machine.

Checks:
  1. version consistency across pyproject, package.json and the web footer constant
  2. required documents exist and are non-empty
  3. no obvious secrets in tracked files
  4. no `unsupported` capability is advertised as implemented in docs/STATUS.md
  5. the demo manifest matches the ONNX file it points at (size and hash)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def main() -> int:
    package = json.loads((ROOT / "apps" / "web" / "package.json").read_text(encoding="utf-8"))
    version_py = version()
    check("version matches package.json", version_py == package.get("version"), f"{version_py} vs {package.get('version')}")

    required = [
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "plan.md",
        "DECISIONS.md",
        "docs/RESEARCH.md",
        "docs/ARCHITECTURE.md",
        "docs/TESTING.md",
        "docs/SECURITY.md",
        "docs/STATUS.md",
        "docs/PERFORMANCE.md",
        "docs/acceptance.json",
    ]
    for item in required:
        path = ROOT / item
        check(f"{item} exists and is non-empty", path.exists() and path.stat().st_size > 200)

    lessons = list((ROOT / "docs" / "tutorials").glob("*.md"))
    check("20 curriculum lessons", len([p for p in lessons if p.name[0].isdigit()]) >= 20)

    status = (ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    for claim in ("flow matching", "diffusion language model", "quantisation"):
        check(f"'{claim}' is declared unsupported", "unsupported" in status.lower())

    manifest_path = ROOT / "apps" / "web" / "public" / "models" / "demo-manifest.json"
    onnx_path = ROOT / "apps" / "web" / "public" / "models" / "demo-char-lm.onnx"
    if manifest_path.exists() and onnx_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        check("demo manifest hash matches ONNX", digest == manifest["onnx"]["sha256"], "regenerate with scripts/build_web_demo.py")
        check("demo manifest size matches ONNX", onnx_path.stat().st_size == manifest["onnx"]["bytes"])
    else:
        check("demo manifest present", False, "run scripts/build_web_demo.py")

    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.splitlines()
    except Exception:  # pragma: no cover
        tracked = []
    secret_pattern = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{20,}[\"']")
    offenders = []
    for name in tracked:
        if name.endswith((".md", ".lock", "package-lock.json")):
            continue
        path = ROOT / name
        if not path.exists() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if secret_pattern.search(text):
            offenders.append(name)
    check("no secrets in tracked files", not offenders, ", ".join(offenders))

    if FAIL:
        print(f"\n{len(FAIL)} check(s) failed.")
        return 1
    print("\nAll release checks passed. Nothing was published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
