"""Pre-release verification.

Checks that must pass before anything is published. This script does **not** publish, tag or push; it
only refuses to let a broken or self-contradicting tree leave the machine.

Checks:
  1. version consistency across pyproject.toml, hcs_foundry.__version__ and the web package
  2. CHANGELOG.md has an entry for the version being released
  3. required documents exist and are non-empty; 20 curriculum lessons present
  4. the brand credit appears in the README and in the shipped web app
  5. docs/acceptance.json parses and every area carries evidence or an explicit `unsupported`
  6. no obvious secrets in tracked files
  7. no U+FFFD replacement characters in tracked text files
  8. the demo manifest matches the ONNX file it points at (size and hash)
  9. the MCP dependency pin and the MCP import path agree (SDK v2)
 10. the built site exists, references no external origin, and ships its ONNX model

`--strict` turns "not built yet" from a skip into a failure, which is what a release needs.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT = "--strict" in sys.argv[1:]
FAIL: list[str] = []
SKIP: list[str] = []

BRAND = "Made with"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS  " if ok else "FAIL  ") + name + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)
    return ok


def skip(name: str, detail: str) -> None:
    print(f"SKIP  {name} — {detail}")
    SKIP.append(name)
    if STRICT:
        FAIL.append(f"{name} (skipped, -strict)")


def text_of(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def version() -> str:
    match = re.search(r'^version = "([^"]+)"', text_of("pyproject.toml"), re.MULTILINE)
    return match.group(1) if match else "unknown"


def tracked_files() -> list[str]:
    try:
        return subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.splitlines()
    except Exception:  # pragma: no cover - git absent
        return []


def check_versions(current: str) -> None:
    package = json.loads(text_of("apps/web/package.json"))
    check("version matches apps/web/package.json", current == package.get("version"), f"{current} vs {package.get('version')}")

    init = text_of("python/hcs_foundry/__init__.py")
    found = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    check("version matches hcs_foundry.__version__", bool(found) and found.group(1) == current, found.group(1) if found else "missing")

    changelog = text_of("CHANGELOG.md")
    check(f"CHANGELOG.md has a [{current}] section", f"## [{current}]" in changelog)


def check_documents() -> None:
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
        "docs/ROADMAP-0.2.0.md",
        "docs/acceptance.json",
    ]
    for item in required:
        path = ROOT / item
        check(f"{item} exists and is non-empty", path.exists() and path.stat().st_size > 200)

    lessons = [p for p in (ROOT / "docs" / "tutorials").glob("*.md") if p.name[0].isdigit()]
    check("20 curriculum lessons", len(lessons) >= 20, f"found {len(lessons)}")


def check_brand() -> None:
    check("brand credit in README", "hcsmedia and timfromhcs" in text_of("README.md"))
    app = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "apps" / "web" / "src").rglob("*")
        if path.suffix in {".tsx", ".ts"} and path.is_file()
    )
    # The credit lives in the i18n bundle, which is what the app actually renders.
    locales = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "apps" / "web" / "src" / "locales").glob("*.ts")
    )
    check("brand credit in web locales", "hcsmedia and timfromhcs" in locales, "add it to the footer strings")
    check("credit string is non-empty in app source", BRAND in locales or BRAND in app)


def check_acceptance() -> None:
    path = ROOT / "docs" / "acceptance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        check("docs/acceptance.json parses", False, str(exc))
        return
    areas = payload.get("areas", [])
    check("acceptance matrix has areas", len(areas) >= 10, f"found {len(areas)}")
    # A non-unsupported capability must justify itself: either with evidence, or with an explicit
    # limitation. Silence is what makes an acceptance matrix decorative.
    bad = [
        area.get("area", "?")
        for area in areas
        if area.get("status") not in {"pass", "declared", "unsupported"}
        or (
            area.get("status") != "unsupported"
            and not area.get("evidence")
            and not str(area.get("limitation", "")).strip()
        )
    ]
    check("every acceptance area has a valid status and evidence or limitation", not bad, ", ".join(bad))


def check_secrets() -> None:
    pattern = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{20,}[\"']")
    offenders = []
    for name in tracked_files():
        if name.endswith((".md", ".lock", "package-lock.json")):
            continue
        path = ROOT / name
        if not path.exists() or path.stat().st_size > 2_000_000:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(body):
            offenders.append(name)
    check("no secrets in tracked files", not offenders, ", ".join(offenders))


def check_encoding() -> None:
    offenders = []
    for name in tracked_files():
        if not name.endswith((".md", ".py", ".ts", ".tsx", ".json", ".css", ".html", ".toml", ".yml")):
            continue
        path = ROOT / name
        if not path.exists():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "\ufffd" in body:
            offenders.append(name)
    check("no U+FFFD replacement characters in tracked text", not offenders, ", ".join(offenders[:5]))


def check_demo_model() -> None:
    manifest_path = ROOT / "apps" / "web" / "public" / "models" / "demo-manifest.json"
    onnx_path = ROOT / "apps" / "web" / "public" / "models" / "demo-char-lm.onnx"
    if not (manifest_path.exists() and onnx_path.exists()):
        check("demo manifest present", False, "run scripts/build_web_demo.py")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    check("demo manifest hash matches ONNX", digest == manifest["onnx"]["sha256"], "regenerate with scripts/build_web_demo.py")
    check("demo manifest size matches ONNX", onnx_path.stat().st_size == manifest["onnx"]["bytes"])


def check_mcp_pin() -> None:
    pin = re.search(r'mcp\s*=\s*\[\s*"mcp>=([0-9]+)', text_of("pyproject.toml"))
    check("MCP extra pins a major version", bool(pin), "expected mcp = [\"mcp>=2..., <3\"]")
    if pin:
        check("MCP pin is v2", pin.group(1) == "2", f"pinned to v{pin.group(1)}")

    server = text_of("python/hcs_foundry/mcp_server.py")
    check("MCP server imports the v2 entry point", "from mcp.server import MCPServer" in server)
    # Comment-free check: the string may legitimately appear in a comment explaining the migration.
    imports_fastmcp = re.search(r"^\s*(from|import)\s+mcp\.server\.fastmcp", server, re.MULTILINE)
    check("MCP server no longer imports FastMCP", imports_fastmcp is None)


def check_built_site() -> None:
    dist = ROOT / "apps" / "web" / "dist"
    index = dist / "index.html"
    if not index.exists():
        skip("built site present", "run 'npm run build' in apps/web")
        return
    html = index.read_text(encoding="utf-8")
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    check("built index.html references no external origin", not external, ", ".join(external[:3]))
    assets = list((dist / "assets").glob("*.wasm"))
    check("built site ships a wasm runtime", bool(assets), "no .wasm in dist/assets")
    check("built site ships the demo model", (dist / "models" / "demo-char-lm.onnx").exists())


def main() -> int:
    current = version()
    print(f"release audit for {current}{' (strict)' if STRICT else ''}\n")
    check_versions(current)
    check_documents()
    check_brand()
    check_acceptance()
    check_secrets()
    check_encoding()
    check_demo_model()
    check_mcp_pin()
    check_built_site()

    if FAIL:
        print(f"\n{len(FAIL)} check(s) failed:")
        for name in FAIL:
            print(f"  - {name}")
        return 1
    if SKIP:
        print(f"\nAll blocking checks passed ({len(SKIP)} skipped). Nothing was published.")
    else:
        print("\nAll release checks passed. Nothing was published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
