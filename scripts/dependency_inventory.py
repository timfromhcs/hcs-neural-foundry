"""Produce a dependency inventory (a deliberately simple SBOM).

Reads `pyproject.toml` and the web `package.json` / lockfile and prints a deterministic JSON summary:
direct dependencies, pinned versions where available, declared licences of the top-level packages.

This is not a full SPDX document. It exists so a reviewer can see, without installing anything, what
the project pulls in and under which licence it is distributed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def python_dependencies() -> dict:
    pyproject = ROOT / "pyproject.toml"
    if tomllib is None or not pyproject.exists():
        return {"error": "pyproject.toml or tomllib unavailable"}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return {
        "requires_python": project.get("requires-python"),
        "runtime": project.get("dependencies", []),
        "optional": project.get("optional-dependencies", {}),
        "license": (project.get("license") or {}).get("text"),
    }


def node_dependencies() -> dict:
    package = ROOT / "apps" / "web" / "package.json"
    if not package.exists():
        return {"error": "apps/web/package.json missing"}
    data = json.loads(package.read_text(encoding="utf-8"))
    lock = ROOT / "apps" / "web" / "package-lock.json"
    resolved: dict[str, str] = {}
    if lock.exists():
        locked = json.loads(lock.read_text(encoding="utf-8"))
        for name, meta in (locked.get("packages") or {}).items():
            if name.startswith("node_modules/") and "version" in meta:
                resolved[name.removeprefix("node_modules/")] = meta["version"]
    return {
        "runtime": data.get("dependencies", {}),
        "dev": data.get("devDependencies", {}),
        "resolved_packages": len(resolved),
        "sample_resolved": dict(sorted(resolved.items())[:25]),
    }


def main(argv: list[str] | None = None) -> int:
    payload = {
        "schema": "hcs-foundry.dependency-inventory/1",
        "python": python_dependencies(),
        "node": node_dependencies(),
        "notes": [
            "Direct dependencies only; transitive inventory comes from the lockfiles.",
            "No model weights or datasets are bundled; the only shipped artifact is the demo ONNX model.",
            "Licences of third-party models and datasets are recorded per artifact, not here.",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    args = list(sys.argv[1:] if argv is None else argv)
    if "--out" in args:
        target = Path(args[args.index("--out") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
