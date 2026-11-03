"""Canonical hashing and id helpers.

Hashes are the backbone of provenance: architecture hashes anchor weight patches, content hashes
anchor checkpoints, dataset hashes anchor benchmark results. Everything here must be stable across
processes and platforms, which is why we canonicalise JSON before hashing.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024


def utcnow() -> str:
    """ISO-8601 UTC timestamp with second resolution, used for every record."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, floats via repr."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def hash_json(payload: Any) -> str:
    """Hash a JSON-serialisable payload canonically."""
    return sha256_text(canonical_json(payload))


def hash_file(path: str | os.PathLike[str]) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(data: bytes) -> str:
    return sha256_hex(data)


def new_id(prefix: str) -> str:
    """Short, sortable-ish identifier. Not a security token."""
    return f"{prefix}_{secrets.token_hex(8)}"


def model_slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def short_hash(value: str, length: int = 12) -> str:
    return value[:length]


class atomic_write:
    """Write to a temp file in the destination directory, fsync, then rename into place.

    Used for every artifact so a crashed process can never leave a half-written checkpoint that
    looks valid to a later run.
    """

    def __init__(self, target: str | os.PathLike[str]):
        self.target = Path(target)
        self.tmp: Path | None = None
        self.handle = None

    def __enter__(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.target.parent, prefix=".tmp-", suffix=self.target.suffix)
        self.tmp = Path(tmp_name)
        self.handle = os.fdopen(fd, "wb")
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        assert self.handle is not None and self.tmp is not None
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        if exc_type is not None:
            self.tmp.unlink(missing_ok=True)
            return False
        os.replace(self.tmp, self.target)
        return False


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    with atomic_write(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
