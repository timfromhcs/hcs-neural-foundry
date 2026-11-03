"""Workspace layout, SQLite metadata store and repositories.

Two storage tiers, on purpose (ADR-006):
* SQLite holds metadata: models, checkpoints, patches, runs, metrics, evaluations, tool calls.
* The artifact directory holds immutable blobs named by content hash.

The store never deletes a checkpoint that is referenced by a run or a patch; retention is explicit.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError
from .hashing import new_id, utcnow
from .schemas import (
    CheckpointRecord,
    DatasetSpec,
    EvalReport,
    ModelDefinition,
    ModelSummary,
    RunRecord,
    RunState,
    WeightPatch,
)

SCHEMA_VERSION = 5

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            family TEXT NOT NULL,
            config_json TEXT NOT NULL,
            architecture_hash TEXT NOT NULL,
            parameter_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            maturity TEXT NOT NULL DEFAULT 'stable',
            source TEXT NOT NULL DEFAULT 'created'
        );
        CREATE INDEX IF NOT EXISTS idx_models_hash ON models(architecture_hash);

        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
            sha256 TEXT NOT NULL,
            path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT 'checkpoint',
            parent_sha256 TEXT,
            origin TEXT NOT NULL DEFAULT 'init'
        );
        CREATE INDEX IF NOT EXISTS idx_checkpoints_model ON checkpoints(model_id, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_checkpoints_sha ON checkpoints(sha256);

        CREATE TABLE IF NOT EXISTS patches (
            patch_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
            checkpoint_id TEXT,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'proposed',
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_patches_model ON patches(model_id, created_at);

        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            model_id TEXT NOT NULL,
            state TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind, started_at);

        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (run_id, step)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            model_id TEXT NOT NULL,
            suite_id TEXT NOT NULL,
            state TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_id, created_at);

        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tool_invocations (
            id TEXT PRIMARY KEY,
            tool TEXT NOT NULL,
            args_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            approved INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            scope TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS proposals (
            proposal_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            approval TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        3,
        """
        ALTER TABLE checkpoints ADD COLUMN architecture_hash TEXT;
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS evolution_archives (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        5,
        """
        CREATE TABLE IF NOT EXISTS model_heads (
            model_id TEXT PRIMARY KEY,
            checkpoint_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
]


def default_home() -> Path:
    env = os.environ.get("HCS_FOUNDRY_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".hcs-foundry").resolve()


class Workspace:
    """Directory layout for one workspace instance."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root).expanduser().resolve() if root else default_home()
        self.db_path = self.root / "foundry.db"
        self.artifacts = self.root / "artifacts"
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.exports = self.root / "exports"
        self.datasets = self.root / "datasets"
        for directory in (self.root, self.artifacts, self.checkpoints, self.logs, self.exports, self.datasets):
            directory.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, sha256: str, suffix: str = "") -> Path:
        return self.artifacts / sha256[:2] / f"{sha256}{suffix}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace({self.root})"


class Database:
    """Thin SQLite wrapper with forward-only migrations.

    One connection, guarded by a re-entrant lock: the application is single-process by design and
    the HTTP API runs handlers in a thread pool.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(workspace.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # -- lifecycle ---------------------------------------------------------------------------
    def migrate(self) -> int:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            current = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
            for version, script in MIGRATIONS:
                if version <= current:
                    continue
                try:
                    self._conn.executescript(script)
                    self._conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, utcnow()),
                    )
                except sqlite3.Error as exc:  # pragma: no cover - corrupt db path
                    raise ConflictError(
                        f"migration {version} failed: {exc}",
                        hint=f"database: {self.workspace.db_path}",
                    ) from exc
            return self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]

    @property
    def version(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchone()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))


class Repository:
    """Entity-level persistence built on :class:`Database`."""

    def __init__(self, workspace: Workspace | None = None, db: Database | None = None):
        self.workspace = workspace or (db.workspace if db else Workspace())
        self.db = db or Database(self.workspace)

    # -- models ------------------------------------------------------------------------------
    def add_model(self, model: ModelDefinition) -> ModelDefinition:
        self.db.execute(
            "INSERT INTO models (id,name,family,config_json,architecture_hash,parameter_count,created_at,notes,maturity,source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                model.id,
                model.name,
                model.config.family,
                model.config.model_dump_json(),
                model.architecture_hash,
                model.parameter_count,
                model.created_at,
                model.notes,
                model.maturity.value,
                model.source,
            ),
        )
        return model

    def get_model(self, model_id: str) -> ModelDefinition:
        row = self.db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            raise NotFoundError(
                f"model '{model_id}' not found",
                hint="list models with: hcs-foundry model list",
            )
        return ModelDefinition(
            id=row["id"],
            name=row["name"],
            config=json.loads(row["config_json"]),
            architecture_hash=row["architecture_hash"],
            parameter_count=row["parameter_count"],
            created_at=row["created_at"],
            notes=row["notes"],
            maturity=row["maturity"],
            source=row["source"],
        )

    def list_models(self) -> list[ModelSummary]:
        rows = self.db.query(
            """
            SELECT m.id, m.name, m.family, m.parameter_count, m.architecture_hash, m.created_at,
                   (SELECT COUNT(*) FROM checkpoints c WHERE c.model_id = m.id) AS checkpoints,
                   (SELECT MAX(json_extract(e.report_json, '$.metrics[0].value'))
                      FROM evaluations e WHERE e.model_id = m.id AND e.state = 'completed') AS best_score
            FROM models m ORDER BY m.created_at DESC
            """
        )
        return [
            ModelSummary(
                id=row["id"],
                name=row["name"],
                family=row["family"],
                parameter_count=row["parameter_count"],
                architecture_hash=row["architecture_hash"],
                created_at=row["created_at"],
                checkpoints=row["checkpoints"],
                best_score=row["best_score"],
            )
            for row in rows
        ]

    def delete_model(self, model_id: str) -> None:
        self.db.execute("DELETE FROM models WHERE id = ?", (model_id,))

    # -- checkpoints -------------------------------------------------------------------------
    def add_checkpoint(self, record: CheckpointRecord, architecture_hash: str | None = None) -> CheckpointRecord:
        existing = self.db.query_one("SELECT id FROM checkpoints WHERE sha256 = ?", (record.sha256,))
        if existing:
            record = self.get_checkpoint(existing["id"])
            self.set_head(record.model_id, record.id)
            return record
        self.db.execute(
            "INSERT INTO checkpoints (id,model_id,sha256,path,bytes,created_at,label,parent_sha256,origin,architecture_hash)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.model_id,
                record.sha256,
                record.path,
                record.bytes,
                record.created_at,
                record.label,
                record.parent_sha256,
                record.origin,
                architecture_hash,
            ),
        )
        self.set_head(record.model_id, record.id)
        return record

    # -- head pointer -----------------------------------------------------------------------
    def set_head(self, model_id: str, checkpoint_id: str) -> None:
        """Rollback and resume both work by moving this pointer; blobs stay immutable."""
        self.db.execute(
            "INSERT OR REPLACE INTO model_heads (model_id, checkpoint_id, updated_at) VALUES (?,?,?)",
            (model_id, checkpoint_id, utcnow()),
        )

    def get_head(self, model_id: str) -> CheckpointRecord | None:
        row = self.db.query_one("SELECT checkpoint_id FROM model_heads WHERE model_id = ?", (model_id,))
        return self.get_checkpoint(row["checkpoint_id"]) if row else None

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        row = self.db.query_one("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
        if row is None:
            raise NotFoundError(f"checkpoint '{checkpoint_id}' not found")
        return CheckpointRecord(
            id=row["id"],
            model_id=row["model_id"],
            sha256=row["sha256"],
            path=row["path"],
            bytes=row["bytes"],
            created_at=row["created_at"],
            label=row["label"],
            parent_sha256=row["parent_sha256"],
            origin=row["origin"],
        )

    def find_checkpoint_by_sha(self, sha256: str) -> CheckpointRecord | None:
        row = self.db.query_one("SELECT * FROM checkpoints WHERE sha256 = ?", (sha256,))
        return self.get_checkpoint(row["id"]) if row else None

    def latest_checkpoint(self, model_id: str) -> CheckpointRecord | None:
        head = self.get_head(model_id)
        if head is not None:
            return head
        row = self.db.query_one(
            "SELECT id FROM checkpoints WHERE model_id = ? ORDER BY rowid DESC LIMIT 1",
            (model_id,),
        )
        return self.get_checkpoint(row["id"]) if row else None

    def list_checkpoints(self, model_id: str) -> list[CheckpointRecord]:
        rows = self.db.query(
            "SELECT id FROM checkpoints WHERE model_id = ? ORDER BY created_at ASC, rowid ASC", (model_id,)
        )
        return [self.get_checkpoint(row["id"]) for row in rows]

    # -- patches -----------------------------------------------------------------------------
    def add_patch(self, model_id: str, patch: WeightPatch, checkpoint_id: str | None = None) -> None:
        if self.db.query_one("SELECT 1 FROM patches WHERE patch_id = ?", (patch.patch_id,)):
            raise ConflictError(
                f"patch '{patch.patch_id}' is already recorded",
                hint="patches are immutable; create a new patch id for a new edit",
            )
        self.db.execute(
            "INSERT INTO patches (patch_id,model_id,checkpoint_id,payload_json,status,created_by,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                patch.patch_id,
                model_id,
                checkpoint_id,
                patch.model_dump_json(),
                patch.status,
                patch.created_by,
                patch.created_at,
            ),
        )

    def update_patch_status(self, patch_id: str, status: str, checkpoint_id: str | None = None) -> None:
        self.db.execute(
            "UPDATE patches SET status = ?, checkpoint_id = COALESCE(?, checkpoint_id) WHERE patch_id = ?",
            (status, checkpoint_id, patch_id),
        )

    def list_patches(self, model_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT payload_json, status FROM patches WHERE model_id = ? ORDER BY created_at DESC", (model_id,)
        )
        out = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["status"] = row["status"]
            out.append(payload)
        return out

    def get_patch(self, patch_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT payload_json, status FROM patches WHERE patch_id = ?", (patch_id,))
        if row is None:
            raise NotFoundError(f"patch '{patch_id}' not found")
        payload = json.loads(row["payload_json"])
        payload["status"] = row["status"]
        return payload

    # -- runs --------------------------------------------------------------------------------
    def add_run(self, run: RunRecord) -> RunRecord:
        self.db.execute(
            "INSERT INTO runs (id,kind,model_id,state,config_json,summary_json,provenance_json,started_at,finished_at,error_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET state=excluded.state, config_json=excluded.config_json,"
            " summary_json=excluded.summary_json, provenance_json=excluded.provenance_json",
            (
                run.id,
                run.kind,
                run.model_id,
                run.state.value,
                json.dumps(run.config, default=str),
                json.dumps(run.summary, default=str),
                run.provenance.model_dump_json(),
                run.started_at,
                run.finished_at,
                json.dumps(run.error) if run.error else None,
            ),
        )
        return run

    def update_run(
        self,
        run_id: str,
        *,
        state: RunState | None = None,
        summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if state is not None:
            sets.append("state = ?")
            params.append(state.value)
        if summary is not None:
            sets.append("summary_json = ?")
            params.append(json.dumps(summary, default=str))
        if error is not None:
            sets.append("error_json = ?")
            params.append(json.dumps(error, default=str))
        if finished:
            sets.append("finished_at = ?")
            params.append(utcnow())
        if not sets:
            return
        params.append(run_id)
        self.db.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", params)

    def add_metric(self, run_id: str, step: int, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO run_metrics (run_id, step, payload_json) VALUES (?,?,?)",
            (run_id, step, json.dumps(payload, default=str)),
        )

    def get_run(self, run_id: str) -> RunRecord:
        row = self.db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if row is None:
            raise NotFoundError(f"run '{run_id}' not found")
        metrics = [
            json.loads(r["payload_json"])
            for r in self.db.query("SELECT payload_json FROM run_metrics WHERE run_id = ? ORDER BY step", (run_id,))
        ]
        from .schemas import MetricPoint, Provenance

        return RunRecord(
            id=row["id"],
            kind=row["kind"],
            model_id=row["model_id"],
            state=RunState(row["state"]),
            config=json.loads(row["config_json"]),
            metrics=[MetricPoint(**m) for m in metrics],
            summary=json.loads(row["summary_json"]),
            provenance=Provenance.model_validate_json(row["provenance_json"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=json.loads(row["error_json"]) if row["error_json"] else None,
        )

    def list_runs(self, *, model_id: str | None = None, kind: str | None = None, limit: int = 50) -> list[RunRecord]:
        sql = "SELECT id FROM runs"
        clauses: list[str] = []
        params: list[Any] = []
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        return [self.get_run(row["id"]) for row in self.db.query(sql, params)]

    # -- evaluations -------------------------------------------------------------------------
    def add_evaluation(self, run_id: str | None, report: EvalReport) -> EvalReport:
        self.db.execute(
            "INSERT INTO evaluations (id,run_id,model_id,suite_id,state,report_json,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                report.run_id,
                run_id,
                report.model_id,
                report.suite_id,
                report.state.value,
                report.model_dump_json(),
                report.created_at,
            ),
        )
        return report

    def get_evaluation(self, evaluation_id: str) -> EvalReport:
        row = self.db.query_one("SELECT report_json FROM evaluations WHERE id = ?", (evaluation_id,))
        if row is None:
            raise NotFoundError(f"evaluation '{evaluation_id}' not found")
        return EvalReport.model_validate_json(row["report_json"])

    def list_evaluations(self, model_id: str, limit: int = 50) -> list[EvalReport]:
        rows = self.db.query(
            "SELECT report_json FROM evaluations WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
            (model_id, limit),
        )
        return [EvalReport.model_validate_json(row["report_json"]) for row in rows]

    # -- datasets ----------------------------------------------------------------------------
    def upsert_dataset(self, spec: DatasetSpec) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO datasets (id, payload_json, created_at) VALUES (?,?,?)",
            (spec.id, spec.model_dump_json(), utcnow()),
        )

    def get_dataset(self, dataset_id: str) -> DatasetSpec:
        row = self.db.query_one("SELECT payload_json FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise NotFoundError(f"dataset '{dataset_id}' not found", hint="run: hcs-foundry dataset list")
        return DatasetSpec.model_validate_json(row["payload_json"])

    def list_datasets(self) -> list[DatasetSpec]:
        rows = self.db.query("SELECT payload_json FROM datasets ORDER BY id")
        return [DatasetSpec.model_validate_json(row["payload_json"]) for row in rows]

    # -- proposal / timeline -----------------------------------------------------------------
    def add_proposal(self, model_id: str, payload: dict[str, Any], approval: str = "pending") -> None:
        self.db.execute(
            "INSERT INTO proposals (proposal_id, model_id, payload_json, approval, created_at) VALUES (?,?,?,?,?)",
            (payload["proposal_id"], model_id, json.dumps(payload), approval, utcnow()),
        )

    def update_proposal(self, proposal_id: str, *, approval: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE proposals SET approval = ?, payload_json = ? WHERE proposal_id = ?",
            (approval, json.dumps(payload), proposal_id),
        )

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT payload_json FROM proposals WHERE proposal_id = ?", (proposal_id,))
        if row is None:
            raise NotFoundError(f"proposal '{proposal_id}' not found")
        return json.loads(row["payload_json"])

    def list_proposals(self, model_id: str | None = None) -> list[dict[str, Any]]:
        if model_id:
            rows = self.db.query(
                "SELECT payload_json FROM proposals WHERE model_id = ? ORDER BY created_at DESC", (model_id,)
            )
        else:
            rows = self.db.query("SELECT payload_json FROM proposals ORDER BY created_at DESC")
        return [json.loads(row["payload_json"]) for row in rows]

    def add_archive(self, model_id: str, payload: dict[str, Any]) -> str:
        archive_id = new_id("arc")
        self.db.execute(
            "INSERT INTO evolution_archives (id,model_id,payload_json,created_at) VALUES (?,?,?,?)",
            (archive_id, model_id, json.dumps(payload, default=str), utcnow()),
        )
        return archive_id

    def list_archives(self, model_id: str | None = None) -> list[dict[str, Any]]:
        if model_id:
            rows = self.db.query(
                "SELECT id, payload_json FROM evolution_archives WHERE model_id = ? ORDER BY created_at DESC",
                (model_id,),
            )
        else:
            rows = self.db.query("SELECT id, payload_json FROM evolution_archives ORDER BY created_at DESC")
        return [{"id": row["id"], **json.loads(row["payload_json"])} for row in rows]

    # -- events / tool calls / settings -------------------------------------------------------
    def log_event(self, level: str, scope: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "INSERT INTO events (id,ts,level,scope,message,payload_json) VALUES (?,?,?,?,?,?)",
            (new_id("evt"), utcnow(), level, scope, message, json.dumps(payload or {}, default=str)),
        )

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM events ORDER BY ts DESC, rowid DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def log_tool_call(
        self, tool: str, args: dict[str, Any], result: dict[str, Any], *, approved: bool = False
    ) -> str:
        call_id = new_id("tool")
        self.db.execute(
            "INSERT INTO tool_invocations (id,tool,args_json,result_json,approved,created_at) VALUES (?,?,?,?,?,?)",
            (call_id, tool, json.dumps(args, default=str), json.dumps(result, default=str), int(approved), utcnow()),
        )
        return call_id

    def list_tool_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM tool_invocations ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value_json FROM settings WHERE key = ?", (key,))
        return json.loads(row["value_json"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO settings (key, value_json, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value, default=str), utcnow()),
        )

    def close(self) -> None:
        self.db.close()
