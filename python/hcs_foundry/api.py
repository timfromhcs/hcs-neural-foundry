"""Local HTTP API.

Hardening (AGENTS.md §8.1): loopback by default, bearer token required for everything except health
and OpenAPI, explicit CORS allowlist, rate-limited job creation, typed error payloads, no shell and no
arbitrary Python endpoints. A mutating request without the token is rejected before any work happens.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from .errors import FoundryError, NotFoundError, SecurityViolation, ValidationFailure
from .schemas import (
    EvolutionConfig,
    GeneratorConfig,
    PatchOperation,
    RunRecord,
    RunState,
    TrainingConfig,
    WeightPatch,
)
from .services import Foundry
from .training.trainer import TrainerControl

DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173")


def resolve_token(home: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HCS_FOUNDRY_API_TOKEN")
    if env:
        return env
    token_file = (home or Path.home() / ".hcs-foundry") / "api-token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    token = secrets.token_urlsafe(32)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(token, encoding="utf-8")
    with contextlib.suppress(OSError):  # best effort on Windows/POSIX
        os.chmod(token_file, 0o600)
    return token


class RateLimiter:
    def __init__(self, limit: int = 30, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                raise SecurityViolation(
                    "rate limit exceeded for this endpoint",
                    hint=f"max {self.limit} requests per {int(self.window)}s",
                )
            bucket.append(now)


class JobManager:
    """Owns background training threads and their cooperative controls."""

    def __init__(self) -> None:
        self.controls: dict[str, TrainerControl] = {}
        self.threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, run_id: str, target) -> None:  # noqa: ANN001
        control = TrainerControl()
        thread = threading.Thread(target=target, name=f"train-{run_id}", daemon=True)
        with self._lock:
            self.controls[run_id] = control
            self.threads[run_id] = thread
        thread.start()

    def control(self, run_id: str) -> TrainerControl:
        with self._lock:
            control = self.controls.get(run_id)
        if control is None:
            raise NotFoundError(f"no active job for run '{run_id}'")
        return control


# --------------------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------------------


class CreateModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    config: dict[str, Any]
    notes: str = ""


class SolveRequest(BaseModel):
    target_parameters: int = Field(..., ge=1)
    policy: str = "closest"
    family: str = "transformer-lm"
    constraints: dict[str, Any] | None = None
    top_k: int = Field(5, ge=1, le=20)


class PatchValidateRequest(BaseModel):
    model_id: str
    patch: dict[str, Any]


class PatchCreateRequest(BaseModel):
    model_id: str
    tensor: str
    operation: PatchOperation
    parameters: dict[str, Any]
    base_checkpoint: str = "latest"
    created_by: str = "user"
    note: str = ""


class TrainRequest(BaseModel):
    model_id: str
    dataset_id: str = "auto"
    steps: int = Field(200, ge=1, le=200_000)
    batch_size: int = Field(16, ge=1, le=4096)
    block_size: int = Field(32, ge=1, le=8192)
    lr: float = Field(3e-3, gt=0, le=10)
    seed: int = Field(0, ge=0)
    eval_every: int = 50
    checkpoint_every: int = 100
    max_seconds: int = 600


class EvalRequest(BaseModel):
    model_id: str
    suite: str = "auto"
    checkpoint: str | None = None
    seed: int = 1234
    batches: int = 4


class ExportRequest(BaseModel):
    model_id: str
    format: str = "onnx"
    checkpoint: str = "latest"


class ProposalRequest(BaseModel):
    model_id: str
    agent_id: str = "agent:http"
    goal: str
    hypothesis: str = ""
    tensor: str
    operation: PatchOperation
    parameters: dict[str, Any]
    base_checkpoint: str = "latest"


class EvolutionRequest(BaseModel):
    model_id: str
    population: int = Field(16, ge=2, le=512)
    generations: int = Field(5, ge=1, le=200)
    sigma: float = Field(0.02, gt=0)
    lr: float = Field(0.05, gt=0)
    seed: int = 0


class GenerateRequest(BaseModel):
    target_model_id: str
    corpus: list[int] = Field(default_factory=lambda: [11, 22, 33, 44, 55, 66, 77, 88])
    epochs: int = 200
    latent_dim: int = 8
    seed: int = 0


# --------------------------------------------------------------------------------------------------
# Application factory
# --------------------------------------------------------------------------------------------------


def create_app(
    foundry: Foundry | None = None,
    *,
    token: str | None = None,
    cors_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    foundry = foundry or Foundry()
    api_token = resolve_token(foundry.workspace.root, token)
    origins = cors_origins or tuple(
        origin.strip()
        for origin in os.environ.get("HCS_FOUNDRY_CORS_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
        if origin.strip()
    )
    jobs = JobManager()
    limiter = RateLimiter()

    app = FastAPI(
        title="HCS Neural Foundry API",
        version="0.1.0",
        description=(
            "Local-first API. All compute happens on this machine. Weight edits are immutable patches; "
            "evaluations are recorded with dataset hashes and evaluator versions."
        ),
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.state.foundry = foundry
    app.state.jobs = jobs
    app.state.token = api_token

    @app.exception_handler(FoundryError)
    async def foundry_error_handler(request: Request, exc: FoundryError):  # noqa: ARG001
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    @app.exception_handler(ValidationError)
    async def pydantic_error_handler(request: Request, exc: ValidationError):  # noqa: ARG001
        errors = exc.errors(include_url=False)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_failed",
                    "message": errors[0]["msg"] if errors else "invalid payload",
                    "hint": "check the field paths in details",
                    "details": {"errors": errors},
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):  # noqa: ARG001
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_failed",
                    "message": "request body failed validation",
                    "details": {"errors": json.loads(json.dumps(exc.errors(), default=str))},
                }
            },
        )

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "missing bearer token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        provided = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(provided, api_token):
            raise HTTPException(
                status_code=403, detail={"code": "forbidden", "message": "invalid token"}
            )

    # -- read-only -------------------------------------------------------------------------
    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "workspace": str(foundry.workspace.root),
            "schema_version": foundry.repo.db.version,
            "auth_required": True,
        }

    @app.get("/api/v1/hardware", dependencies=[Depends(auth)])
    def hardware() -> dict[str, Any]:
        return foundry.hardware.model_dump(mode="json")

    @app.get("/api/v1/backends", dependencies=[Depends(auth)])
    def backends() -> list[dict[str, Any]]:
        return [backend.model_dump(mode="json") for backend in foundry.hardware.backends]

    @app.get("/api/v1/summary", dependencies=[Depends(auth)])
    def summary() -> dict[str, Any]:
        return foundry.summary()

    @app.get("/api/v1/models", dependencies=[Depends(auth)])
    def list_models() -> list[dict[str, Any]]:
        return [model.model_dump(mode="json") for model in foundry.list_models()]

    @app.post("/api/v1/models", dependencies=[Depends(auth)])
    def create_model(payload: CreateModelRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:create_model")
        model = foundry.create_model(payload.name, payload.config, notes=payload.notes)
        return model.model_dump(mode="json")

    @app.get("/api/v1/models/{model_id}", dependencies=[Depends(auth)])
    def get_model(model_id: str) -> dict[str, Any]:
        model = foundry.get_model(model_id)
        payload = model.model_dump(mode="json")
        payload["checkpoints"] = [
            record.model_dump(mode="json") for record in foundry.repo.list_checkpoints(model_id)
        ]
        return payload

    @app.get("/api/v1/models/{model_id}/parameters", dependencies=[Depends(auth)])
    def parameters(model_id: str) -> dict[str, Any]:
        return foundry.parameter_report(model_id)

    @app.get("/api/v1/models/{model_id}/estimate", dependencies=[Depends(auth)])
    def estimate(model_id: str, batch_size: int = 8, block_size: int = 32) -> dict[str, Any]:
        return foundry.estimate(model_id, batch_size=batch_size, block_size=block_size)

    @app.get("/api/v1/models/{model_id}/tensors", dependencies=[Depends(auth)])
    def tensors(model_id: str) -> list[dict[str, Any]]:
        return [tensor.model_dump(mode="json") for tensor in foundry.list_tensors(model_id)]

    @app.get("/api/v1/models/{model_id}/tensors/{name}/slice", dependencies=[Depends(auth)])
    def tensor_slice(model_id: str, name: str, offset: str | None = None, shape: str | None = None) -> dict[str, Any]:
        parsed_offset = tuple(int(v) for v in offset.split(",")) if offset else None
        parsed_shape = tuple(int(v) for v in shape.split(",")) if shape else None
        return foundry.tensor_slice(model_id, name, offset=parsed_offset, shape=parsed_shape)

    @app.post("/api/v1/solve", dependencies=[Depends(auth)])
    def solve(payload: SolveRequest) -> dict[str, Any]:
        return foundry.solve(
            payload.target_parameters,
            policy=payload.policy,
            family=payload.family,
            constraints=payload.constraints,
            top_k=payload.top_k,
        )

    @app.get("/api/v1/suites", dependencies=[Depends(auth)])
    def suites() -> list[dict[str, Any]]:
        from .evaluation import list_suites

        return list_suites()

    @app.get("/api/v1/history", dependencies=[Depends(auth)])
    def history(model_id: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return foundry.history(model_id=model_id, kind=kind, limit=limit)

    @app.get("/api/v1/events", dependencies=[Depends(auth)])
    def events(limit: int = 100) -> list[dict[str, Any]]:
        return foundry.repo.list_events(limit=limit)

    @app.get("/api/v1/tool-calls", dependencies=[Depends(auth)])
    def tool_calls(limit: int = 50) -> list[dict[str, Any]]:
        return foundry.repo.list_tool_calls(limit=limit)

    @app.get("/api/v1/artifacts/{artifact_id}", dependencies=[Depends(auth)])
    def get_artifact(artifact_id: str) -> FileResponse:
        """Serve a content-addressed artifact (reports, exports). Ids are validated as sha256."""
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ValidationFailure(
                "artifact ids are 64-character lowercase hex digests",
                hint="use the sha256 returned by the run that produced the artifact",
            )
        candidates = list(foundry.workspace.artifacts.glob(f"{artifact_id[:2]}/{artifact_id}*"))
        if not candidates:
            raise NotFoundError(f"artifact {artifact_id[:12]}… not found in this workspace")
        return FileResponse(candidates[0], filename=candidates[0].name)

    @app.get("/api/v1/proposals", dependencies=[Depends(auth)])
    def proposals(model_id: str | None = None) -> list[dict[str, Any]]:
        return foundry.repo.list_proposals(model_id)

    # -- weight patches --------------------------------------------------------------------
    @app.post("/api/v1/weight-patches/validate", dependencies=[Depends(auth)])
    def validate_patch(payload: PatchValidateRequest) -> dict[str, Any]:
        foundry.get_model(payload.model_id)
        patch = WeightPatch.model_validate(payload.patch)
        return {"valid": True, "patch": patch.model_dump(mode="json"), "applied": False}

    @app.post("/api/v1/weight-patches", dependencies=[Depends(auth)])
    def create_patch(payload: PatchCreateRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:patch")
        from .agents import build_patch
        from .checkpoints import flattened_state

        handle, record = foundry.handle_for(payload.model_id, payload.base_checkpoint)
        if record is None:
            raise ValidationFailure("no checkpoint to patch")
        state = flattened_state(handle.module)["tensors"]
        tensor = state.get(payload.tensor)
        if tensor is None:
            raise NotFoundError(f"tensor '{payload.tensor}' not found")
        patch = build_patch(
            model_id=payload.model_id,
            architecture_hash=handle.architecture_hash,
            base_checkpoint_sha256=record.sha256,
            tensor_name=payload.tensor,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            operation=payload.operation,
            parameters=payload.parameters,
            created_by=payload.created_by,
            note=payload.note,
        )
        return foundry.apply_patch(payload.model_id, patch)

    # -- training --------------------------------------------------------------------------
    @app.post("/api/v1/training-runs", dependencies=[Depends(auth)])
    def start_training(payload: TrainRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:train")
        model = foundry.get_model(payload.model_id)
        dataset_id = payload.dataset_id
        if dataset_id == "auto":
            dataset_id = "tiny-text" if model.config.family == "transformer-lm" else "tiny-shapes"
        config = TrainingConfig(
            model_id=payload.model_id,
            dataset_id=dataset_id,
            steps=payload.steps,
            batch_size=payload.batch_size,
            block_size=payload.block_size,
            seed=payload.seed,
            eval_every=payload.eval_every,
            checkpoint_every=payload.checkpoint_every,
            max_seconds=payload.max_seconds,
        )
        config.optimizer.lr = payload.lr
        run_id = f"run_{secrets.token_hex(8)}"
        # Register the run before the worker starts so a client can poll immediately without a 404.
        foundry.repo.add_run(
            RunRecord(
                id=run_id,
                kind="training",
                model_id=payload.model_id,
                state=RunState.QUEUED,
                config=config.model_dump(mode="json"),
            )
        )

        def target() -> None:
            control = jobs.controls[run_id]
            try:
                foundry.train(config, control=control, run_id=run_id)
            except Exception as exc:  # pragma: no cover - background safety net
                foundry.repo.update_run(
                    run_id,
                    state=RunState.FAILED,
                    error={"message": str(exc), "type": type(exc).__name__},
                    finished=True,
                )

        jobs.start(run_id, target)
        return {"run_id": run_id, "state": "queued", "poll": f"/api/v1/training-runs/{run_id}"}

    @app.get("/api/v1/training-runs/{run_id}", dependencies=[Depends(auth)])
    def get_training(run_id: str) -> dict[str, Any]:
        return foundry.repo.get_run(run_id).model_dump(mode="json")

    @app.post("/api/v1/training-runs/{run_id}/pause", dependencies=[Depends(auth)])
    def pause_training(run_id: str) -> dict[str, Any]:
        jobs.control(run_id).request_pause()
        foundry.repo.update_run(run_id, state=RunState.PAUSED)
        return {"run_id": run_id, "state": "paused"}

    @app.post("/api/v1/training-runs/{run_id}/resume", dependencies=[Depends(auth)])
    def resume_training(run_id: str) -> dict[str, Any]:
        jobs.control(run_id).request_resume()
        foundry.repo.update_run(run_id, state=RunState.RUNNING)
        return {"run_id": run_id, "state": "running"}

    @app.post("/api/v1/training-runs/{run_id}/cancel", dependencies=[Depends(auth)])
    def cancel_training(run_id: str) -> dict[str, Any]:
        jobs.control(run_id).request_cancel()
        return {"run_id": run_id, "state": "cancelling"}

    @app.get("/api/v1/events/{run_id}")
    def stream_events(run_id: str, token: str = "") -> StreamingResponse:
        """Server-sent events with metric updates. Token may be passed as a query parameter."""
        if not secrets.compare_digest(token, api_token):
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "invalid token"})

        def generator():
            last_step = -1
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                try:
                    run = foundry.repo.get_run(run_id)
                except NotFoundError:
                    yield "event: error\ndata: {\"message\":\"run not found\"}\n\n"
                    return
                for point in run.metrics:
                    if point.step > last_step:
                        last_step = point.step
                        yield f"data: {json.dumps(point.model_dump(mode='json'))}\n\n"
                if run.state in {
                    RunState.COMPLETED,
                    RunState.FAILED,
                    RunState.CANCELLED,
                    RunState.INVALID,
                    RunState.INTERRUPTED,
                }:
                    yield f"event: done\ndata: {json.dumps({'state': run.state.value})}\n\n"
                    return
                yield ": keep-alive\n\n"
                time.sleep(0.4)

        return StreamingResponse(generator(), media_type="text/event-stream")

    # -- evaluation / research / export ----------------------------------------------------
    @app.post("/api/v1/evaluations", dependencies=[Depends(auth)])
    def create_evaluation(payload: EvalRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:eval")
        report = foundry.evaluate(
            payload.model_id,
            suite=payload.suite,
            checkpoint=payload.checkpoint,
            seed=payload.seed,
            batches=payload.batches,
        )
        return report.model_dump(mode="json")

    @app.get("/api/v1/evaluations/{evaluation_id}", dependencies=[Depends(auth)])
    def get_evaluation(evaluation_id: str) -> dict[str, Any]:
        return foundry.repo.get_evaluation(evaluation_id).model_dump(mode="json")

    @app.post("/api/v1/export", dependencies=[Depends(auth)])
    def export(payload: ExportRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:export")
        return foundry.export(payload.model_id, fmt=payload.format, checkpoint=payload.checkpoint)

    @app.post("/api/v1/evolution", dependencies=[Depends(auth)])
    def evolution(payload: EvolutionRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:evolve")
        return foundry.evolve(EvolutionConfig(**payload.model_dump()))

    @app.post("/api/v1/generators", dependencies=[Depends(auth)])
    def generators(payload: GenerateRequest, request: Request) -> dict[str, Any]:
        limiter.check(f"{request.client.host if request.client else 'local'}:generate")
        return foundry.generate(GeneratorConfig(**payload.model_dump()))

    @app.post("/api/v1/proposals", dependencies=[Depends(auth)])
    def create_proposal(payload: ProposalRequest) -> dict[str, Any]:
        return foundry.propose_patch(
            payload.model_id,
            agent_id=payload.agent_id,
            goal=payload.goal,
            hypothesis=payload.hypothesis,
            tensor=payload.tensor,
            operation=payload.operation,
            parameters=payload.parameters,
            base_checkpoint=payload.base_checkpoint,
        )

    @app.post("/api/v1/proposals/{proposal_id}/approve", dependencies=[Depends(auth)])
    def approve_proposal(proposal_id: str) -> dict[str, Any]:
        return foundry.approve_proposal(proposal_id)

    @app.post("/api/v1/proposals/{proposal_id}/verify", dependencies=[Depends(auth)])
    def verify_proposal(proposal_id: str, suite: str = "weight-sanity") -> dict[str, Any]:
        return foundry.verify_proposal(proposal_id, suite=suite)

    return app


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    home: str | Path | None = None,
    token: str | None = None,
    reload: bool = False,
) -> None:  # pragma: no cover - server bootstrap
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: binding outside loopback exposes this API to the network. "
            "The bearer token is the only protection; use a firewall and TLS in front of it."
        )
    foundry = Foundry(home)
    api_token = resolve_token(foundry.workspace.root, token)
    app = create_app(foundry, token=api_token)
    print(f"HCS Neural Foundry API on http://{host}:{port}/api/v1/docs")
    print(f"token: {api_token}")
    print("workspace:", foundry.workspace.root)
    uvicorn.run(app, host=host, port=port, log_level="info")
