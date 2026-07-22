"""Bounded, local-first FastAPI adapter for the cascade pipeline."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Protocol, TypeVar

from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pii_zh import __version__
from pii_zh.cascade import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    DEFAULT_SERVICE_PROFILE_VERSION,
    CascadeConfig,
    CascadeDetection,
    CascadeMode,
    build_community_model_service_pipeline,
    build_rules_only_service_pipeline,
)
from pii_zh.full_bie73 import (
    COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION,
    FULL_BIE73_DEFAULT_SCOPE,
    FullBie73Scope,
    build_full_bie73_service_pipeline,
)

SERVICE_OUTPUT_SCHEMA_VERSION: Final = "pii-zh.service.output.v1"
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8000
DEFAULT_MAX_REQUEST_BODY_BYTES: Final = 256 * 1024
DEFAULT_MAX_TEXT_CHARS: Final = 100_000
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_CONCURRENCY: Final = 4
DEFAULT_CUDA_MAX_CONCURRENCY: Final = 1
MAX_ENTITY_FILTERS: Final = 64
MAX_REPLACEMENT_CHARS: Final = 256

_PRIVATE_RESPONSE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
T = TypeVar("T")


class Pipeline(Protocol):
    """Narrow injection contract implemented by :class:`CascadePipeline`."""

    config: CascadeConfig

    def detect(
        self, text: str, *, entities: Sequence[str] | None = None
    ) -> list[CascadeDetection]: ...

    def redact(
        self,
        text: str,
        replacement: str = "<PII>",
        *,
        entities: Sequence[str] | None = None,
    ) -> str: ...


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class AnalyzeRequest(_StrictRequest):
    text: str = Field(max_length=DEFAULT_MAX_TEXT_CHARS)
    entities: list[str] | None = Field(default=None, max_length=MAX_ENTITY_FILTERS)


class RedactRequest(AnalyzeRequest):
    replacement: str = Field(default="<PII>", max_length=MAX_REPLACEMENT_CHARS)


class DetectionResponse(BaseModel):
    schema_version: str
    start: int
    end: int
    entity_type: str
    score: float
    source: str
    sources: list[str]
    decision_process: list[str]


class AnalyzeResponse(BaseModel):
    schema_version: str = SERVICE_OUTPUT_SCHEMA_VERSION
    mode: CascadeMode
    profile_version: str
    model_identity: dict[str, str | int | bool | None] | None
    detections: list[DetectionResponse]


class RedactResponse(BaseModel):
    schema_version: str = SERVICE_OUTPUT_SCHEMA_VERSION
    mode: CascadeMode
    profile_version: str
    model_identity: dict[str, str | int | bool | None] | None
    redacted_text: str


class HealthResponse(BaseModel):
    status: str = "ok"
    schema_version: str = SERVICE_OUTPUT_SCHEMA_VERSION
    mode: CascadeMode
    profile_version: str
    model_enabled: bool
    model_identity: dict[str, str | int | bool | None] | None
    max_request_body_bytes: int
    max_text_chars: int
    request_timeout_seconds: float
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class ServiceLimits:
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    def __post_init__(self) -> None:
        for name, value in (
            ("max_request_body_bytes", self.max_request_body_bytes),
            ("max_text_chars", self.max_text_chars),
            ("max_concurrency", self.max_concurrency),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_text_chars > DEFAULT_MAX_TEXT_CHARS:
            raise ValueError(f"max_text_chars must not exceed {DEFAULT_MAX_TEXT_CHARS}")
        timeout = self.request_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("request_timeout_seconds must be numeric")
        if not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")


class _BodyTooLarge(Exception):
    pass


class _RequestBodyLimitMiddleware:
    """Reject declared and streamed request bodies before JSON validation."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await _json_error(400, "invalid request body")(scope, receive, send)
                return
            if content_length < 0:
                await _json_error(400, "invalid request body")(scope, receive, send)
                return
            if content_length > self.max_body_bytes:
                await _json_error(413, "request body too large")(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _json_error(413, "request body too large")(scope, receive, send)


class _ServiceBusy(Exception):
    pass


class _ServiceTimeout(Exception):
    pass


class _BoundedExecutor:
    def __init__(self, *, max_concurrency: int, timeout_seconds: float) -> None:
        self._slots = asyncio.Semaphore(max_concurrency)
        self._timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="pii-zh-service",
        )

    async def run(self, function: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=self._timeout_seconds)
        except TimeoutError:
            raise _ServiceBusy from None
        remaining = self._timeout_seconds - (loop.time() - started)
        if remaining <= 0:
            self._slots.release()
            raise _ServiceBusy
        try:
            future = loop.run_in_executor(self._executor, function)
        except Exception:
            self._slots.release()
            raise
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except TimeoutError:
            if future.done():
                self._slots.release()
            else:
                future.add_done_callback(lambda _future: self._slots.release())
            raise _ServiceTimeout from None
        except BaseException:
            if future.done():
                self._slots.release()
            else:
                future.add_done_callback(lambda _future: self._slots.release())
            raise
        self._slots.release()
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers=_PRIVATE_RESPONSE_HEADERS,
    )


def _set_private_headers(response: Response) -> None:
    for name, value in _PRIVATE_RESPONSE_HEADERS.items():
        response.headers[name] = value


def _default_concurrency(device: str) -> int:
    return (
        DEFAULT_CUDA_MAX_CONCURRENCY
        if device.strip().casefold().startswith("cuda")
        else DEFAULT_MAX_CONCURRENCY
    )


def _model_identity(pipeline: Pipeline) -> dict[str, str | int | bool | None] | None:
    value = getattr(pipeline, "model_identity", None)
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, (str, int, bool, type(None)))
        for key, item in value.items()
    ):
        raise TypeError("pipeline model identity is not privacy-safe")
    return dict(value)


def create_app(
    pipeline: Pipeline | None = None,
    *,
    profile_version: str = DEFAULT_SERVICE_PROFILE_VERSION,
    mode: CascadeMode | None = None,
    model_path: str | Path | None = None,
    scope: FullBie73Scope | None = None,
    device: str = "cpu",
    micro_batch_size: int = 16,
    calibration: Mapping[str, object] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_concurrency: int | None = None,
) -> FastAPI:
    """Create an offline service with an optional explicit local model.

    The zero-argument behavior remains the historical rules-only profile.  The
    stable full-BIE73 profile defaults to its formally selected Open-24 cascade
    and requires only a local checkpoint path; callers may explicitly request
    its model-only service ablation or closed-8 pre-decode scope.  Historical
    profiles preserve their existing construction paths.  Model verification
    failures never fall back to rules-only behavior.
    """

    resolved_max_concurrency = (
        _default_concurrency(device) if max_concurrency is None else max_concurrency
    )
    limits = ServiceLimits(
        max_request_body_bytes=max_request_body_bytes,
        max_text_chars=max_text_chars,
        request_timeout_seconds=request_timeout_seconds,
        max_concurrency=resolved_max_concurrency,
    )
    resolved_mode: CascadeMode = (
        "cascade"
        if mode is None and profile_version == COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION
        else "rules-only"
        if mode is None
        else mode
    )
    if pipeline is not None:
        if (
            profile_version != DEFAULT_SERVICE_PROFILE_VERSION
            or resolved_mode != "rules-only"
            or model_path is not None
            or scope is not None
            or device != "cpu"
            or micro_batch_size != 16
            or calibration is not None
            or thresholds is not None
        ):
            raise ValueError(
                "pipeline cannot be combined with profile or model construction options"
            )
        active_pipeline: Pipeline = pipeline
    elif profile_version == COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION:
        if resolved_mode not in {"model-only", "cascade"}:
            raise ValueError(
                f"{COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION} supports only "
                "model-only or cascade mode"
            )
        if model_path is None:
            raise ValueError(f"model_path is required in {resolved_mode} mode")
        if calibration is not None or thresholds is not None:
            raise ValueError(
                "the stable BIE73 HTTP profile uses its frozen thresholds and does not "
                "accept calibration or threshold overrides"
            )
        active_pipeline = build_full_bie73_service_pipeline(
            model_path,
            scope=FULL_BIE73_DEFAULT_SCOPE if scope is None else scope,
            mode=resolved_mode,
            device=device,
            micro_batch_size=micro_batch_size,
        )
    elif resolved_mode == "rules-only":
        if scope is not None:
            raise ValueError(
                f"scope requires profile_version={COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION}"
            )
        if model_path is not None:
            raise ValueError("model_path is only valid in model-only or cascade mode")
        if calibration is not None or thresholds is not None:
            raise ValueError("calibration and thresholds require a model-enabled mode")
        active_pipeline = build_rules_only_service_pipeline(profile_version)
    else:
        if resolved_mode not in {"model-only", "cascade"}:
            raise ValueError("mode must be rules-only, model-only, or cascade")
        if scope is not None:
            raise ValueError(
                f"scope requires profile_version={COMMUNITY_FULL_BIE73_CASCADE_PROFILE_VERSION}"
            )
        if profile_version != COMMUNITY_MODEL_SERVICE_PROFILE_VERSION:
            raise ValueError("model-enabled HTTP service requires the community model profile")
        if model_path is None:
            raise ValueError(f"model_path is required in {resolved_mode} mode")
        model_options: dict[str, object] = {
            "mode": resolved_mode,
            "device": device,
            "micro_batch_size": micro_batch_size,
        }
        if calibration is not None:
            model_options["calibration"] = calibration
        if thresholds is not None:
            model_options["thresholds"] = thresholds
        active_pipeline = build_community_model_service_pipeline(
            model_path,
            **model_options,  # type: ignore[arg-type]
        )
    model_identity = _model_identity(active_pipeline)
    executor = _BoundedExecutor(
        max_concurrency=limits.max_concurrency,
        timeout_seconds=float(limits.request_timeout_seconds),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        executor.close()

    app = FastAPI(
        title="pii-zh local cascade service",
        version=__version__,
        debug=False,
        lifespan=lifespan,
    )
    app.add_middleware(
        _RequestBodyLimitMiddleware,
        max_body_bytes=limits.max_request_body_bytes,
    )
    app.state.pipeline = active_pipeline
    app.state.service_limits = limits

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: object, _exc: RequestValidationError
    ) -> JSONResponse:
        return _json_error(422, "request validation failed")

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(response: Response) -> HealthResponse:
        _set_private_headers(response)
        return HealthResponse(
            mode=active_pipeline.config.mode,
            profile_version=active_pipeline.config.profile_version,
            model_enabled=active_pipeline.config.uses_model,
            model_identity=model_identity,
            max_request_body_bytes=limits.max_request_body_bytes,
            max_text_chars=limits.max_text_chars,
            request_timeout_seconds=float(limits.request_timeout_seconds),
            max_concurrency=limits.max_concurrency,
        )

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    async def analyze(
        payload: AnalyzeRequest, response: Response
    ) -> AnalyzeResponse | JSONResponse:
        _set_private_headers(response)
        if len(payload.text) > limits.max_text_chars:
            return _json_error(413, "text too long")
        try:
            detections = await executor.run(
                partial(active_pipeline.detect, payload.text, entities=payload.entities)
            )
        except _ServiceBusy:
            return _json_error(503, "service busy")
        except _ServiceTimeout:
            return _json_error(504, "analysis timed out")
        except Exception:
            return _json_error(500, "analysis failed")
        return AnalyzeResponse(
            mode=active_pipeline.config.mode,
            profile_version=active_pipeline.config.profile_version,
            model_identity=model_identity,
            detections=[DetectionResponse.model_validate(item.to_dict()) for item in detections],
        )

    @app.post("/v1/redact", response_model=RedactResponse)
    async def redact(payload: RedactRequest, response: Response) -> RedactResponse | JSONResponse:
        _set_private_headers(response)
        if len(payload.text) > limits.max_text_chars:
            return _json_error(413, "text too long")
        try:
            redacted = await executor.run(
                partial(
                    active_pipeline.redact,
                    payload.text,
                    payload.replacement,
                    entities=payload.entities,
                )
            )
        except _ServiceBusy:
            return _json_error(503, "service busy")
        except _ServiceTimeout:
            return _json_error(504, "redaction timed out")
        except Exception:
            return _json_error(500, "redaction failed")
        return RedactResponse(
            mode=active_pipeline.config.mode,
            profile_version=active_pipeline.config.profile_version,
            model_identity=model_identity,
            redacted_text=redacted,
        )

    return app


def run(
    *,
    pipeline: Pipeline | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    profile_version: str = DEFAULT_SERVICE_PROFILE_VERSION,
    mode: CascadeMode | None = None,
    model_path: str | Path | None = None,
    scope: FullBie73Scope | None = None,
    device: str = "cpu",
    micro_batch_size: int = 16,
    calibration: Mapping[str, object] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_concurrency: int | None = None,
) -> None:
    """Run one local worker with access logging disabled by default."""

    import uvicorn

    uvicorn.run(
        create_app(
            pipeline=pipeline,
            profile_version=profile_version,
            mode=mode,
            model_path=model_path,
            scope=scope,
            device=device,
            micro_batch_size=micro_batch_size,
            calibration=calibration,
            thresholds=thresholds,
            max_request_body_bytes=max_request_body_bytes,
            max_text_chars=max_text_chars,
            request_timeout_seconds=request_timeout_seconds,
            max_concurrency=max_concurrency,
        ),
        host=host,
        port=port,
        access_log=False,
        log_level="warning",
        workers=1,
    )


if __name__ == "__main__":  # pragma: no cover
    run()


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_CUDA_MAX_CONCURRENCY",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_REQUEST_BODY_BYTES",
    "DEFAULT_MAX_TEXT_CHARS",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "SERVICE_OUTPUT_SCHEMA_VERSION",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "DetectionResponse",
    "HealthResponse",
    "RedactRequest",
    "RedactResponse",
    "ServiceLimits",
    "create_app",
    "run",
]
