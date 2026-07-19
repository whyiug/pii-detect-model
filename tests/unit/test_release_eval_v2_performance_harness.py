from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import validate_community_cascade_release_v2 as release_v2

from pii_zh.data.schema import (
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    dumps_document,
)
from pii_zh.evaluation import PredictionRecord, Span, write_prediction_jsonl
from pii_zh.evaluation import release_eval_v2_performance as performance


def test_cli_bootstrap_makes_repository_scripts_importable(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    cli = repository_root / "scripts/benchmark_release_eval_v2_candidate.py"
    code = (
        "import runpy; "
        f"runpy.run_path({str(cli)!r}, run_name='performance_cli_import'); "
        "import scripts.build_aiguard24_v3_release_eval_v2_amendment"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _record(doc_id: str = "synthetic-1", text: str = "测试文本") -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh-Hans-CN",
        domain="synthetic-test",
        scene="performance-fixture",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="synthetic_fixture",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
    )


def _snapshot(payload: bytes) -> performance.FileSnapshot:
    return performance.FileSnapshot(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        nonempty_lines=sum(bool(line.strip()) for line in payload.splitlines()),
    )


def _prediction_bytes(record: PredictionRecord) -> bytes:
    handle = io.StringIO()
    write_prediction_jsonl([record], handle)
    return handle.getvalue().encode()


def _identity() -> performance.PreparedIdentity:
    return performance.PreparedIdentity(
        model={
            "model_id": "aiguard24-full-seed42",
            "model_identity_sha256": "a" * 64,
            "training_manifest_sha256": "b" * 64,
        },
        model_payload=b"model-binding\n",
        service={
            "service_id": "community-model-cascade-v1",
            "profile_id": "community-model-cascade-v1",
            "calibration_bundle_file_sha256": "c" * 64,
            "configuration_sha256": "d" * 64,
            "implementation_sha256": "e" * 64,
        },
        service_payload=b"service-binding\n",
        calibration={
            "global_temperature": 1.0,
            "entity_temperatures": {},
            "entity_thresholds": {},
            "default_threshold": 0.5,
        },
        calibration_sha256="c" * 64,
        unlock={"receipt_sha256": "f" * 64},
    )


def _source_v4(runtime: dict[str, str]) -> dict[str, Any]:
    changed_logical_ids = performance.EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
    delta: dict[str, Any] = {
        "added": {},
        "changed": {logical_id: runtime[logical_id] for logical_id in sorted(changed_logical_ids)},
        "removed": {},
        "added_count": 0,
        "changed_count": len(changed_logical_ids),
        "removed_count": 0,
    }
    delta["summary_sha256"] = performance.canonical_json_hash(delta)
    source: dict[str, Any] = {
        "schema_version": performance.SOURCE_SCHEMA_VERSION,
        "service_id": "community-model-cascade-v1",
        "profile_id": "community-model-cascade-v1",
        "implementation_source_sha256": {"pipeline": "1" * 64},
        "implementation_sha256": "2" * 64,
        "runtime_package_source_sha256": runtime,
        "runtime_package_inventory_sha256": performance.canonical_json_hash(runtime),
        "runtime_package_file_count": len(runtime),
        "runtime_package_inventory_complete": True,
        "predecessor": {
            "schema_version": "pii-zh.community-service-source-identity.v3",
            "file_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "runtime_package_inventory_sha256": "5" * 64,
            "runtime_package_file_count": len(runtime),
        },
        "successor_delta": delta,
        "contains_paths": False,
    }
    source["manifest_sha256"] = performance.canonical_json_hash(source)
    return source


def _request(
    tmp_path: Path,
    track: performance.PerformanceTrack = "model_raw",
) -> performance.HarnessRequest:
    return performance.HarnessRequest(
        track=track,
        target_split="fixture",
        model=tmp_path / "model",
        calibration=tmp_path / "calibration.json",
        source=tmp_path / "input.jsonl",
        expected_predictions=tmp_path / "predictions.jsonl",
        final_model_binding=tmp_path / "model-binding.json",
        service_configuration_binding=tmp_path / "service-binding.json",
        device="cpu",
    )


def test_measurement_is_computed_and_requires_exact_prediction_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(text="张三")
    source = _snapshot((dumps_document(record) + "\n").encode())
    expected = _snapshot(
        _prediction_bytes(
            PredictionRecord(
                doc_id=record.doc_id,
                spans=(Span(start=0, end=2, label="PERSON_NAME", score=0.9),),
            )
        )
    )

    def detector(_request: object, _identity: object) -> Any:
        def run(texts: object) -> list[list[object]]:
            return [
                [SimpleNamespace(start=0, end=2, entity_type="PERSON_NAME", score=0.9)]
                for _text in texts  # type: ignore[union-attr]
            ]

        return run

    monkeypatch.setattr(performance, "_build_detector", detector)
    observed = performance._measure_execution(
        _request(tmp_path),
        identity=_identity(),
        input_snapshot=source,
        expected_snapshot=expected,
    )

    assert observed.predictions_sha256 == expected.sha256
    assert observed.measurement["sample_count"] == 1
    assert 0 < observed.measurement["latency_p50_ms"]
    assert (
        observed.measurement["latency_p50_ms"]
        <= observed.measurement["latency_p95_ms"]
        <= observed.measurement["latency_p99_ms"]
    )
    assert observed.measurement["peak_gpu_memory_mb"] == 0.0

    tampered = _snapshot(expected.payload.replace(b"0.9", b"0.8"))
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="byte-identical"):
        performance._measure_execution(
            _request(tmp_path),
            identity=_identity(),
            input_snapshot=source,
            expected_snapshot=tampered,
        )


def test_formal_stage_strict_replay_happens_before_guard_and_input_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    stage = performance.FormalStageReplayPaths(
        protocol=tmp_path / "protocol",
        development_manifest=tmp_path / "development",
        v1_release_manifest=tmp_path / "v1",
        candidates=(tmp_path / "c13", tmp_path / "c42", tmp_path / "c97"),
        selection_receipt=tmp_path / "selection",
        amendment=tmp_path / "amendment",
        dataset_manifest=tmp_path / "dataset",
        materialization_receipt=tmp_path / "materialization",
        freeze_receipt=tmp_path / "freeze",
        calibration_gold=tmp_path / "calibration-gold",
        calibration_authorization=tmp_path / "authorization",
        calibration_diagnostics=tmp_path / "diagnostics",
        calibration_fit_predictions=tmp_path / "fit-predictions",
        calibration_fit_generation_receipt=tmp_path / "fit-generation",
        calibration_fit_prediction_manifest=tmp_path / "fit-manifest",
        internal_unlock=tmp_path / "unlock",
    )
    fixture_request = _request(tmp_path)
    request = performance.HarnessRequest(
        track=fixture_request.track,
        target_split="internal_evaluation",
        model=fixture_request.model,
        calibration=fixture_request.calibration,
        source=fixture_request.source,
        expected_predictions=fixture_request.expected_predictions,
        final_model_binding=fixture_request.final_model_binding,
        service_configuration_binding=fixture_request.service_configuration_binding,
        device=fixture_request.device,
        stage_replay=stage,
    )

    def fake_replay(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        events.append("strict-replay")
        return {
            "status": "INTERNAL_UNLOCK_STRICT_REPLAY_PASS",
            "internal_evaluation_opened_hashed_or_decoded": False,
        }

    def fake_guard(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        events.append("preopen-guard")
        return {"status": "INTERNAL_PREOPEN_PASS", "expected_input_sha256": "a" * 64}

    monkeypatch.setattr(performance, "replay_internal_unlock", fake_replay)
    monkeypatch.setattr(performance, "guard_internal_preopen", fake_guard)

    replay, guard = performance._formal_preopen_replay(request)

    assert replay["status"] == "INTERNAL_UNLOCK_STRICT_REPLAY_PASS"
    assert guard["status"] == "INTERNAL_PREOPEN_PASS"
    assert events == ["strict-replay", "preopen-guard"]


def test_formal_stage_failure_cannot_reach_internal_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request = performance.HarnessRequest(
        track=request.track,
        target_split="internal_evaluation",
        model=request.model,
        calibration=request.calibration,
        source=request.source,
        expected_predictions=request.expected_predictions,
        final_model_binding=request.final_model_binding,
        service_configuration_binding=request.service_configuration_binding,
        device=request.device,
        stage_replay=SimpleNamespace(),  # type: ignore[arg-type]
    )
    opened: list[Path] = []

    def fail_replay(_request: object) -> object:
        raise performance.ReleaseEvalV2PerformanceError("stage failed")

    def spy_read(path: Path, **kwargs: object) -> object:
        del kwargs
        opened.append(path)
        raise AssertionError("internal input must not be opened")

    monkeypatch.setattr(performance, "_formal_preopen_replay", fail_replay)
    monkeypatch.setattr(performance, "_read_regular", spy_read)

    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="stage failed"):
        performance._execute(request)
    assert opened == []


def test_remeasurement_accepts_noise_but_rejects_broken_relations() -> None:
    recorded = {
        "sample_count": 2,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 20.0,
        "latency_p99_ms": 30.0,
        "throughput_documents_per_second": 100.0,
        "throughput_characters_per_second": 1000.0,
        "peak_cpu_rss_mb": 100.0,
        "peak_gpu_memory_mb": 0.0,
    }
    observed = dict(recorded)
    observed.update(
        {
            "latency_p50_ms": 12.0,
            "latency_p95_ms": 24.0,
            "latency_p99_ms": 36.0,
            "throughput_documents_per_second": 80.0,
            "throughput_characters_per_second": 800.0,
            "peak_cpu_rss_mb": 110.0,
        }
    )
    performance._validate_remeasurement(
        recorded,
        observed,
        document_count=2,
        character_count=20,
        device="cpu",
    )

    broken = dict(observed)
    broken["latency_p95_ms"] = 1.0
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="monotonic"):
        performance._validate_remeasurement(
            recorded,
            broken,
            document_count=2,
            character_count=20,
            device="cpu",
        )


def test_performance_replay_accepts_only_current_source_v4_approved_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {
        **{
            logical_id: "a" * 64
            for logical_id in performance.EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
        },
        "pii_zh.fusion.replacement.py": "b" * 64,
    }
    monkeypatch.setattr(
        performance,
        "_runtime_package_source_hashes",
        lambda _root: dict(runtime),
    )
    source = _source_v4(runtime)
    performance._validate_source_successor(source)

    extra_delta = json.loads(json.dumps(source))
    extra_delta["successor_delta"]["added"] = {"pii_zh.extra.py": "c" * 64}
    extra_delta["successor_delta"]["added_count"] = 1
    delta = dict(extra_delta["successor_delta"])
    delta.pop("summary_sha256")
    extra_delta["successor_delta"]["summary_sha256"] = performance.canonical_json_hash(delta)
    extra_delta.pop("manifest_sha256")
    extra_delta["manifest_sha256"] = performance.canonical_json_hash(extra_delta)
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="closed schema"):
        performance._validate_source_successor(extra_delta)

    stale_runtime = dict(runtime)
    stale_runtime["pii_zh.fusion.replacement.py"] = "d" * 64
    monkeypatch.setattr(
        performance,
        "_runtime_package_source_hashes",
        lambda _root: stale_runtime,
    )
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="current runtime"):
        performance._validate_source_successor(source)

    predecessor = json.loads(json.dumps(source))
    predecessor["schema_version"] = "pii-zh.community-service-source-identity.v3"
    predecessor.pop("manifest_sha256")
    predecessor["manifest_sha256"] = performance.canonical_json_hash(predecessor)
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="service source"):
        performance._validate_source_successor(predecessor)


def test_release_replay_rejects_stale_source_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request = performance.HarnessRequest(
        track=request.track,
        target_split="internal_evaluation",
        model=request.model,
        calibration=request.calibration,
        source=request.source,
        expected_predictions=request.expected_predictions,
        final_model_binding=request.final_model_binding,
        service_configuration_binding=request.service_configuration_binding,
        device=request.device,
    )
    snapshot = _snapshot(b"{}\n")
    monkeypatch.setattr(performance, "_read_json", lambda *a, **k: ({}, snapshot))
    events: list[str] = []

    def reject(_source: object) -> None:
        events.append("source")
        raise performance.ReleaseEvalV2PerformanceError("stale successor")

    monkeypatch.setattr(performance, "_validate_source_successor", reject)
    monkeypatch.setattr(
        performance,
        "_execute",
        lambda _request: events.append("execute"),
    )
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="stale successor"):
        performance.replay(
            request,
            performance_manifest=tmp_path / "performance.json",
            quality_receipt=tmp_path / "quality.json",
            service_source_manifest=tmp_path / "source.json",
        )
    assert events == ["source"]


def test_execution_rejects_implementation_drift_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    identity = _identity()
    snapshot = _snapshot(b"{}\n")
    observations = SimpleNamespace()
    identities = iter([("1" * 64, "2" * 64), ("1" * 64, "3" * 64)])
    monkeypatch.setattr(performance, "_load_binding_pair", lambda _request: identity)
    monkeypatch.setattr(performance, "_implementation_identity", lambda _track: next(identities))
    monkeypatch.setattr(performance, "_read_regular", lambda *a, **k: snapshot)
    monkeypatch.setattr(performance, "_measure_execution", lambda *a, **k: observations)
    monkeypatch.setattr(
        performance,
        "_model_identity",
        lambda _model: {"identity_sha256": identity.model["model_identity_sha256"]},
    )
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="implementation changed"):
        performance._execute(request)


def test_performance_implementation_identity_binds_noncomponent_runtime_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_id = "pii_zh.unlisted_runtime_dependency.py"
    runtime_sources = {logical_id: "a" * 64}
    monkeypatch.setattr(
        performance,
        "_runtime_package_source_hashes",
        lambda _root: dict(runtime_sources),
    )
    monkeypatch.setattr(
        performance,
        "_read_regular",
        lambda *_args, **_kwargs: _snapshot(b"stable-component\n"),
    )
    monkeypatch.setattr(
        performance,
        "generation_implementation_identity",
        lambda _track: {"identity_sha256": "b" * 64},
    )

    first_file, first_identity = performance._implementation_identity("model_raw")
    runtime_sources[logical_id] = "c" * 64
    second_file, second_identity = performance._implementation_identity("model_raw")

    assert first_file == second_file
    assert first_identity != second_identity


@pytest.mark.parametrize("track", ["model_raw", "model_calibrated", "full_system"])
def test_performance_manifest_is_quality_compatible_and_self_hashed(
    tmp_path: Path, track: performance.PerformanceTrack
) -> None:
    identity = _identity()
    observation = performance.ExecutionObservation(
        measurement={
            "sample_count": 2,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
            "latency_p99_ms": 3.0,
            "throughput_documents_per_second": 10.0,
            "throughput_characters_per_second": 100.0,
            "peak_cpu_rss_mb": 100.0,
            "peak_gpu_memory_mb": 0.0,
        },
        input_sha256="1" * 64,
        predictions_sha256="2" * 64,
        document_count=2,
        character_count=20,
    )
    value = performance.build_performance_manifest(_request(tmp_path, track), identity, observation)

    assert value["schema_version"] == performance.PERFORMANCE_SCHEMA_VERSION
    assert value["track"] == track
    assert value["system_id"] == (
        "community-model-cascade-v1" if track == "full_system" else "aiguard24-full-seed42"
    )
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256")
    assert claimed == performance.canonical_json_hash(unsigned)
    assert str(tmp_path) not in json.dumps(value)


def test_release_replay_receipt_exactly_matches_release_gate_derivation() -> None:
    track: performance.CanonicalReleaseTrack = "model_raw"
    performance_manifest = {"manifest_sha256": "1" * 64}
    quality = {
        "receipt_sha256": "2" * 64,
        "reported_status": "PASS",
        "input_bindings": {"gold": {"file_sha256": "3" * 64}},
        "tracks": {track: {"declared_status": "PASS"}},
    }
    identity = {
        "model_id": "model-v1",
        "service_id": "service-v1",
        "training_manifest_sha256": "4" * 64,
        "model_identity_sha256": "5" * 64,
        "calibration_bundle_file_sha256": "6" * 64,
        "service_configuration_sha256": "7" * 64,
        "service_implementation_sha256": "8" * 64,
        "source_manifest_sha256": "9" * 64,
        "quality_receipt_sha256": "2" * 64,
        "internal_unlock_sha256": "a" * 64,
    }
    actual = performance.build_release_replay_receipt(
        track=track,
        quality=quality,
        quality_file_sha256="b" * 64,
        performance=performance_manifest,
        performance_file_sha256="c" * 64,
        identity=identity,
        implementation_file_sha256="d" * 64,
        implementation_identity_sha256="e" * 64,
    )
    expected_core = release_v2._expected_replay_receipt_core(
        receipt_type="performance_harness_replay",
        track=track,
        quality_receipt=quality,
        quality_file_sha256="b" * 64,
        identity=identity,
        target_document=performance_manifest,
        target_file_sha256="c" * 64,
        target_kind="performance_manifest",
        implementation_file_sha256="d" * 64,
        implementation_identity_sha256="e" * 64,
    )
    expected = dict(expected_core)
    expected["receipt_sha256"] = release_v2.canonical_json_hash(expected_core)

    assert actual == expected


def test_read_snapshot_rejects_symlink_and_writer_is_0444_no_clobber(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="unavailable"):
        performance._read_regular(link, field="fixture", maximum_bytes=1024)

    output = tmp_path / "manifest.json"
    performance.write_read_only_json(output, {"status": "PASS"})
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(performance.ReleaseEvalV2PerformanceError, match="overwrite"):
        performance.write_read_only_json(output, {"status": "PASS"})
