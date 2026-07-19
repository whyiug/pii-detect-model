from __future__ import annotations

import copy
import hashlib
import json
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import produce_community_cascade_release_v2_evidence as evidence


def _self(document: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result[key] = ""
    result[key] = evidence.canonical_json_hash(result, remove=key)
    return result


def _write(path: Path, document: dict[str, Any] | bytes) -> bytes:
    payload = (
        document
        if isinstance(document, bytes)
        else (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    )
    path.write_bytes(payload)
    return payload


def _candidate(track: str = "model_raw") -> dict[str, Any]:
    return _self(
        {
            "track": track,
            "prediction": {"file_sha256": "1" * 64, "document_count": 2},
            "generation": {
                "receipt_file_sha256": "2" * 64,
                "receipt_sha256": "3" * 64,
                "generator_implementation_sha256": "4" * 64,
            },
            "provenance_implementation": {
                "module_file_sha256": "5" * 64,
                "schema_file_sha256": "6" * 64,
                "cli_file_sha256": "7" * 64,
                "implementation_sha256": "8" * 64,
            },
        },
        "manifest_sha256",
    )


def _context(candidate: dict[str, Any], candidate_payload: bytes) -> evidence.ReleaseContext:
    runtime_sources = evidence._runtime_package_source_hashes(evidence.REPOSITORY_ROOT)
    quality = {
        "receipt_sha256": "9" * 64,
        "reported_status": "PASS",
        "input_bindings": {
            "tracks": {
                "model_raw": {
                    "candidate_prediction_manifest": {
                        "file_sha256": hashlib.sha256(candidate_payload).hexdigest(),
                        "size_bytes": len(candidate_payload),
                    }
                }
            }
        },
        "tracks": {"model_raw": {"declared_status": "PASS"}},
    }
    identity = {
        "model_id": "fixture-model",
        "service_id": "fixture-service",
        "training_manifest_sha256": "a" * 64,
        "model_identity_sha256": "b" * 64,
        "calibration_bundle_file_sha256": "c" * 64,
        "service_configuration_sha256": "d" * 64,
        "service_implementation_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "quality_receipt_sha256": "9" * 64,
        "internal_unlock_sha256": "0" * 64,
    }
    return evidence.ReleaseContext(
        quality=quality,
        quality_payload=b"fixture-quality",
        model={},
        service={},
        source={
            "runtime_package_inventory_sha256": evidence.canonical_json_hash(runtime_sources),
            "runtime_package_file_count": len(runtime_sources),
            "runtime_package_inventory_complete": True,
        },
        internal_unlock={},
        full_candidate={},
        identity=identity,
    )


def _candidate_args(tmp_path: Path, candidate_path: Path) -> SimpleNamespace:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_bytes(b'{"doc_id":"a","spans":[]}\n{"doc_id":"b","spans":[]}\n')
    generation = tmp_path / "generation.json"
    generation.write_text('{"fixture":true}\n', encoding="utf-8")
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("fixture", encoding="utf-8")
    return SimpleNamespace(
        track="model_raw",
        manifest=candidate_path,
        predictions=predictions,
        generation_receipt=generation,
        gold=placeholder,
        dataset_manifest=placeholder,
        materialization_receipt=placeholder,
        freeze_receipt=placeholder,
        model_artifact=placeholder,
        selection_receipt=placeholder,
        amendment=placeholder,
        protocol=placeholder,
        development_manifest=placeholder,
        v1_release_manifest=placeholder,
        candidate=[placeholder, placeholder, placeholder],
        calibration_bundle=placeholder,
        calibration_diagnostics=placeholder,
        calibration_fit_gold=placeholder,
        calibration_fit_predictions=placeholder,
        calibration_fit_generation_receipt=placeholder,
        calibration_fit_prediction_manifest=placeholder,
        model_raw_predictions=None,
        model_raw_generation_receipt=None,
        model_raw_prediction_manifest=None,
        device="cpu",
        quality_receipt=placeholder,
        final_model_manifest=placeholder,
        service_configuration_manifest=placeholder,
        service_source_manifest=placeholder,
        predecessor_service_source_manifest=placeholder,
        internal_preopen_unlock=placeholder,
        full_system_candidate_manifest=placeholder,
        native_output=tmp_path / "candidate-native.json",
        output=tmp_path / "candidate-wrapper.json",
    )


def test_native_pass_has_no_public_synthesis_api_and_publish_is_no_clobber(
    tmp_path: Path,
) -> None:
    assert not hasattr(evidence, "build_candidate_native_evidence")
    assert not hasattr(evidence, "build_quality_native_evidence")
    native = {"schema_version": "fixture.path-free.v1", "status": "TEST_ONLY"}
    output = tmp_path / "native.json"
    evidence.publish_read_only_json(output, native)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="overwrite"):
        evidence.publish_read_only_json(output, native)


def test_runtime_package_inventory_rejects_ambiguous_logical_ids(tmp_path: Path) -> None:
    package = tmp_path / "src" / "pii_zh"
    (package / "a").mkdir(parents=True)
    (package / "a" / "b.py").write_text("nested = True\n", encoding="utf-8")
    (package / "a.b.py").write_text("flat = True\n", encoding="utf-8")

    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="ambiguous"):
        evidence._runtime_package_source_hashes(tmp_path)
    with pytest.raises(evidence.release.CommunityCascadeReleaseV2Error, match="ambiguous"):
        evidence.release._runtime_package_source_hashes(tmp_path)


def test_source_v4_delta_rejects_any_change_beyond_approved_amendment() -> None:
    expected_sources = evidence._EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
    predecessor = {
        **{logical_id: "1" * 64 for logical_id in expected_sources},
        "pii_zh.fusion.replacement.py": "2" * 64,
    }
    admitted = dict(predecessor)
    admitted.update({logical_id: "3" * 64 for logical_id in expected_sources})
    assert evidence._source_successor_delta(predecessor, admitted)["changed"] == {
        logical_id: "3" * 64 for logical_id in sorted(expected_sources)
    }

    extra_change = dict(admitted)
    extra_change["pii_zh.fusion.replacement.py"] = "4" * 64
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="portability amendment"):
        evidence._source_successor_delta(predecessor, extra_change)

    added_file = dict(admitted)
    added_file["pii_zh.new_module.py"] = "5" * 64
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="portability amendment"):
        evidence._source_successor_delta(predecessor, added_file)


def test_actual_release_source_validator_accepts_only_bound_v3_to_v4_delta(
    tmp_path: Path,
) -> None:
    runtime = evidence.release._runtime_package_source_hashes(evidence.REPOSITORY_ROOT)
    predecessor_runtime = dict(runtime)
    predecessor_runtime.update(
        {
            logical_id: "0" * 64
            for logical_id in evidence._EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
        }
    )
    implementation_sources = {"pipeline": "1" * 64}
    predecessor = _self(
        {
            "schema_version": evidence.PREDECESSOR_SOURCE_SCHEMA_VERSION,
            "service_id": "fixture-service",
            "profile_id": "fixture-profile",
            "implementation_source_sha256": implementation_sources,
            "implementation_sha256": "2" * 64,
            "runtime_package_source_sha256": predecessor_runtime,
            "runtime_package_inventory_sha256": evidence.canonical_json_hash(predecessor_runtime),
            "runtime_package_file_count": len(predecessor_runtime),
            "runtime_package_inventory_complete": True,
            "contains_paths": False,
        },
        "manifest_sha256",
    )
    predecessor_path = tmp_path / "source-v3.json"
    predecessor_payload = _write(predecessor_path, predecessor)
    delta = evidence._source_successor_delta(predecessor_runtime, runtime)
    source = _self(
        {
            "schema_version": evidence.SOURCE_SCHEMA_VERSION,
            "service_id": "fixture-service",
            "profile_id": "fixture-profile",
            "implementation_source_sha256": implementation_sources,
            "implementation_sha256": "2" * 64,
            "runtime_package_source_sha256": runtime,
            "runtime_package_inventory_sha256": evidence.canonical_json_hash(runtime),
            "runtime_package_file_count": len(runtime),
            "runtime_package_inventory_complete": True,
            "predecessor": {
                "schema_version": evidence.PREDECESSOR_SOURCE_SCHEMA_VERSION,
                "file_sha256": hashlib.sha256(predecessor_payload).hexdigest(),
                "manifest_sha256": predecessor["manifest_sha256"],
                "runtime_package_inventory_sha256": predecessor["runtime_package_inventory_sha256"],
                "runtime_package_file_count": len(predecessor_runtime),
            },
            "successor_delta": delta,
            "contains_paths": False,
        },
        "manifest_sha256",
    )
    source_schema = evidence.load_json_path(
        evidence.REPOSITORY_ROOT / evidence.SOURCE_SCHEMA_PATH,
        field="source v4 schema",
    )
    predecessor_schema = evidence.load_json_path(
        evidence.REPOSITORY_ROOT / evidence.PREDECESSOR_SOURCE_SCHEMA_PATH,
        field="source v3 schema",
    )
    service = {
        "service_id": "fixture-service",
        "profile_id": "fixture-profile",
        "implementation_sha256": "2" * 64,
    }
    candidate = {
        "service": {
            "implementation_sha256": "2" * 64,
            "implementation_source_sha256": implementation_sources,
        }
    }
    evidence.release._validate_source_manifest(
        source,
        schema=source_schema,
        predecessor_document=predecessor,
        predecessor_payload=predecessor_payload,
        predecessor_schema=predecessor_schema,
        service=service,
        full_candidate=candidate,
    )

    stale = copy.deepcopy(source)
    stale["successor_delta"]["changed"] = {
        evidence._EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_ID: "3" * 64
    }
    stale["manifest_sha256"] = evidence.canonical_json_hash(stale, remove="manifest_sha256")
    with pytest.raises(evidence.CommunityContractError, match="closed schema"):
        evidence.release._validate_source_manifest(
            stale,
            schema=source_schema,
            predecessor_document=predecessor,
            predecessor_payload=predecessor_payload,
            predecessor_schema=predecessor_schema,
            service=service,
            full_candidate=candidate,
        )


def test_source_v4_predecessor_must_be_mode_0444(tmp_path: Path) -> None:
    predecessor = tmp_path / "source-v3.json"
    predecessor.write_text("{}\n", encoding="utf-8")
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="mode-0444"):
        evidence._load_immutable_json(predecessor, field="predecessor")

    predecessor.chmod(0o444)
    document, payload = evidence._load_immutable_json(predecessor, field="predecessor")
    assert document == {}
    assert payload == b"{}\n"


def test_quality_release_context_is_validated_before_long_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    args = SimpleNamespace(
        native_output=tmp_path / "native.json",
        output=tmp_path / "wrapper.json",
        quality_receipt=tmp_path / "quality.json",
    )

    def context(_args: Any) -> Any:
        events.append("context")
        return SimpleNamespace(quality={})

    def replay(*_args: Any, **_kwargs: Any) -> Any:
        events.append("replay")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(evidence, "_release_context_from_args", context)
    monkeypatch.setattr(evidence, "_quality_inputs", lambda _args: {})
    monkeypatch.setattr(evidence.quality, "replay_receipt", replay)
    with pytest.raises(RuntimeError, match="ordering assertion"):
        evidence._produce_quality_replay(args)
    assert events == ["context", "replay"]


def test_candidate_wrapper_is_issued_only_after_execution_bytes_and_provenance_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_payload = _write(candidate_path, candidate)
    args = _candidate_args(tmp_path, candidate_path)
    context = _context(candidate, candidate_payload)
    candidate["prediction"]["file_sha256"] = hashlib.sha256(
        args.predictions.read_bytes()
    ).hexdigest()
    candidate["generation"]["receipt_file_sha256"] = hashlib.sha256(
        args.generation_receipt.read_bytes()
    ).hexdigest()
    candidate = _self(
        {key: value for key, value in candidate.items() if key != "manifest_sha256"},
        "manifest_sha256",
    )
    candidate_payload = _write(candidate_path, candidate)
    context = _context(candidate, candidate_payload)

    monkeypatch.setattr(evidence, "_release_context_from_args", lambda _args: context)
    monkeypatch.setattr(evidence.release, "_validate_candidate_manifest", lambda *a, **k: None)
    monkeypatch.setattr(
        evidence, "_load_candidate_target", lambda _args: (candidate, candidate_payload)
    )

    def fake_generator(_args: Any, *, predictions: Path, generation_receipt: Path) -> None:
        shutil.copyfile(args.predictions, predictions)
        shutil.copyfile(args.generation_receipt, generation_receipt)

    monkeypatch.setattr(evidence, "_run_candidate_generator", fake_generator)
    expected_replay = {
        "status": "PASS",
        "track": "model_raw",
        "target_split": "internal_evaluation",
        "manifest_file_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "manifest_sha256": candidate["manifest_sha256"],
        "predictions_file_sha256": candidate["prediction"]["file_sha256"],
        "prediction_document_count": 2,
    }
    monkeypatch.setattr(
        evidence,
        "replay_release_eval_v2_prediction_manifest",
        lambda *_args, **_kwargs: expected_replay,
    )

    result = evidence._produce_candidate_replay(args)
    assert result["wrapper_receipt_sha256"]
    wrapper = json.loads(args.output.read_text(encoding="utf-8"))
    native = json.loads(args.native_output.read_text(encoding="utf-8"))
    assert wrapper["status"] == "EXACT_REPLAY_PASS"
    assert native["replayed"]["predictions_byte_identical"] is True
    assert stat.S_IMODE(args.output.stat().st_mode) == 0o444


def test_candidate_replay_difference_leaves_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_payload = _write(candidate_path, candidate)
    args = _candidate_args(tmp_path, candidate_path)
    context = _context(candidate, candidate_payload)
    monkeypatch.setattr(evidence, "_release_context_from_args", lambda _args: context)
    monkeypatch.setattr(evidence.release, "_validate_candidate_manifest", lambda *a, **k: None)
    monkeypatch.setattr(
        evidence, "_load_candidate_target", lambda _args: (candidate, candidate_payload)
    )

    def different_generator(_args: Any, *, predictions: Path, generation_receipt: Path) -> None:
        predictions.write_bytes(b'{"doc_id":"different","spans":[]}\n')
        shutil.copyfile(args.generation_receipt, generation_receipt)

    monkeypatch.setattr(evidence, "_run_candidate_generator", different_generator)
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="bytes differ"):
        evidence._produce_candidate_replay(args)
    assert not args.native_output.exists()
    assert not args.output.exists()


def test_service_source_requires_current_candidate_source_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _self(
        {
            "model_id": "fixture-model",
            "model_identity_sha256": "1" * 64,
        },
        "manifest_sha256",
    )
    model_path = tmp_path / "model.json"
    model_payload = _write(model_path, model)
    service = _self(
        {
            "service_id": "fixture-service",
            "profile_id": "fixture-profile",
            "implementation_sha256": "2" * 64,
            "model_identity_sha256": "1" * 64,
        },
        "manifest_sha256",
    )
    service_path = tmp_path / "service.json"
    _write(service_path, service)
    candidate = _self(
        {
            "track": "full_system",
            "service": {
                "profile_id": service["profile_id"],
                "implementation_sha256": service["implementation_sha256"],
                "model_identity_sha256": service["model_identity_sha256"],
                "implementation_source_sha256": {"pipeline": "3" * 64},
            },
        },
        "manifest_sha256",
    )
    candidate_path = tmp_path / "candidate.json"
    _write(candidate_path, candidate)
    expected_sources = evidence._EXPECTED_SOURCE_SUCCESSOR_CHANGED_LOGICAL_IDS
    predecessor_runtime = {
        **{logical_id: "5" * 64 for logical_id in expected_sources},
        "pii_zh.fusion.replacement.py": "6" * 64,
    }
    predecessor = _self(
        {
            "schema_version": evidence.PREDECESSOR_SOURCE_SCHEMA_VERSION,
            "service_id": service["service_id"],
            "profile_id": service["profile_id"],
            "implementation_source_sha256": {"pipeline": "3" * 64},
            "implementation_sha256": service["implementation_sha256"],
            "runtime_package_source_sha256": predecessor_runtime,
            "runtime_package_inventory_sha256": evidence.canonical_json_hash(predecessor_runtime),
            "runtime_package_file_count": len(predecessor_runtime),
            "runtime_package_inventory_complete": True,
            "contains_paths": False,
        },
        "manifest_sha256",
    )
    predecessor_path = tmp_path / "source-v3.json"
    predecessor_payload = _write(predecessor_path, predecessor)
    predecessor_path.chmod(0o444)
    monkeypatch.setattr(evidence.release, "_validate_model_manifest", lambda *_args: None)
    monkeypatch.setattr(
        evidence.release, "_validate_service_manifest", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(evidence, "validate_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evidence,
        "validate_release_eval_v2_prediction_manifest",
        lambda *_args: {"status": "PASS_METADATA_ONLY", "full_replay_required": True},
    )
    monkeypatch.setattr(evidence, "_source_hashes", lambda _root: {"pipeline": "3" * 64})
    monkeypatch.setattr(
        evidence,
        "_runtime_package_source_hashes",
        lambda _root: {
            **{logical_id: "7" * 64 for logical_id in expected_sources},
            "pii_zh.fusion.replacement.py": "6" * 64,
        },
    )
    result = evidence.build_service_source_manifest(
        final_model_manifest=model_path,
        service_configuration_manifest=service_path,
        full_system_candidate_manifest=candidate_path,
        predecessor_source_manifest=predecessor_path,
    )
    assert result["schema_version"] == evidence.SOURCE_SCHEMA_VERSION
    assert result["implementation_source_sha256"] == {"pipeline": "3" * 64}
    assert result["runtime_package_file_count"] == len(expected_sources) + 1
    assert result["runtime_package_inventory_complete"] is True
    assert result["predecessor"] == {
        "schema_version": evidence.PREDECESSOR_SOURCE_SCHEMA_VERSION,
        "file_sha256": hashlib.sha256(predecessor_payload).hexdigest(),
        "manifest_sha256": predecessor["manifest_sha256"],
        "runtime_package_inventory_sha256": predecessor["runtime_package_inventory_sha256"],
        "runtime_package_file_count": len(expected_sources) + 1,
    }
    assert result["successor_delta"]["added"] == {}
    assert result["successor_delta"]["changed"] == {
        logical_id: "7" * 64 for logical_id in sorted(expected_sources)
    }
    assert result["successor_delta"]["removed"] == {}
    assert result["successor_delta"]["changed_count"] == len(expected_sources)
    delta = dict(result["successor_delta"])
    claimed_delta_hash = delta.pop("summary_sha256")
    assert claimed_delta_hash == evidence.canonical_json_hash(delta)
    assert result["contains_paths"] is False
    assert hashlib.sha256(model_payload).hexdigest()

    candidate["service"]["implementation_source_sha256"]["pipeline"] = "4" * 64
    candidate = _self(
        {key: value for key, value in candidate.items() if key != "manifest_sha256"},
        "manifest_sha256",
    )
    _write(candidate_path, candidate)
    with pytest.raises(evidence.CommunityReleaseEvidenceError, match="current final service"):
        evidence.build_service_source_manifest(
            final_model_manifest=model_path,
            service_configuration_manifest=service_path,
            full_system_candidate_manifest=candidate_path,
            predecessor_source_manifest=predecessor_path,
        )
