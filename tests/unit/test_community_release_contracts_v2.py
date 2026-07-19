from __future__ import annotations

import copy
import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from scripts import build_community_cascade_release_v2_contract as contract_builder
from scripts import produce_community_cascade_release_v2_evidence as evidence_producer
from scripts import validate_community_cascade_release_v2 as release

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict[str, Any] | bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value
        if isinstance(value, bytes)
        else (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    )
    path.write_bytes(payload)
    return payload


def _self(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[field] = ""
    result[field] = release.canonical_json_hash(result, remove=field)
    return result


def _replay_receipt(core: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(core)
    result["receipt_sha256"] = release.canonical_json_hash(result)
    return result


def _hash_binding(logical_id: str, payload: bytes) -> dict[str, str]:
    return {"logical_id": logical_id, "file_sha256": hashlib.sha256(payload).hexdigest()}


def _self_binding(
    logical_id: str, payload: bytes, document: dict[str, Any], field: str
) -> dict[str, str]:
    return {
        **_hash_binding(logical_id, payload),
        "canonical_sha256": document[field],
    }


def _quality_file_binding(payload: bytes) -> dict[str, int | str]:
    return {"file_sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _quality_native_fixture(
    quality_document: dict[str, Any], quality_payload: bytes
) -> dict[str, Any]:
    implementation = quality_document["implementation_bindings"]
    return _self(
        {
            "schema_version": evidence_producer.QUALITY_EVIDENCE_SCHEMA_VERSION,
            "receipt_type": "quality_strict_semantic_replay",
            "status": "PASS",
            "target": {
                "artifact_kind": "quality_result_receipt",
                "file_sha256": hashlib.sha256(quality_payload).hexdigest(),
                "canonical_sha256": quality_document["receipt_sha256"],
            },
            "replay": {
                "producer_file_sha256": implementation["producer_sha256"],
                "implementation_identity_sha256": release.canonical_json_hash(implementation),
                "rebuilt_receipt_sha256": quality_document["receipt_sha256"],
                "exact_semantic_equality": True,
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_document_ids": False,
                "contains_entity_values": False,
                "aggregate_only": True,
                "network_used": False,
            },
        },
        "receipt_sha256",
    )


def _candidate_native_fixture(
    candidate_document: dict[str, Any], candidate_payload: bytes
) -> dict[str, Any]:
    target = release._candidate_execution_identity(
        candidate_document,
        manifest_file_sha256=hashlib.sha256(candidate_payload).hexdigest(),
    )
    runtime_sources = release._runtime_package_source_hashes(REPOSITORY_ROOT)
    return _self(
        {
            "schema_version": evidence_producer.CANDIDATE_EVIDENCE_SCHEMA_VERSION,
            "receipt_type": "candidate_prediction_full_execution_replay",
            "status": "PASS",
            "track": candidate_document["track"],
            "target": target,
            "replayed": {
                **target,
                "predictions_byte_identical": True,
                "generation_receipt_exact_equal": True,
                "provenance_exact_equal": True,
            },
            "implementation": {
                "replay_producer_file_sha256": hashlib.sha256(
                    (REPOSITORY_ROOT / evidence_producer.PRODUCER_PATH).read_bytes()
                ).hexdigest(),
                "generator_implementation_sha256": candidate_document["generation"][
                    "generator_implementation_sha256"
                ],
                "provenance_implementation_sha256": candidate_document["provenance_implementation"][
                    "implementation_sha256"
                ],
                "runtime_package_inventory_sha256": release.canonical_json_hash(runtime_sources),
                "runtime_package_file_count": len(runtime_sources),
                "runtime_package_inventory_complete": True,
            },
            "privacy": {
                "contains_paths": False,
                "contains_raw_text": False,
                "contains_document_ids": False,
                "contains_entity_values": False,
                "aggregate_only": True,
                "network_used": False,
            },
        },
        "receipt_sha256",
    )


def _implementation_contract() -> dict[str, dict[str, str]]:
    values = release._implementation_hashes(REPOSITORY_ROOT)
    return {
        logical_id: {"logical_id": logical_id, "file_sha256": digest}
        for logical_id, digest in values.items()
    }


def _candidate_document(
    *,
    track: str,
    model: dict[str, Any],
    service: dict[str, Any],
    calibration_sha: str,
    calibration_version: str,
    candidate_predictions_sha: str,
) -> dict[str, Any]:
    service_identity = {
        "profile_id": service["profile_id"],
        "configuration_sha256": service["configuration_sha256"],
        "implementation_sha256": service["implementation_sha256"],
        "implementation_source_sha256": {"pipeline": "a" * 64, "routing": "b" * 64},
        "model_identity_sha256": model["model_identity_sha256"],
        "calibration_bundle_file_sha256": calibration_sha,
    }
    return _self(
        {
            "track": track,
            "prediction": {
                "file_sha256": candidate_predictions_sha,
                "document_count": 10,
            },
            "model": {
                "training_manifest_sha256": model["training_manifest_sha256"],
                "identity_sha256": model["model_identity_sha256"],
                "output_artifact_sha256": model["artifact_sha256"],
            },
            "calibration": {
                "bundle_file_sha256": calibration_sha,
                "calibration_version": calibration_version,
                "model_training_manifest_sha256": model["training_manifest_sha256"],
                "diagnostics_manifest_sha256": "e" * 64,
            },
            "service": None if track == "model_raw" else service_identity,
            "provenance_implementation": {
                "module_file_sha256": "c" * 64,
                "implementation_sha256": "d" * 64,
            },
            "generation": {
                "receipt_file_sha256": "a" * 64,
                "receipt_sha256": "b" * 64,
                "generator_implementation_sha256": "f" * 64,
            },
        },
        "manifest_sha256",
    )


def _verification(check_id: str) -> dict[str, Any]:
    return {
        "schema_version": "pii-zh.community-verification-receipt.v1",
        "check_id": check_id,
        "status": "PASS",
        "scope": "local_non_production",
        "contains_raw_records": False,
        "evidence_sha256": hashlib.sha256(check_id.encode()).hexdigest(),
    }


def _complete_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, release.ReleaseEvidencePaths, dict[str, Path]]:
    calibration_version = "open24-calibration-v1"
    calibration = {
        "model_version": "2" * 64,
        "calibration_version": calibration_version,
        "global_temperature": 1.0,
        "entity_temperatures": {},
        "entity_thresholds": {},
        "default_threshold": 0.5,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = _write(calibration_path, calibration)
    calibration_sha = hashlib.sha256(calibration_payload).hexdigest()
    pii_bench_path = tmp_path / "pii-bench-posthoc.json"
    _write(pii_bench_path, b"fixture pii-bench posthoc report\n")

    model = _self(
        {
            "schema_version": release.quality.MODEL_MANIFEST_SCHEMA_VERSION,
            "model_id": "fixture-aiguard24-full",
            "artifact_class": "24_label_zh_hans_token_classifier",
            "label_count": 24,
            "ordered_labels": list(release.quality.PII_CORE_LABELS),
            "taxonomy_version": "1.0.0",
            "attention_mode": "full",
            "training_manifest_file_sha256": "1" * 64,
            "training_manifest_sha256": "2" * 64,
            "model_identity_sha256": "3" * 64,
            "artifact_sha256": "4" * 64,
        },
        "manifest_sha256",
    )
    model_path = tmp_path / "final-model.json"
    model_payload = _write(model_path, model)

    service = _self(
        {
            "schema_version": release.quality.SERVICE_MANIFEST_SCHEMA_VERSION,
            "service_id": "fixture-community-cascade",
            "profile_id": "community-model-cascade-v1",
            "canonical_track": "full_system",
            "final_model_id": model["model_id"],
            "final_model_manifest_sha256": hashlib.sha256(model_payload).hexdigest(),
            "model_identity_sha256": model["model_identity_sha256"],
            "calibration_bundle_file_sha256": calibration_sha,
            "implementation_sha256": "5" * 64,
            "configuration_sha256": "6" * 64,
        },
        "manifest_sha256",
    )
    service_path = tmp_path / "service.json"
    service_payload = _write(service_path, service)

    runtime_sources = release._runtime_package_source_hashes(REPOSITORY_ROOT)
    changed_source_ids = {
        "pii_zh.evaluation.release_eval_v2_performance.py",
        "pii_zh.data.synthetic.sota_release_eval_v1.py",
        "pii_zh.data.synthetic.sota_release_eval_v2.py",
        "pii_zh.data.synthetic.sota_v1.py",
        "pii_zh.data.synthetic.sota_v2_resplit.py",
    }
    predecessor_runtime_sources = dict(runtime_sources)
    predecessor_runtime_sources.update({logical_id: "0" * 64 for logical_id in changed_source_ids})
    predecessor_source = _self(
        {
            "schema_version": "pii-zh.community-service-source-identity.v3",
            "service_id": service["service_id"],
            "profile_id": service["profile_id"],
            "implementation_source_sha256": {"pipeline": "a" * 64, "routing": "b" * 64},
            "implementation_sha256": service["implementation_sha256"],
            "runtime_package_source_sha256": predecessor_runtime_sources,
            "runtime_package_inventory_sha256": release.canonical_json_hash(
                predecessor_runtime_sources
            ),
            "runtime_package_file_count": len(predecessor_runtime_sources),
            "runtime_package_inventory_complete": True,
            "contains_paths": False,
        },
        "manifest_sha256",
    )
    predecessor_source_path = tmp_path / "source-v3.json"
    predecessor_source_payload = _write(predecessor_source_path, predecessor_source)
    predecessor_source_path.chmod(0o444)
    delta = {
        "added": {},
        "changed": {
            logical_id: runtime_sources[logical_id] for logical_id in sorted(changed_source_ids)
        },
        "removed": {},
        "added_count": 0,
        "changed_count": len(changed_source_ids),
        "removed_count": 0,
        "summary_sha256": "",
    }
    delta["summary_sha256"] = release.canonical_json_hash(delta, remove="summary_sha256")
    source = _self(
        {
            "schema_version": release.SOURCE_SCHEMA_VERSION,
            "service_id": service["service_id"],
            "profile_id": service["profile_id"],
            "implementation_source_sha256": {"pipeline": "a" * 64, "routing": "b" * 64},
            "implementation_sha256": service["implementation_sha256"],
            "runtime_package_source_sha256": runtime_sources,
            "runtime_package_inventory_sha256": release.canonical_json_hash(runtime_sources),
            "runtime_package_file_count": len(runtime_sources),
            "runtime_package_inventory_complete": True,
            "predecessor": {
                "schema_version": predecessor_source["schema_version"],
                "file_sha256": hashlib.sha256(predecessor_source_payload).hexdigest(),
                "manifest_sha256": predecessor_source["manifest_sha256"],
                "runtime_package_inventory_sha256": predecessor_source[
                    "runtime_package_inventory_sha256"
                ],
                "runtime_package_file_count": predecessor_source["runtime_package_file_count"],
            },
            "successor_delta": delta,
            "contains_paths": False,
        },
        "manifest_sha256",
    )
    source_path = tmp_path / "source.json"
    source_payload = _write(source_path, source)

    candidate_paths: dict[str, Path] = {}
    comparator_paths: dict[str, Path] = {}
    performance_paths: dict[str, Path] = {}
    candidate_documents: dict[str, dict[str, Any]] = {}
    comparator_documents: dict[str, dict[str, Any]] = {}
    performance_documents: dict[str, dict[str, Any]] = {}
    input_track_bindings: dict[str, Any] = {}
    result_tracks: dict[str, Any] = {}
    for index, track in enumerate(release.TRACKS, start=1):
        candidate_predictions_sha = str(index + 6) * 64
        comparator_predictions_sha = str(index + 8) * 64
        candidate = _candidate_document(
            track=track,
            model=model,
            service=service,
            calibration_sha=calibration_sha,
            calibration_version=calibration_version,
            candidate_predictions_sha=candidate_predictions_sha,
        )
        candidate_path = tmp_path / f"{track}-candidate.json"
        candidate_payload = _write(candidate_path, candidate)
        comparator = _self(
            {
                "track": track,
                "system_id": f"fixture-{track}-comparator",
                "prediction": {
                    "file_sha256": comparator_predictions_sha,
                    "document_count": 10,
                },
                "system_bindings": {
                    "generator_implementation": {
                        "file_sha256": str(index) * 64,
                        "size_bytes": 100,
                    },
                    "generation_receipt": {
                        "file_sha256": str(index + 1) * 64,
                        "size_bytes": 100,
                    },
                },
            },
            "manifest_sha256",
        )
        comparator_path = tmp_path / f"{track}-comparator.json"
        comparator_payload = _write(comparator_path, comparator)
        measurement = {
            "sample_count": 10,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
            "latency_p99_ms": 3.0,
            "throughput_documents_per_second": 10.0,
            "throughput_characters_per_second": 1000.0,
            "peak_cpu_rss_mb": 100.0,
            "peak_gpu_memory_mb": 0.0,
        }
        performance = _self(
            {
                "schema_version": release.quality.PERFORMANCE_SCHEMA_VERSION,
                "track": track,
                "system_id": model["model_id"] if track == "model_raw" else service["service_id"],
                "gold_sha256": "f" * 64,
                "predictions_sha256": candidate_predictions_sha,
                "final_model_manifest_sha256": hashlib.sha256(model_payload).hexdigest(),
                "service_configuration_manifest_sha256": hashlib.sha256(
                    service_payload
                ).hexdigest(),
                "measurement": measurement,
            },
            "manifest_sha256",
        )
        performance_path = tmp_path / f"{track}-performance.json"
        performance_payload = _write(performance_path, performance)
        candidate_paths[track] = candidate_path
        comparator_paths[track] = comparator_path
        performance_paths[track] = performance_path
        candidate_documents[track] = candidate
        comparator_documents[track] = comparator
        performance_documents[track] = performance
        input_track_bindings[track] = {
            "candidate_predictions": {"file_sha256": candidate_predictions_sha, "size_bytes": 1},
            "candidate_prediction_manifest": _quality_file_binding(candidate_payload),
            "candidate_prediction_manifest_sha256": candidate["manifest_sha256"],
            "comparator_predictions": {"file_sha256": comparator_predictions_sha, "size_bytes": 1},
            "comparator_prediction_manifest": _quality_file_binding(comparator_payload),
            "comparator_prediction_manifest_sha256": comparator["manifest_sha256"],
            "candidate_prediction_full_replay_required": True,
            "comparator_generation_replay_required": True,
            "performance_harness_replay_required": True,
            "performance_manifest": _quality_file_binding(performance_payload),
        }
        result_tracks[track] = {
            "declared_status": "PASS",
            "candidate_id": model["model_id"] if track == "model_raw" else service["service_id"],
            "comparator_id": comparator["system_id"],
            "performance": measurement,
        }

    implementation_bindings = {
        "producer_sha256": "a" * 64,
        "result_schema_sha256": "b" * 64,
        "registry_schema_sha256": "c" * 64,
        "metric_source_sha256": {"metric": "d" * 64},
        "prediction_provenance_source_sha256": {"provenance": "e" * 64},
    }
    quality_receipt = _self(
        {
            "schema_version": release.quality.RESULT_SCHEMA_VERSION,
            "reported_status": "PASS",
            "protocol": {"document_count": 10},
            "candidate": {"model_id": model["model_id"], "service_id": service["service_id"]},
            "input_bindings": {
                "gold": {"file_sha256": "f" * 64, "size_bytes": 1},
                "final_model_manifest": _quality_file_binding(model_payload),
                "service_configuration_manifest": _quality_file_binding(service_payload),
                "tracks": input_track_bindings,
            },
            "tracks": result_tracks,
            "claims": {"claim_activation_allowed": False},
            "implementation_bindings": implementation_bindings,
        },
        "receipt_sha256",
    )
    quality_path = tmp_path / "quality.json"
    quality_payload = _write(quality_path, quality_receipt)

    monkeypatch.setattr(release.quality, "validate_receipt_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_verify_quality_implementation", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_validate_candidate_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_validate_comparator_manifest", lambda *args, **kwargs: None)

    def fixture_artifact(
        path: Path,
        binding: dict[str, Any],
        *,
        artifact_id: str,
        repository_root: Path,
    ) -> tuple[bytes, dict[str, Any] | None]:
        del repository_root
        payload = release._verify_file_hash_binding(
            path, binding, field_name=f"release artifact {artifact_id}"
        )
        document = (
            {"artifact_sha256": hashlib.sha256(payload + artifact_id.encode()).hexdigest()}
            if artifact_id in release.release_artifacts.ARTIFACT_IDS
            else None
        )
        return payload, document

    def fixture_verification(
        path: Path,
        binding: dict[str, Any],
        *,
        check_id: str,
        repository_root: Path,
    ) -> tuple[dict[str, Any], bytes]:
        del repository_root
        return release._verify_hash_binding(path, binding, field_name=f"verification {check_id}")

    monkeypatch.setattr(release, "_verify_typed_release_artifact", fixture_artifact)
    monkeypatch.setattr(release, "_validate_typed_artifact_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_replay_typed_artifact_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_verify_typed_verification", fixture_verification)
    monkeypatch.setattr(
        release, "_validate_verification_subject_graph", lambda *args, **kwargs: None
    )

    internal_unlock = _self(
        {
            "schema_version": "pii-zh.release-eval-v2-internal-unlock.v1",
            "receipt_type": "release_eval_v2_internal_preopen_unlock",
            "status": "INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT",
            "dataset": {
                "target_split": "internal_evaluation",
                "gold_file_sha256": "f" * 64,
                "gold_size_bytes": 1,
                "gold_document_count": 10,
            },
            "bindings": {
                "final_model_binding_sha256": model["manifest_sha256"],
                "service_configuration_binding_sha256": service["manifest_sha256"],
                "calibration_bundle_file_sha256": calibration_sha,
                "calibration_diagnostics_manifest_sha256": "e" * 64,
            },
            "authorization": {"quality_production_allowed": True},
            "privacy": {"internal_evaluation_opened_hashed_or_decoded_during_unlock": False},
        },
        "receipt_sha256",
    )
    internal_unlock_path = tmp_path / "internal-unlock.json"
    internal_unlock_payload = _write(internal_unlock_path, internal_unlock)
    monkeypatch.setattr(
        release,
        "_validate_internal_unlock",
        lambda *args, **kwargs: None,
    )

    identity = release._identity_payload(
        quality_receipt=quality_receipt,
        model=model,
        service=service,
        source=source,
        internal_unlock=internal_unlock,
    )
    quality_native = _quality_native_fixture(quality_receipt, quality_payload)
    quality_native_path = tmp_path / "quality-native-replay.json"
    quality_native_payload = _write(quality_native_path, quality_native)
    quality_native_binding = {
        "schema_version": quality_native["schema_version"],
        "file_sha256": hashlib.sha256(quality_native_payload).hexdigest(),
        "canonical_sha256": quality_native["receipt_sha256"],
    }
    quality_impl_file, quality_impl_identity = release._wrapper_replay_implementation(
        receipt_type="quality_strict_replay",
        target_document=quality_receipt,
        repository_root=REPOSITORY_ROOT,
    )
    quality_core = release._expected_replay_receipt_core(
        receipt_type="quality_strict_replay",
        track=None,
        quality_receipt=quality_receipt,
        quality_file_sha256=hashlib.sha256(quality_payload).hexdigest(),
        identity=identity,
        target_document=quality_receipt,
        target_file_sha256=hashlib.sha256(quality_payload).hexdigest(),
        target_kind="quality_result_receipt",
        implementation_file_sha256=quality_impl_file,
        implementation_identity_sha256=quality_impl_identity,
        upstream_replay_evidence=quality_native_binding,
    )
    quality_replay = _replay_receipt(quality_core)
    quality_replay_path = tmp_path / "quality-replay.json"
    quality_replay_payload = _write(quality_replay_path, quality_replay)
    quality_replay_binding = {
        **_self_binding(
            "quality_strict_replay",
            quality_replay_payload,
            quality_replay,
            "receipt_sha256",
        ),
        "implementation_file_sha256": quality_core["replay"]["implementation_file_sha256"],
        "implementation_identity_sha256": quality_core["replay"]["implementation_identity_sha256"],
    }

    replay_paths: dict[str, Path] = {}
    native_paths: dict[str, Path] = {}
    replay_inventory: dict[str, dict[str, Any]] = {track: {} for track in release.TRACKS}

    def fake_comparator_native(path: Path, **_kwargs: Any) -> dict[str, str]:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = path.read_bytes()
        return {
            "schema_version": document["schema_version"],
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_sha256": document["receipt_sha256"],
        }

    monkeypatch.setattr(release, "_verify_comparator_native_replay", fake_comparator_native)
    for track in release.TRACKS:
        candidate_native = _candidate_native_fixture(
            candidate_documents[track], candidate_paths[track].read_bytes()
        )
        candidate_native_path = tmp_path / f"{track}-candidate-native.json"
        candidate_native_payload = _write(candidate_native_path, candidate_native)
        candidate_native_binding = {
            "schema_version": candidate_native["schema_version"],
            "file_sha256": hashlib.sha256(candidate_native_payload).hexdigest(),
            "canonical_sha256": candidate_native["receipt_sha256"],
        }
        candidate_key = f"{track}:candidate_prediction_full_replay"
        native_paths[candidate_key] = candidate_native_path

        comparator_native = _self(
            {
                "schema_version": "pii-zh.fixture-comparator-native-replay.v1",
                "status": "PASS",
                "track": track,
            },
            "receipt_sha256",
        )
        comparator_native_path = tmp_path / f"{track}-comparator-native.json"
        comparator_native_payload = _write(comparator_native_path, comparator_native)
        comparator_native_binding = {
            "schema_version": comparator_native["schema_version"],
            "file_sha256": hashlib.sha256(comparator_native_payload).hexdigest(),
            "canonical_sha256": comparator_native["receipt_sha256"],
        }
        comparator_key = f"{track}:comparator_generation_replay"
        native_paths[comparator_key] = comparator_native_path

        candidate_impl = release._wrapper_replay_implementation(
            receipt_type="candidate_prediction_full_replay",
            target_document=candidate_documents[track],
            repository_root=REPOSITORY_ROOT,
        )
        comparator_impl = release._wrapper_replay_implementation(
            receipt_type="comparator_generation_replay",
            target_document=comparator_documents[track],
            repository_root=REPOSITORY_ROOT,
        )
        performance_impl = release._performance_replay_implementation(track)
        target_sets = {
            "candidate_prediction_full_replay": (
                candidate_documents[track],
                candidate_paths[track].read_bytes(),
                "candidate_prediction_manifest",
                candidate_impl[0],
                candidate_impl[1],
                candidate_native_binding,
            ),
            "comparator_generation_replay": (
                comparator_documents[track],
                comparator_paths[track].read_bytes(),
                "comparator_prediction_manifest",
                comparator_impl[0],
                comparator_impl[1],
                comparator_native_binding,
            ),
            "performance_harness_replay": (
                performance_documents[track],
                performance_paths[track].read_bytes(),
                "performance_manifest",
                performance_impl[0],
                performance_impl[1],
                None,
            ),
        }
        for kind, (
            target,
            target_payload,
            target_kind,
            impl_file,
            impl_identity,
            upstream_replay_evidence,
        ) in target_sets.items():
            core = release._expected_replay_receipt_core(
                receipt_type=kind,
                track=track,
                quality_receipt=quality_receipt,
                quality_file_sha256=hashlib.sha256(quality_payload).hexdigest(),
                identity=identity,
                target_document=target,
                target_file_sha256=hashlib.sha256(target_payload).hexdigest(),
                target_kind=target_kind,
                implementation_file_sha256=impl_file,
                implementation_identity_sha256=impl_identity,
                upstream_replay_evidence=upstream_replay_evidence,
            )
            receipt = _replay_receipt(core)
            path = tmp_path / f"{track}-{kind}.json"
            payload = _write(path, receipt)
            key = f"{track}:{kind}"
            replay_paths[key] = path
            replay_inventory[track][kind] = {
                **_self_binding(key.replace(":", "."), payload, receipt, "receipt_sha256"),
                "implementation_file_sha256": impl_file,
                "implementation_identity_sha256": impl_identity,
            }

    release_artifacts: dict[str, Path] = {
        "final_model_manifest": model_path,
        "service_configuration_manifest": service_path,
        "service_source_manifest": source_path,
        "benchmark_report": quality_path,
    }
    artifact_bindings: dict[str, Any] = {
        "final_model_manifest": _hash_binding("final_model_manifest", model_payload),
        "service_configuration_manifest": _hash_binding(
            "service_configuration_manifest", service_payload
        ),
        "service_source_manifest": _hash_binding("service_source_manifest", source_payload),
        "benchmark_report": _hash_binding("quality_result", quality_payload),
    }
    for artifact_id in release.ARTIFACT_IDS:
        if artifact_id in release_artifacts:
            continue
        path = tmp_path / f"artifact-{artifact_id}"
        payload = _write(path, f"fixture {artifact_id}\n".encode())
        release_artifacts[artifact_id] = path
        artifact_bindings[artifact_id] = _hash_binding(artifact_id, payload)

    verification_paths: dict[str, Path] = {}
    verification_bindings: dict[str, Any] = {}
    for check_id in release.VERIFICATION_IDS:
        path = tmp_path / f"verification-{check_id}.json"
        payload = _write(path, _verification(check_id))
        verification_paths[check_id] = path
        verification_bindings[check_id] = _hash_binding(check_id, payload)

    contract = {
        "schema_version": release.CONTRACT_SCHEMA_VERSION,
        "contract_id": release.CONTRACT_ID,
        "status": "candidate_complete_pending_user_authorization",
        "purpose": "fail_closed_release_gate_for_open24_quality_and_independent_replays",
        "scope": release.EXPECTED_SCOPE,
        "quality_gate": {
            "result_receipt": _self_binding(
                "quality_result", quality_payload, quality_receipt, "receipt_sha256"
            ),
            "strict_replay_receipt": quality_replay_binding,
            "required_schema_version": release.quality.RESULT_SCHEMA_VERSION,
            "required_status": "PASS",
            "reported_pass_is_not_replay_proof": True,
        },
        "candidate_identity": {
            "model_id": model["model_id"],
            "service_id": service["service_id"],
            "training_manifest_sha256": model["training_manifest_sha256"],
            "model_identity_sha256": model["model_identity_sha256"],
            "calibration_bundle_file_sha256": calibration_sha,
            "service_configuration_sha256": service["configuration_sha256"],
            "service_implementation_sha256": service["implementation_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "internal_unlock_sha256": internal_unlock["receipt_sha256"],
            "final_model_manifest": _self_binding(
                "final_model_manifest", model_payload, model, "manifest_sha256"
            ),
            "service_configuration_manifest": _self_binding(
                "service_configuration_manifest", service_payload, service, "manifest_sha256"
            ),
            "calibration_bundle": _hash_binding("calibration_bundle", calibration_payload),
            "service_source_manifest": _self_binding(
                "service_source_manifest", source_payload, source, "manifest_sha256"
            ),
            "internal_preopen_unlock": _self_binding(
                "internal_preopen_unlock",
                internal_unlock_payload,
                internal_unlock,
                "receipt_sha256",
            ),
        },
        "replay_evidence": replay_inventory,
        "release_artifacts": artifact_bindings,
        "verification_receipts": verification_bindings,
        "publication": {
            "github": {
                "target_id": None,
                "authorized": False,
                "published": False,
                "publication_receipt": None,
            },
            "hugging_face": {
                "target_id": None,
                "authorized": False,
                "published": False,
                "publication_receipt": None,
            },
            "publication_requires_same_turn_user_authorization": True,
        },
        "claims": release.EXPECTED_CLAIMS,
        "privacy": release.EXPECTED_PRIVACY,
        "implementation": _implementation_contract(),
        "historical_policy": release.EXPECTED_HISTORICAL_POLICY,
        "contract_sha256": "",
    }
    contract["contract_sha256"] = release.canonical_json_hash(contract, remove="contract_sha256")
    contract_path = tmp_path / "completed-contract.json"
    _write(contract_path, contract)
    evidence = release.ReleaseEvidencePaths(
        quality_receipt=quality_path,
        quality_replay_receipt=quality_replay_path,
        quality_native_replay=quality_native_path,
        final_model_manifest=model_path,
        service_configuration_manifest=service_path,
        pii_bench_posthoc_report=pii_bench_path,
        calibration_bundle=calibration_path,
        service_source_manifest=source_path,
        predecessor_service_source_manifest=predecessor_source_path,
        internal_preopen_unlock=internal_unlock_path,
        candidate_manifests=candidate_paths,
        comparator_manifests=comparator_paths,
        performance_manifests=performance_paths,
        replay_receipts=replay_paths,
        native_replay_evidence=native_paths,
        release_artifacts=release_artifacts,
        model_package_root=tmp_path,
        release_wheel=release_artifacts["wheel_manifest"],
        wheelhouse_root=tmp_path,
        verification_receipts=verification_paths,
    )
    return contract_path, evidence, replay_paths


def test_checked_in_v2_contract_is_path_free_closed_and_honestly_blocked() -> None:
    contract = release.validate_contract()
    schema = json.loads((REPOSITORY_ROOT / release.CONTRACT_SCHEMA_PATH).read_text())
    receipt_schema = json.loads((REPOSITORY_ROOT / release.RECEIPT_SCHEMA_PATH).read_text())
    replay_schema = json.loads((REPOSITORY_ROOT / release.REPLAY_SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(receipt_schema)
    Draft202012Validator.check_schema(replay_schema)
    Draft202012Validator(schema).validate(contract)

    serialized = json.dumps(contract, ensure_ascii=False)
    assert '"path"' not in serialized
    assert "/data/" not in serialized
    report = release.build_report()
    assert report["status"] == "BLOCKED"
    assert report["local_candidate_complete"] is False
    assert len([item for item in report["blocker_ids"] if item.startswith("replay:")]) == 6
    assert "quality_strict_replay" in report["blocker_ids"]


def test_documented_community_commands_cover_current_required_inputs() -> None:
    documentation = (REPOSITORY_ROOT / "docs/community-cascade-release-v2.md").read_text()
    model_card_command = documentation.split(
        "produce_community_cascade_release_v2_artifacts.py model-card \\", 1
    )[1].split("```", 1)[0]
    assert all(
        token in model_card_command
        for token in (
            "--training-manifest",
            "--pii-bench-report",
            "--template-asset",
            "--markdown-output",
        )
    )
    package_command = documentation.split("scripts/build_release.py \\", 1)[1].split("```", 1)[0]
    assert all(
        token in package_command
        for token in (
            "--model-card",
            "--model-card-artifact",
            "--pii-bench-report",
            "--final-model-manifest",
            "--community-v2-preauthorization",
        )
    )
    license_command = documentation.split(
        "produce_community_cascade_release_v2_artifacts.py license-report \\", 1
    )[1].split("```", 1)[0]
    assert "--model-package-root" in license_command
    contract_command = documentation.split(
        "scripts/build_community_cascade_release_v2_contract.py \\", 1
    )[1].split("```", 1)[0]
    assert "--pii-bench-posthoc-report" in contract_command

    parser_actions = {action.dest: action for action in contract_builder._parser()._actions}
    assert parser_actions["pii_bench_posthoc_report"].required is True


def test_completed_fixture_reaches_authorization_only_after_seven_exact_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, evidence, _replay_paths = _complete_fixture(tmp_path, monkeypatch)
    report = release.build_report(contract_path=contract_path, evidence=evidence)

    assert report["status"] == "READY_FOR_USER_AUTHORIZATION"
    assert report["local_candidate_complete"] is True
    assert report["quality"]["all_required_replays"] == "PASS"
    assert sum(len(value) for value in report["replay_evidence"].values()) == 6
    assert report["identity"]["calibration_bundle_file_sha256"]
    assert report["privacy"]["model_weights_read"] is True
    assert report["privacy"]["network_used"] is False
    serialized = json.dumps(report, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert '"path"' not in serialized


def test_contract_builder_derives_ready_contract_and_publishes_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract_path, evidence, _replay_paths = _complete_fixture(tmp_path, monkeypatch)

    contract = contract_builder.build_contract(evidence)
    assert contract["status"] == "candidate_complete_pending_user_authorization"
    assert contract["publication"]["github"]["authorized"] is False
    assert contract["publication"]["hugging_face"]["authorized"] is False
    assert contract["contract_sha256"] == release.canonical_json_hash(
        contract, remove="contract_sha256"
    )

    output = tmp_path / "builder" / "completed-contract.json"
    contract_builder.publish_contract(output, contract)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    report = release.build_report(contract_path=output, evidence=evidence)
    assert report["status"] == "READY_FOR_USER_AUTHORIZATION"
    with pytest.raises(contract_builder.CommunityContractBuilderError, match="overwrite"):
        contract_builder.publish_contract(output, contract)


def test_contract_builder_refuses_incomplete_inventory_and_publication_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract_path, evidence, _replay_paths = _complete_fixture(tmp_path, monkeypatch)
    incomplete_artifacts = dict(evidence.release_artifacts)
    incomplete_artifacts.pop("model_card")

    with pytest.raises(
        contract_builder.CommunityContractBuilderError,
        match="release artifact inventory is not exact",
    ):
        contract_builder.build_contract(replace(evidence, release_artifacts=incomplete_artifacts))

    with pytest.raises(
        contract_builder.CommunityContractBuilderError,
        match="refuses publication receipts",
    ):
        contract_builder.build_contract(
            replace(evidence, publication_receipts={"github": evidence.quality_receipt})
        )


@pytest.mark.parametrize(
    "target_update",
    [
        {"authorized": True},
        {
            "target_id": "forged-target",
            "authorized": True,
            "published": True,
            "publication_receipt": {
                "logical_id": "forged",
                "file_sha256": "1" * 64,
                "size_bytes": 1,
            },
        },
    ],
)
def test_community_v2_contract_cannot_claim_authorized_or_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_update: dict[str, Any],
) -> None:
    contract_path, evidence, _replay_paths = _complete_fixture(tmp_path, monkeypatch)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["publication"]["github"].update(target_update)
    contract["contract_sha256"] = release.canonical_json_hash(contract, remove="contract_sha256")
    _write(contract_path, contract)
    with pytest.raises(release.CommunityContractError):
        release.build_report(contract_path=contract_path, evidence=evidence)


def test_final_gate_consumes_frozen_dependency_scan_without_network_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = {
        artifact_id: {"artifact_id": artifact_id}
        for artifact_id in release.release_artifacts.ARTIFACT_IDS
    }
    documents["container_manifest"]["result"] = {"image_id": "sha256:" + "1" * 64}
    artifact_paths: dict[str, Path] = {}
    for artifact_id in documents:
        path = tmp_path / f"{artifact_id}.json"
        path.write_text("{}\n")
        artifact_paths[artifact_id] = path
    physical = tmp_path / "physical"
    physical.mkdir()
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"wheel")
    pii_bench = tmp_path / "pii-bench-posthoc.json"
    pii_bench.write_text("{}\n")
    evidence = release.ReleaseEvidencePaths(
        quality_receipt=tmp_path / "quality.json",
        final_model_manifest=tmp_path / "model.json",
        service_configuration_manifest=tmp_path / "service.json",
        pii_bench_posthoc_report=pii_bench,
        model_package_root=physical,
        release_wheel=wheel,
        wheelhouse_root=physical,
        release_artifacts=artifact_paths,
    )
    builders = {
        "build_model_card_artifact": "model_card",
        "build_model_package_manifest": "model_package_manifest",
        "build_technical_documentation_manifest": "technical_documentation_manifest",
        "build_wheel_manifest": "wheel_manifest",
        "build_wheelhouse_manifest": "wheelhouse_manifest",
        "build_container_manifest": "container_manifest",
        "build_sbom_artifact": "sbom",
        "build_license_report": "license_report",
        "build_public_artifact_scan": "public_artifact_scan",
    }
    for builder_name, artifact_id in builders.items():
        monkeypatch.setattr(
            release.release_artifacts,
            builder_name,
            lambda *args, _artifact_id=artifact_id, **kwargs: documents[_artifact_id],
        )

    def forbidden_osv(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("final gate must not rerun OSV")

    monkeypatch.setattr(release.release_artifacts, "build_dependency_scan", forbidden_osv)
    release._replay_typed_artifact_sources(
        documents,
        evidence,
        repository_root=REPOSITORY_ROOT,
    )


def test_forged_replay_bool_or_disconnected_target_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, evidence, replay_paths = _complete_fixture(tmp_path, monkeypatch)
    target = replay_paths["full_system:performance_harness_replay"]
    document = json.loads(target.read_text())
    document["status"] = True
    document["receipt_sha256"] = release.canonical_json_hash(document, remove="receipt_sha256")
    _write(target, document)

    with pytest.raises(release.CommunityContractError):
        release.build_report(contract_path=contract_path, evidence=evidence)


def test_stale_candidate_native_after_runtime_source_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, supplied, _replay_paths = _complete_fixture(tmp_path, monkeypatch)
    native_path = supplied.native_replay_evidence["model_raw:candidate_prediction_full_replay"]
    document = json.loads(native_path.read_text(encoding="utf-8"))
    document["implementation"]["runtime_package_inventory_sha256"] = "0" * 64
    document["receipt_sha256"] = release.canonical_json_hash(document, remove="receipt_sha256")
    _write(native_path, document)

    with pytest.raises(release.CommunityContractError, match="candidate native replay"):
        release.build_report(contract_path=contract_path, evidence=supplied)


def test_quality_pass_without_one_independent_replay_remains_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, evidence, _replay_paths = _complete_fixture(tmp_path, monkeypatch)
    contract = json.loads(contract_path.read_text())
    contract["replay_evidence"]["full_system"]["performance_harness_replay"] = None
    contract["status"] = "blocked_pending_replays_and_artifacts"
    contract["contract_sha256"] = release.canonical_json_hash(contract, remove="contract_sha256")
    _write(contract_path, contract)

    report = release.build_report(contract_path=contract_path, evidence=evidence)
    assert report["status"] == "BLOCKED"
    assert report["quality"] is None
    assert "replay:full_system:performance_harness_replay" in report["blocker_ids"]


def test_publish_report_is_read_only_no_clobber(tmp_path: Path) -> None:
    report = json.loads(json.dumps(release.build_report(), ensure_ascii=False, sort_keys=True))
    output = tmp_path / "release-receipt.json"
    release.publish_report(output, report)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert json.loads(output.read_text()) == report
    with pytest.raises(release.CommunityCascadeReleaseV2Error, match="overwrite"):
        release.publish_report(output, report)
