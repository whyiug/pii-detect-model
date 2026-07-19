#!/usr/bin/env python3
"""Assemble one complete community release v2 contract from verified paths.

The builder accepts paths only.  It derives every digest, canonical identity,
status and PASS binding from the supplied bytes, validates the resulting
contract with the same release gate, and publishes only when the derived state
is exactly ``READY_FOR_USER_AUTHORIZATION``.  It cannot authorize or publish a
GitHub/Hugging Face target.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import validate_community_cascade_release_v2 as release
    from scripts.community_contract_utils import (
        CommunityContractError,
        canonical_json_hash,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    import validate_community_cascade_release_v2 as release  # type: ignore[no-redef]
    from community_contract_utils import (  # type: ignore[no-redef]
        CommunityContractError,
        canonical_json_hash,
        read_regular_file,
        sha256_bytes,
        strict_json_bytes,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = "scripts/build_community_cascade_release_v2_contract.py"


class CommunityContractBuilderError(CommunityContractError):
    """Raised when the complete contract cannot be derived honestly."""


def _load_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path, field=field)
    return strict_json_bytes(payload, field=field), payload


def _self_binding(
    path: Path,
    *,
    logical_id: str,
    self_field: str,
    field: str,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    document, payload = _load_json(path, field=field)
    canonical = document.get(self_field)
    if not isinstance(canonical, str) or canonical != canonical_json_hash(
        document, remove=self_field
    ):
        raise CommunityContractBuilderError(f"{field} self hash does not verify")
    return (
        document,
        payload,
        {
            "logical_id": logical_id,
            "file_sha256": sha256_bytes(payload),
            "canonical_sha256": canonical,
        },
    )


def _hash_binding(path: Path, *, logical_id: str, field: str) -> tuple[bytes, dict[str, str]]:
    payload = read_regular_file(path, field=field)
    if not payload:
        raise CommunityContractBuilderError(f"{field} is empty")
    return payload, {"logical_id": logical_id, "file_sha256": sha256_bytes(payload)}


def _replay_binding(path: Path, *, logical_id: str, field: str) -> dict[str, str]:
    document, payload = _load_json(path, field=field)
    canonical = document.get("receipt_sha256")
    replay = document.get("replay")
    if (
        not isinstance(canonical, str)
        or canonical != canonical_json_hash(document, remove="receipt_sha256")
        or not isinstance(replay, Mapping)
    ):
        raise CommunityContractBuilderError(f"{field} is not a self-bound replay receipt")
    implementation_file = replay.get("implementation_file_sha256")
    implementation_identity = replay.get("implementation_identity_sha256")
    for value in (implementation_file, implementation_identity):
        release._digest(value, field_name=f"{field} implementation")
    return {
        "logical_id": logical_id,
        "file_sha256": sha256_bytes(payload),
        "canonical_sha256": canonical,
        "implementation_file_sha256": implementation_file,
        "implementation_identity_sha256": implementation_identity,
    }


def _implementation_inventory(repository_root: Path) -> dict[str, dict[str, str]]:
    return {
        logical_id: {"logical_id": logical_id, "file_sha256": digest}
        for logical_id, digest in release._implementation_hashes(repository_root).items()
    }


def _require_exact_path_inventory(
    values: Mapping[str, Path], expected: Sequence[str], *, field: str
) -> None:
    if set(values) != set(expected):
        raise CommunityContractBuilderError(f"{field} inventory is not exact")


def build_contract(
    evidence: release.ReleaseEvidencePaths,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    required_paths = {
        "quality_receipt": evidence.quality_receipt,
        "quality_replay_receipt": evidence.quality_replay_receipt,
        "quality_native_replay": evidence.quality_native_replay,
        "final_model_manifest": evidence.final_model_manifest,
        "service_configuration_manifest": evidence.service_configuration_manifest,
        "pii_bench_posthoc_report": evidence.pii_bench_posthoc_report,
        "calibration_bundle": evidence.calibration_bundle,
        "service_source_manifest": evidence.service_source_manifest,
        "predecessor_service_source_manifest": evidence.predecessor_service_source_manifest,
        "internal_preopen_unlock": evidence.internal_preopen_unlock,
        "model_package_root": evidence.model_package_root,
        "release_wheel": evidence.release_wheel,
        "wheelhouse_root": evidence.wheelhouse_root,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise CommunityContractBuilderError(
            "complete contract inputs are missing: " + ",".join(sorted(missing))
        )
    _require_exact_path_inventory(evidence.candidate_manifests, release.TRACKS, field="candidate")
    _require_exact_path_inventory(evidence.comparator_manifests, release.TRACKS, field="comparator")
    _require_exact_path_inventory(
        evidence.performance_manifests, release.TRACKS, field="performance"
    )
    replay_keys = [f"{track}:{kind}" for track in release.TRACKS for kind in release.REPLAY_KINDS]
    _require_exact_path_inventory(evidence.replay_receipts, replay_keys, field="replay receipt")
    native_keys = [
        f"{track}:{kind}"
        for track in release.TRACKS
        for kind in (
            "candidate_prediction_full_replay",
            "comparator_generation_replay",
        )
    ]
    _require_exact_path_inventory(
        evidence.native_replay_evidence, native_keys, field="native replay"
    )
    _require_exact_path_inventory(
        evidence.release_artifacts, release.ARTIFACT_IDS, field="release artifact"
    )
    _require_exact_path_inventory(
        evidence.verification_receipts,
        release.VERIFICATION_IDS,
        field="verification receipt",
    )
    if evidence.publication_receipts:
        raise CommunityContractBuilderError(
            "pre-authorization contract builder refuses publication receipts"
        )

    assert evidence.quality_receipt is not None
    assert evidence.quality_replay_receipt is not None
    assert evidence.final_model_manifest is not None
    assert evidence.service_configuration_manifest is not None
    assert evidence.calibration_bundle is not None
    assert evidence.service_source_manifest is not None
    assert evidence.internal_preopen_unlock is not None

    quality_document, _quality_payload, quality_binding = _self_binding(
        evidence.quality_receipt,
        logical_id="quality_result",
        self_field="receipt_sha256",
        field="quality result receipt",
    )
    model, _model_payload, model_binding = _self_binding(
        evidence.final_model_manifest,
        logical_id="final_model_manifest",
        self_field="manifest_sha256",
        field="final model manifest",
    )
    service, _service_payload, service_binding = _self_binding(
        evidence.service_configuration_manifest,
        logical_id="service_configuration_manifest",
        self_field="manifest_sha256",
        field="service configuration manifest",
    )
    source, _source_payload, source_binding = _self_binding(
        evidence.service_source_manifest,
        logical_id="service_source_manifest",
        self_field="manifest_sha256",
        field="service source manifest",
    )
    internal_unlock, _unlock_payload, unlock_binding = _self_binding(
        evidence.internal_preopen_unlock,
        logical_id="internal_preopen_unlock",
        self_field="receipt_sha256",
        field="internal pre-open unlock",
    )
    calibration_payload, calibration_binding = _hash_binding(
        evidence.calibration_bundle,
        logical_id="calibration_bundle",
        field="calibration bundle",
    )
    if quality_document.get("reported_status") != "PASS":
        raise CommunityContractBuilderError("quality result is not a reported PASS")

    quality_replay_binding = _replay_binding(
        evidence.quality_replay_receipt,
        logical_id="quality_strict_replay",
        field="quality strict replay",
    )
    replay_inventory: dict[str, dict[str, dict[str, str]]] = {track: {} for track in release.TRACKS}
    for track in release.TRACKS:
        for kind in release.REPLAY_KINDS:
            key = f"{track}:{kind}"
            replay_inventory[track][kind] = _replay_binding(
                evidence.replay_receipts[key],
                logical_id=key.replace(":", "."),
                field=f"{key} replay receipt",
            )

    artifact_bindings: dict[str, dict[str, str]] = {}
    for artifact_id in release.ARTIFACT_IDS:
        if artifact_id == "final_model_manifest":
            artifact_bindings[artifact_id] = {
                key: model_binding[key] for key in ("logical_id", "file_sha256")
            }
        elif artifact_id == "service_configuration_manifest":
            artifact_bindings[artifact_id] = {
                key: service_binding[key] for key in ("logical_id", "file_sha256")
            }
        elif artifact_id == "service_source_manifest":
            artifact_bindings[artifact_id] = {
                key: source_binding[key] for key in ("logical_id", "file_sha256")
            }
        elif artifact_id == "benchmark_report":
            artifact_bindings[artifact_id] = {
                key: quality_binding[key] for key in ("logical_id", "file_sha256")
            }
        else:
            _payload, artifact_bindings[artifact_id] = _hash_binding(
                evidence.release_artifacts[artifact_id],
                logical_id=artifact_id,
                field=f"release artifact {artifact_id}",
            )

    verification_bindings: dict[str, dict[str, str]] = {}
    for check_id in release.VERIFICATION_IDS:
        _payload, verification_bindings[check_id] = _hash_binding(
            evidence.verification_receipts[check_id],
            logical_id=check_id,
            field=f"verification receipt {check_id}",
        )

    contract: dict[str, Any] = {
        "schema_version": release.CONTRACT_SCHEMA_VERSION,
        "contract_id": release.CONTRACT_ID,
        "status": "candidate_complete_pending_user_authorization",
        "purpose": "fail_closed_release_gate_for_open24_quality_and_independent_replays",
        "scope": release.EXPECTED_SCOPE,
        "quality_gate": {
            "result_receipt": quality_binding,
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
            "calibration_bundle_file_sha256": sha256_bytes(calibration_payload),
            "service_configuration_sha256": service["configuration_sha256"],
            "service_implementation_sha256": service["implementation_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "internal_unlock_sha256": internal_unlock["receipt_sha256"],
            "final_model_manifest": model_binding,
            "service_configuration_manifest": service_binding,
            "calibration_bundle": calibration_binding,
            "service_source_manifest": source_binding,
            "internal_preopen_unlock": unlock_binding,
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
        "implementation": _implementation_inventory(repository_root),
        "historical_policy": release.EXPECTED_HISTORICAL_POLICY,
        "contract_sha256": "",
    }
    contract["contract_sha256"] = canonical_json_hash(contract, remove="contract_sha256")

    with tempfile.TemporaryDirectory(prefix="pii-zh-community-contract-") as temporary_directory:
        temporary_contract = Path(temporary_directory) / "candidate-contract.json"
        temporary_contract.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = release.build_report(
            contract_path=temporary_contract,
            evidence=evidence,
            repository_root=repository_root,
        )
    if (
        report["status"] != "READY_FOR_USER_AUTHORIZATION"
        or report["local_candidate_complete"] is not True
    ):
        raise CommunityContractBuilderError(
            "derived contract did not reach exact READY_FOR_USER_AUTHORIZATION"
        )
    return contract


def publish_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CommunityContractBuilderError("refusing to overwrite completed contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise CommunityContractBuilderError("completed contract mode is not 0444")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_named(values: Sequence[str], *, field: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CommunityContractBuilderError(f"{field} must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in result:
            raise CommunityContractBuilderError(f"{field} contains an invalid entry")
        result[name] = Path(raw_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-receipt", required=True, type=Path)
    parser.add_argument("--quality-replay-receipt", required=True, type=Path)
    parser.add_argument("--quality-native-replay", required=True, type=Path)
    parser.add_argument("--final-model-manifest", required=True, type=Path)
    parser.add_argument("--service-configuration-manifest", required=True, type=Path)
    parser.add_argument("--pii-bench-posthoc-report", required=True, type=Path)
    parser.add_argument("--calibration-bundle", required=True, type=Path)
    parser.add_argument("--service-source-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-service-source-manifest", required=True, type=Path)
    parser.add_argument("--internal-preopen-unlock", required=True, type=Path)
    parser.add_argument("--candidate-manifest", action="append", required=True)
    parser.add_argument("--comparator-manifest", action="append", required=True)
    parser.add_argument("--performance-manifest", action="append", required=True)
    parser.add_argument("--replay-receipt", action="append", required=True)
    parser.add_argument("--native-replay-evidence", action="append", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--model-package-root", required=True, type=Path)
    parser.add_argument("--release-wheel", required=True, type=Path)
    parser.add_argument("--wheelhouse-root", required=True, type=Path)
    parser.add_argument("--verification", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evidence = release.ReleaseEvidencePaths(
            quality_receipt=args.quality_receipt,
            quality_replay_receipt=args.quality_replay_receipt,
            quality_native_replay=args.quality_native_replay,
            final_model_manifest=args.final_model_manifest,
            service_configuration_manifest=args.service_configuration_manifest,
            pii_bench_posthoc_report=args.pii_bench_posthoc_report,
            calibration_bundle=args.calibration_bundle,
            service_source_manifest=args.service_source_manifest,
            predecessor_service_source_manifest=args.predecessor_service_source_manifest,
            internal_preopen_unlock=args.internal_preopen_unlock,
            candidate_manifests=_parse_named(args.candidate_manifest, field="candidate manifest"),
            comparator_manifests=_parse_named(
                args.comparator_manifest, field="comparator manifest"
            ),
            performance_manifests=_parse_named(
                args.performance_manifest, field="performance manifest"
            ),
            replay_receipts=_parse_named(args.replay_receipt, field="replay receipt"),
            native_replay_evidence=_parse_named(
                args.native_replay_evidence, field="native replay evidence"
            ),
            release_artifacts=_parse_named(args.artifact, field="release artifact"),
            model_package_root=args.model_package_root,
            release_wheel=args.release_wheel,
            wheelhouse_root=args.wheelhouse_root,
            verification_receipts=_parse_named(args.verification, field="verification receipt"),
        )
        contract = build_contract(evidence)
        publish_contract(args.output, contract)
        print(
            json.dumps(
                {
                    "contract_id": contract["contract_id"],
                    "contract_sha256": contract["contract_sha256"],
                    "status": contract["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, CommunityContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
