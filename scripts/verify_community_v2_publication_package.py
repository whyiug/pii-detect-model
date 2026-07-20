#!/usr/bin/env python3
"""Verify a publication-successor model package locally or after immutable HF download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

try:
    from scripts import build_community_v2_publication_successor as successor
    from scripts import materialize_community_v2_hf_snapshot as hf_provenance
    from scripts import scan_public_artifacts
except ImportError:  # pragma: no cover - direct script execution
    import build_community_v2_publication_successor as successor  # type: ignore[no-redef]
    import materialize_community_v2_hf_snapshot as hf_provenance  # type: ignore[no-redef]
    import scan_public_artifacts  # type: ignore[no-redef]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "configs/release/community_v2_publication_package_verification.schema.json"
)
PACKAGE_VERSION = "0.2.0rc1"
CONTEXTS = frozenset({"local_publication_successor", "hugging_face_immutable_download"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_PROVENANCE_BYTES = 32 * 1024 * 1024
MAX_SMOKE_LOG_BYTES = 64 * 1024
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "community_v2_preauthorization.json",
        "token",
        "tokens",
        "credentials",
    }
)
EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
    "AutoModelForTokenClassification": (
        "modeling_qwen3_bi.Qwen3BiForTokenClassification"
    ),
}
EXPECTED_LABELS = (
    "O",
    "B-PERSON_NAME",
    "I-PERSON_NAME",
    "B-PHONE_NUMBER",
    "I-PHONE_NUMBER",
    "B-EMAIL_ADDRESS",
    "I-EMAIL_ADDRESS",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-DATE_OF_BIRTH",
    "I-DATE_OF_BIRTH",
    "B-CN_RESIDENT_ID",
    "I-CN_RESIDENT_ID",
    "B-PASSPORT_NUMBER",
    "I-PASSPORT_NUMBER",
    "B-DRIVER_LICENSE_NUMBER",
    "I-DRIVER_LICENSE_NUMBER",
    "B-SOCIAL_SECURITY_NUMBER",
    "I-SOCIAL_SECURITY_NUMBER",
    "B-BANK_CARD_NUMBER",
    "I-BANK_CARD_NUMBER",
    "B-BANK_ACCOUNT_NUMBER",
    "I-BANK_ACCOUNT_NUMBER",
    "B-VEHICLE_LICENSE_PLATE",
    "I-VEHICLE_LICENSE_PLATE",
    "B-EMPLOYEE_ID",
    "I-EMPLOYEE_ID",
    "B-STUDENT_ID",
    "I-STUDENT_ID",
    "B-MEDICAL_RECORD_NUMBER",
    "I-MEDICAL_RECORD_NUMBER",
    "B-WECHAT_ID",
    "I-WECHAT_ID",
    "B-QQ_NUMBER",
    "I-QQ_NUMBER",
    "B-ALIPAY_ACCOUNT",
    "I-ALIPAY_ACCOUNT",
    "B-USERNAME",
    "I-USERNAME",
    "B-IP_ADDRESS",
    "I-IP_ADDRESS",
    "B-MAC_ADDRESS",
    "I-MAC_ADDRESS",
    "B-DEVICE_ID",
    "I-DEVICE_ID",
    "B-GEO_COORDINATE",
    "I-GEO_COORDINATE",
    "B-SECRET",
    "I-SECRET",
)
EXPECTED_ID2LABEL = {str(index): label for index, label in enumerate(EXPECTED_LABELS)}
EXPECTED_LABEL2ID = {label: index for index, label in enumerate(EXPECTED_LABELS)}

MODEL_SMOKE_CODE = r"""
import json
import platform
import sys

sys.path.insert(0, sys.argv[2])

import torch
import transformers
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

root = sys.argv[1]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
tokenizer = AutoTokenizer.from_pretrained(
    root, local_files_only=True, trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    root, local_files_only=True, trust_remote_code=True
)
assert config.pii_release_eligible is False
assert config.pii_attention_mode == "full"
assert config.num_labels == 49
model = AutoModelForTokenClassification.from_pretrained(
    root, local_files_only=True, trust_remote_code=True
)
assert model.config.pii_release_eligible is False
assert model.config.pii_attention_mode == "full"
assert model.config.num_labels == 49
model.eval()
inputs = tokenizer("社区发布验证样例", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.ndim == 3
assert logits.shape[0] == 1
assert logits.shape[-1] == 49
assert torch.isfinite(logits).all().item()
print(json.dumps({
    "status": "PASS",
    "offline_model_load": True,
    "finite_forward": True,
    "logit_shape": list(logits.shape),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}, sort_keys=True))
""".strip()


class PublicationVerificationError(RuntimeError):
    """Raised when publication package verification fails closed."""


def _strict_json_bytes(payload: bytes, *, field: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationVerificationError(f"{field} repeats key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationVerificationError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise PublicationVerificationError(f"{field} must be a JSON object")
    return document


def _read_regular(path: Path, *, field: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationVerificationError(f"{field} is missing") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationVerificationError(
                f"{field} must be a regular non-symlink file"
            )
        if before.st_size <= 0 or before.st_size > maximum:
            raise PublicationVerificationError(f"{field} has an invalid size")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise PublicationVerificationError(f"{field} exceeds its size limit")
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_state = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            before_state != after_state
            or before_state != current_state
            or not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or total != before.st_size
        ):
            raise PublicationVerificationError(f"{field} changed while it was read")
        return b"".join(chunks)
    except PublicationVerificationError:
        raise
    except OSError as exc:
        raise PublicationVerificationError(f"{field} could not be read safely") from exc
    finally:
        os.close(descriptor)


def _load_schema() -> Mapping[str, Any]:
    return _strict_json_bytes(
        _read_regular(
            RECEIPT_SCHEMA_PATH,
            field="verification receipt schema",
            maximum=MAX_RECEIPT_BYTES,
        ),
        field="verification receipt schema",
    )


def _validate_receipt_document(document: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(_load_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise PublicationVerificationError(
            f"verification receipt schema failed at {location}: {first.message}"
        )
    if document.get("receipt_sha256") != successor.canonical_json_hash(
        document, remove="receipt_sha256"
    ):
        raise PublicationVerificationError("verification receipt self-hash failed")


def validate_receipt(path: Path) -> Mapping[str, Any]:
    document = _strict_json_bytes(
        _read_regular(path, field="verification receipt", maximum=MAX_RECEIPT_BYTES),
        field="verification receipt",
    )
    _validate_receipt_document(document)
    return document


def _validate_identities(
    *,
    context: str,
    source_commit: str,
    github_repository: str,
    hugging_face_repository: str,
    hugging_face_commit: str | None,
    hugging_face_download_provenance: Path | None,
) -> None:
    if context not in CONTEXTS:
        raise PublicationVerificationError("unknown verification context")
    if GIT_SHA_RE.fullmatch(source_commit) is None:
        raise PublicationVerificationError("source commit must be a full lowercase SHA-1")
    if REPO_ID_RE.fullmatch(github_repository) is None or REPO_ID_RE.fullmatch(
        hugging_face_repository
    ) is None:
        raise PublicationVerificationError("publication target repository ID is invalid")
    if context == "local_publication_successor" and (
        hugging_face_commit is not None or hugging_face_download_provenance is not None
    ):
        raise PublicationVerificationError(
            "local verification must not claim a HF commit or download provenance"
        )
    if context == "hugging_face_immutable_download" and (
        hugging_face_commit is None or GIT_SHA_RE.fullmatch(hugging_face_commit) is None
    ):
        raise PublicationVerificationError(
            "immutable HF verification requires a full lowercase commit SHA"
        )
    if (
        context == "hugging_face_immutable_download"
        and hugging_face_download_provenance is None
    ):
        raise PublicationVerificationError(
            "immutable HF verification requires reviewed download provenance"
        )


def _resolve_package_root(package_root: Path) -> tuple[Path, tuple[int, int]]:
    try:
        before = os.lstat(package_root)
    except OSError as exc:
        raise PublicationVerificationError("package root is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise PublicationVerificationError("package root must be a non-symlink directory")
    try:
        root = package_root.resolve(strict=True)
        resolved = os.lstat(root)
    except OSError as exc:
        raise PublicationVerificationError("package root could not be resolved") from exc
    identity = (before.st_dev, before.st_ino)
    if (
        stat.S_ISLNK(resolved.st_mode)
        or not stat.S_ISDIR(resolved.st_mode)
        or (resolved.st_dev, resolved.st_ino) != identity
    ):
        raise PublicationVerificationError("package root changed while it was resolved")
    return root, identity


def _verify_root_identity(root: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise PublicationVerificationError("package root disappeared during verification") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise PublicationVerificationError("package root changed during verification")


def _load_manifest(root: Path) -> Mapping[str, Any]:
    payload = _read_regular(
        root / successor.MANIFEST_NAME,
        field="publication successor manifest",
        maximum=MAX_RECEIPT_BYTES,
    )
    return _strict_json_bytes(payload, field="publication successor manifest")


def _verify_forbidden_files(root: Path) -> None:
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise PublicationVerificationError("publication package contains a symlink")
        relative = candidate.relative_to(root)
        if any(part in FORBIDDEN_NAMES or part == "wheelhouse" for part in relative.parts):
            raise PublicationVerificationError("publication package contains a forbidden path")
        if candidate.is_file() and candidate.suffix.lower() == ".whl":
            raise PublicationVerificationError("publication package contains a wheel")


def _verify_remote_code_contract(root: Path) -> None:
    config = _strict_json_bytes(
        _read_regular(root / "config.json", field="model config", maximum=4 * 1024 * 1024),
        field="model config",
    )
    labels = _strict_json_bytes(
        _read_regular(root / "id2label.json", field="label mapping", maximum=4 * 1024 * 1024),
        field="label mapping",
    )
    if config.get("auto_map") != EXPECTED_AUTO_MAP:
        raise PublicationVerificationError("model auto_map differs from the reviewed contract")
    if config.get("pii_release_eligible") is not False:
        raise PublicationVerificationError("candidate lineage release flag drifted")
    if config.get("pii_attention_mode") != "full":
        raise PublicationVerificationError("candidate attention mode drifted")
    if labels != EXPECTED_ID2LABEL:
        raise PublicationVerificationError(
            "label mapping differs from the reviewed core-24 BIO contract"
        )
    if config.get("id2label") != EXPECTED_ID2LABEL:
        raise PublicationVerificationError(
            "model config id2label differs from the reviewed label mapping"
        )
    if config.get("label2id") != EXPECTED_LABEL2ID:
        raise PublicationVerificationError(
            "model config label2id differs from the reviewed label mapping"
        )
    for name in ("configuration_qwen3_bi.py", "modeling_qwen3_bi.py"):
        _read_regular(root / name, field=f"reviewed remote code {name}", maximum=16 * 1024 * 1024)


def _scan_public_package(root: Path) -> None:
    findings = scan_public_artifacts.scan_paths([root])
    if findings:
        kinds = sorted({finding.kind for finding in findings})
        raise PublicationVerificationError(
            f"public artifact scan found {len(findings)} redacted finding(s): {kinds}"
        )


def _offline_environment(temp_root: Path) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
    }
    return environment


def _sandbox_environment() -> dict[str, str]:
    return {
        "HOME": "/scratch",
        "HF_HOME": "/scratch/huggingface",
        "XDG_CACHE_HOME": "/scratch/cache",
        "TMPDIR": "/scratch",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _run_model_smoke(root: Path) -> Mapping[str, Any]:
    bubblewrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    systemd_run = shutil.which("systemd-run")
    if bubblewrap is None or prlimit is None or systemd_run is None:
        raise PublicationVerificationError(
            "offline model smoke requires bwrap, prlimit, and systemd-run isolation"
        )
    runtime_prefix = Path(sys.base_prefix).resolve(strict=True)
    environment_prefix = Path(sys.prefix).resolve(strict=True)
    python_binary = Path(sys.executable).resolve(strict=True)
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    try:
        python_relative = python_binary.relative_to(runtime_prefix)
        purelib_relative = purelib.relative_to(environment_prefix)
    except ValueError as exc:
        raise PublicationVerificationError(
            "Python runtime paths cannot be isolated safely"
        ) from exc
    runtime_environment = _sandbox_environment()
    try:
        with tempfile.TemporaryDirectory(prefix="pii-publication-smoke-") as tempdir:
            command = [
                systemd_run,
                "--user",
                "--scope",
                "--quiet",
                "-p",
                "MemoryMax=16G",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "TasksMax=64",
                "-p",
                "CPUQuota=100%",
                "--",
                prlimit,
                "--as=34359738368",
                "--cpu=600",
                "--nofile=1024",
                "--core=0",
                "--",
                bubblewrap,
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--ro-bind",
                str(runtime_prefix),
                "/python",
                "--ro-bind",
                str(environment_prefix),
                "/venv",
                "--ro-bind",
                str(root),
                "/model",
                "--bind",
                tempdir,
                "/scratch",
                "--tmpfs",
                "/tmp",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--chdir",
                "/scratch",
                "--clearenv",
            ]
            for key, value in runtime_environment.items():
                command.extend(("--setenv", key, value))
            command.extend(
                (
                    f"/python/{python_relative.as_posix()}",
                    "-I",
                    "-c",
                    MODEL_SMOKE_CODE,
                    "/model",
                    f"/venv/{purelib_relative.as_posix()}",
                )
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_offline_environment(Path(tempdir)),
                cwd=REPOSITORY_ROOT,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=600)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise PublicationVerificationError(
                    "offline model smoke exceeded its wall-time limit"
                ) from exc
            completed = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
    except OSError as exc:
        raise PublicationVerificationError("offline model smoke could not complete") from exc
    if len(completed.stdout) > MAX_SMOKE_LOG_BYTES or len(completed.stderr) > MAX_SMOKE_LOG_BYTES:
        raise PublicationVerificationError("offline model smoke log exceeds the limit")
    if completed.returncode != 0:
        raise PublicationVerificationError(
            f"offline model smoke failed with return code {completed.returncode}"
        )
    output = _strict_json_bytes(completed.stdout, field="offline model smoke output")
    if (
        output.get("status") != "PASS"
        or output.get("offline_model_load") is not True
        or output.get("finite_forward") is not True
    ):
        raise PublicationVerificationError("offline model smoke did not report PASS")
    for field in ("python", "torch", "transformers"):
        value = output.get(field)
        if not isinstance(value, str) or not value or len(value) > 64:
            raise PublicationVerificationError("offline model smoke runtime identity is invalid")
    return output


def _sha256_regular(path: Path, *, field: str, maximum: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationVerificationError(f"{field} is missing") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationVerificationError(
                f"{field} must be a regular non-symlink file"
            )
        if before.st_size <= 0 or before.st_size > maximum:
            raise PublicationVerificationError(f"{field} has an invalid size")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or before_identity != current_identity
            or not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or total != before.st_size
        ):
            raise PublicationVerificationError(f"{field} changed while it was hashed")
        return digest.hexdigest()
    except PublicationVerificationError:
        raise
    except OSError as exc:
        raise PublicationVerificationError(f"{field} could not be hashed safely") from exc
    finally:
        os.close(descriptor)


def _verify_hf_download_provenance(
    *,
    root: Path,
    provenance_path: Path,
    repository: str,
    commit: str,
) -> dict[str, Any]:
    before_digest = _sha256_regular(
        provenance_path,
        field="HF download provenance",
        maximum=MAX_PROVENANCE_BYTES,
    )
    try:
        document = hf_provenance.load_and_validate_provenance(provenance_path)
    except hf_provenance.HfDownloadProvenanceError as exc:
        raise PublicationVerificationError(
            "HF download provenance did not validate"
        ) from exc
    after_digest = _sha256_regular(
        provenance_path,
        field="HF download provenance",
        maximum=MAX_PROVENANCE_BYTES,
    )
    if before_digest != after_digest:
        raise PublicationVerificationError(
            "HF download provenance changed during verification"
        )
    if (
        document.get("repository") != repository
        or document.get("requested_revision") != commit
        or document.get("resolved_commit") != commit
    ):
        raise PublicationVerificationError(
            "HF download provenance repository or revision does not match"
        )
    remote_snapshot = document["remote_snapshot"]
    remote_coverage = remote_snapshot["metadata_coverage"]
    if (
        remote_coverage["size_count"] != remote_snapshot["file_count"]
        or remote_coverage["content_verified_count"]
        != remote_snapshot["file_count"]
    ):
        raise PublicationVerificationError(
            "HF download provenance lacks complete remote content verification"
        )
    try:
        hf_provenance.verify_local_root_binding(root, document)
    except hf_provenance.HfDownloadProvenanceError as exc:
        raise PublicationVerificationError(
            "HF download provenance does not bind the package"
        ) from exc
    local_root = document["local_root"]
    generator = document["generator"]
    return {
        "file_sha256": before_digest,
        "receipt_sha256": document["receipt_sha256"],
        "repository": document["repository"],
        "requested_revision": document["requested_revision"],
        "resolved_commit": document["resolved_commit"],
        "remote_visibility": remote_snapshot["visibility"],
        "remote_file_count": remote_snapshot["file_count"],
        "remote_metadata_inventory_sha256": remote_snapshot[
            "metadata_inventory_sha256"
        ],
        "remote_content_verified_count": remote_coverage[
            "content_verified_count"
        ],
        "local_root_inventory_sha256": local_root["inventory_sha256"],
        "generator_file_sha256": generator["file_sha256"],
    }


def verify_package(
    *,
    package_root: Path,
    context: str,
    source_commit: str,
    github_repository: str,
    hugging_face_repository: str,
    hugging_face_commit: str | None,
    hugging_face_download_provenance: Path | None = None,
    verified_at: str | None = None,
) -> Mapping[str, Any]:
    _validate_identities(
        context=context,
        source_commit=source_commit,
        github_repository=github_repository,
        hugging_face_repository=hugging_face_repository,
        hugging_face_commit=hugging_face_commit,
        hugging_face_download_provenance=hugging_face_download_provenance,
    )
    root, root_identity = _resolve_package_root(package_root)

    static_result_before = successor.verify_successor_package(root)
    manifest = _load_manifest(root)
    if manifest.get("package_version") != PACKAGE_VERSION:
        raise PublicationVerificationError("publication package version drifted")
    if manifest.get("source_control", {}).get("git_source_commit") != source_commit:
        raise PublicationVerificationError("publication package source commit does not match")
    targets = manifest.get("publication_targets", {})
    if (
        targets.get("github_repository") != github_repository
        or targets.get("hugging_face_repository") != hugging_face_repository
    ):
        raise PublicationVerificationError("publication package target does not match")

    hf_binding: dict[str, Any] | None = None
    if context == "hugging_face_immutable_download":
        assert hugging_face_commit is not None
        assert hugging_face_download_provenance is not None
        hf_binding = _verify_hf_download_provenance(
            root=root,
            provenance_path=hugging_face_download_provenance,
            repository=hugging_face_repository,
            commit=hugging_face_commit,
        )
    _verify_forbidden_files(root)
    _verify_remote_code_contract(root)
    _scan_public_package(root)
    smoke = _run_model_smoke(root)

    _verify_root_identity(root, root_identity)
    if context == "hugging_face_immutable_download":
        assert hugging_face_commit is not None
        assert hugging_face_download_provenance is not None
        hf_binding_after = _verify_hf_download_provenance(
            root=root,
            provenance_path=hugging_face_download_provenance,
            repository=hugging_face_repository,
            commit=hugging_face_commit,
        )
        if hf_binding_after != hf_binding:
            raise PublicationVerificationError(
                "HF download provenance changed during model smoke"
            )
    _verify_forbidden_files(root)
    _verify_remote_code_contract(root)
    _scan_public_package(root)
    model_digest = _sha256_regular(
        root / "model.safetensors",
        field="model weights",
        maximum=16 * 1024 * 1024 * 1024,
    )
    if SHA256_RE.fullmatch(model_digest) is None:
        raise PublicationVerificationError("model digest is invalid")
    try:
        static_result_after = successor.verify_successor_package(root)
    except successor.PublicationSuccessorError as exc:
        raise PublicationVerificationError(
            "publication package failed post-smoke closure verification"
        ) from exc
    if static_result_after != static_result_before or _load_manifest(root) != manifest:
        raise PublicationVerificationError("publication package changed during model smoke")
    if model_digest != manifest["payload_files"]["model.safetensors"]["file_sha256"]:
        raise PublicationVerificationError("model digest differs from the manifest binding")
    _verify_root_identity(root, root_identity)

    timestamp = verified_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    receipt: dict[str, Any] = {
        "schema_version": "pii-zh.community-v2-publication-package-verification.v1",
        "context": context,
        "package_version": PACKAGE_VERSION,
        "target": {
            "github_repository": github_repository,
            "hugging_face_repository": hugging_face_repository,
            "hugging_face_commit": hugging_face_commit,
        },
        "source_control": {"git_source_commit": source_commit},
        "hf_download_provenance": hf_binding,
        "package_identity": {
            "manifest_sha256": manifest["manifest_sha256"],
            "checksums_file_sha256": static_result_before["checksums_file_sha256"],
            "payload_inventory_sha256": manifest["payload_inventory_sha256"],
            "model_file_sha256": model_digest,
            "verified_file_count": static_result_before["verified_file_count"],
        },
        "checks": {
            "checksum_closure": True,
            "manifest_schema_and_self_hash": True,
            "target_and_source_binding": True,
            "hf_download_provenance_binding": True,
            "forbidden_file_absence": True,
            "remote_code_contract": True,
            "public_artifact_scan": True,
            "post_smoke_reverification": True,
            "offline_model_load": True,
            "finite_forward": True,
        },
        "runtime": {
            "python": smoke.get("python", platform.python_version()),
            "torch": smoke["torch"],
            "transformers": smoke["transformers"],
            "network_disabled": True,
            "cuda_visible_devices": "",
        },
        "status": "PASS",
        "verified_at": timestamp,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = successor.canonical_json_hash(
        receipt, remove="receipt_sha256"
    )
    _validate_receipt_document(receipt)
    return receipt


def _write_new_receipt(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        parent_metadata = os.lstat(path.parent)
    except OSError as exc:
        raise PublicationVerificationError(
            "verification receipt output parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise PublicationVerificationError(
            "verification receipt output parent must be a real directory"
        )
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    parent_descriptor = os.open(path.parent, parent_flags)
    parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
    if (os.fstat(parent_descriptor).st_dev, os.fstat(parent_descriptor).st_ino) != (
        parent_identity
    ):
        os.close(parent_descriptor)
        raise PublicationVerificationError(
            "verification receipt output parent changed during validation"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.name, flags, 0o444, dir_fd=parent_descriptor)
    except FileExistsError as exc:
        os.close(parent_descriptor)
        raise PublicationVerificationError("verification receipt output already exists") from exc
    except OSError as exc:
        os.close(parent_descriptor)
        raise PublicationVerificationError(
            "verification receipt output cannot be created safely"
        ) from exc
    created = os.fstat(descriptor)
    created_identity = (created.st_dev, created.st_ino)

    def remove_owned_output() -> None:
        try:
            current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError:
            return
        if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (
            created_identity
        ):
            try:
                os.unlink(path.name, dir_fd=parent_descriptor)
            except OSError:
                pass

    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short verification receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        final_descriptor_metadata = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        remove_owned_output()
        os.close(parent_descriptor)
        raise
    os.close(descriptor)
    try:
        final_path_metadata = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        final_parent_metadata = os.lstat(path.parent)
    except OSError as exc:
        remove_owned_output()
        os.close(parent_descriptor)
        raise PublicationVerificationError(
            "verification receipt output could not be finalized"
        ) from exc
    valid = (
        stat.S_ISREG(final_path_metadata.st_mode)
        and (final_path_metadata.st_dev, final_path_metadata.st_ino) == created_identity
        and (final_descriptor_metadata.st_dev, final_descriptor_metadata.st_ino)
        == created_identity
        and stat.S_IMODE(final_path_metadata.st_mode) == 0o444
        and final_path_metadata.st_size == len(payload)
        and not stat.S_ISLNK(final_parent_metadata.st_mode)
        and (final_parent_metadata.st_dev, final_parent_metadata.st_ino) == parent_identity
    )
    if not valid:
        remove_owned_output()
        os.close(parent_descriptor)
        raise PublicationVerificationError(
            "verification receipt output identity, mode, or size is invalid"
        )
    os.close(parent_descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--package-root", required=True, type=Path)
    verify.add_argument("--context", required=True, choices=sorted(CONTEXTS))
    verify.add_argument("--source-commit", required=True)
    verify.add_argument("--github-repository", required=True)
    verify.add_argument("--hugging-face-repository", required=True)
    verify.add_argument("--hugging-face-commit")
    verify.add_argument("--hugging-face-download-provenance", type=Path)
    verify.add_argument("--output", required=True, type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", required=True, type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "validate":
            document = validate_receipt(args.receipt)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "context": document["context"],
                        "receipt_sha256": document["receipt_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        document = verify_package(
            package_root=args.package_root,
            context=args.context,
            source_commit=args.source_commit,
            github_repository=args.github_repository,
            hugging_face_repository=args.hugging_face_repository,
            hugging_face_commit=args.hugging_face_commit,
            hugging_face_download_provenance=args.hugging_face_download_provenance,
        )
        _write_new_receipt(args.output, document)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "context": document["context"],
                    "receipt_sha256": document["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        PublicationVerificationError,
        successor.PublicationSuccessorError,
    ) as exc:
        print(f"publication package verification blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
