#!/usr/bin/env python3
"""Audit every installed entry point and the complete declared wheel source set.

This successor is local, read-only, CPU-only, and network-free.  It proves a
bounded execution-surface property only; it does not replace a current source
closure, clean-wheel execution, hosted CI, quality, approval, or publication
gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

try:  # pragma: no cover - Python 3.10 fallback
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs/release/release_execution_surface_v2.json"
DEFAULT_SCHEMA = REPOSITORY_ROOT / "configs/release/release_execution_surface_v2.schema.json"
REPORT_SCHEMA_VERSION = "pii-zh.release-execution-surface-report.v2"
REPORT_TYPE = "release_execution_surface_policy_receipt"
PASS_STATUS = "PASS"
MAX_FILE_BYTES = 64 * 1024 * 1024
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENTRYPOINT_TARGET = re.compile(
    r"(?P<module>[A-Za-z_][A-Za-z0-9_.]*):(?P<callable>[A-Za-z_][A-Za-z0-9_]*)\Z"
)
_FIXED_PHYSICAL_GPU_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cuda_visible_devices_assignment_zero",
        re.compile(r"\bCUDA_VISIBLE_DEVICES\s*=\s*[\"']?0(?:[\"']|\b)"),
    ),
    (
        "cuda_visible_devices_mapping_zero",
        re.compile(r"[\"']CUDA_VISIBLE_DEVICES[\"']\s*:\s*[\"']0[\"']"),
    ),
    (
        "cuda_visible_devices_comparison_zero",
        re.compile(
            r"(?:get\(\s*[\"']CUDA_VISIBLE_DEVICES[\"']\s*\)|"
            r"environ\[\s*[\"']CUDA_VISIBLE_DEVICES[\"']\s*\])"
            r"\s*(?:==|!=)\s*[\"']0[\"']"
        ),
    ),
    (
        "physical_gpu_id_zero",
        re.compile(r"(?:[\"']physical_gpu_id[\"']|\bphysical_gpu_id)\s*[:=]\s*0\b"),
    ),
)
_PRIVATE_ABSOLUTE_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unix_private_mount_or_user",
        re.compile(r"(?<![A-Za-z0-9_.-])/(?:data[0-9]*|home|Users)/[A-Za-z0-9._~-]+(?:/|\b)"),
    ),
    ("windows_user", re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._ ~-]+\\")),
)


class ReleaseExecutionSurfaceV2Error(RuntimeError):
    """Raised when the successor surface cannot be checked safely."""


def canonical_json_hash(value: object, *, remove: str | None = None) -> str:
    document: object = value
    if remove is not None:
        if not isinstance(value, Mapping):
            raise ReleaseExecutionSurfaceV2Error("hash removal requires a JSON object")
        document = dict(value)
        document.pop(remove, None)
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReleaseExecutionSurfaceV2Error("document is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ReleaseExecutionSurfaceV2Error(f"{field} must be a repository-relative path")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReleaseExecutionSurfaceV2Error(f"{field} must be a safe repository-relative path")
    return value


def _read_descriptor(descriptor: int, *, field: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ReleaseExecutionSurfaceV2Error(f"{field} is not a bounded regular file")
    payload = bytearray()
    while block := os.read(descriptor, 1024 * 1024):
        payload.extend(block)
        if len(payload) > MAX_FILE_BYTES:
            raise ReleaseExecutionSurfaceV2Error(f"{field} exceeds the file-size limit")
    after = os.fstat(descriptor)
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
    if before_identity != after_identity or len(payload) != after.st_size:
        raise ReleaseExecutionSurfaceV2Error(f"{field} changed while it was read")
    return bytes(payload)


def _read_regular(path: Path, *, field: str, required_mode: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseExecutionSurfaceV2Error(f"{field} is unavailable") from exc
    try:
        if required_mode is not None:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != required_mode
            ):
                raise ReleaseExecutionSurfaceV2Error(
                    f"{field} must be a regular file with mode {required_mode:04o}"
                )
        return _read_descriptor(descriptor, field=field)
    finally:
        os.close(descriptor)


def _require_read_only_regular(path: Path, *, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReleaseExecutionSurfaceV2Error(f"{field} is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        raise ReleaseExecutionSurfaceV2Error(
            f"{field} must be a read-only regular non-symlink file with mode 0444"
        )


def _read_repository_file(
    root: Path,
    relative: object,
    *,
    field: str,
    required_mode: int | None = None,
) -> bytes:
    safe = _safe_relative_path(relative, field=field)
    current = root
    for part in PurePosixPath(safe).parts[:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ReleaseExecutionSurfaceV2Error(f"{field} has an unavailable ancestor") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseExecutionSurfaceV2Error(
                f"{field} traverses a symlink or non-directory ancestor"
            )
    return _read_regular(
        root.joinpath(*PurePosixPath(safe).parts),
        field=field,
        required_mode=required_mode,
    )


def _json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseExecutionSurfaceV2Error(f"{field} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseExecutionSurfaceV2Error(f"{field} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseExecutionSurfaceV2Error(f"{field} must be a JSON object")
    return value


def _load_json_path(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, field=field)
    return _json_object(payload, field=field), payload


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _text(payload: bytes, *, field: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeError as exc:
        raise ReleaseExecutionSurfaceV2Error(f"{field} is not UTF-8 text") from exc


def _parse_toml(payload: bytes) -> dict[str, Any]:
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseExecutionSurfaceV2Error("pyproject is not valid UTF-8 TOML") from exc
    if not isinstance(value, dict):
        raise ReleaseExecutionSurfaceV2Error("pyproject root must be a table")
    return value


def _verify_binding(root: Path, binding: Mapping[str, Any], *, field: str) -> tuple[str, bytes]:
    path = _safe_relative_path(binding.get("path"), field=f"{field}.path")
    expected = binding.get("file_sha256")
    if not isinstance(expected, str) or _HEX_SHA256.fullmatch(expected) is None:
        raise ReleaseExecutionSurfaceV2Error(f"{field}.file_sha256 is invalid")
    payload = _read_repository_file(root, path, field=field)
    if _sha256_bytes(payload) != expected:
        raise ReleaseExecutionSurfaceV2Error(f"{field} content hash does not match")
    return path, payload


def _load_source_closure_verifier(path: Path, *, expected_repository_path: str) -> Any:
    module_name = "_pii_zh_successor_source_closure_v8_verifier"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReleaseExecutionSurfaceV2Error("source closure verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ReleaseExecutionSurfaceV2Error(
            "source closure verifier module initialization failed"
        ) from exc
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    origin = getattr(module, "__file__", None)
    verify = getattr(module, "verify_closure", None)
    if (
        not isinstance(origin, str)
        or Path(origin).resolve(strict=True) != path.resolve(strict=True)
        or not path.as_posix().endswith(expected_repository_path)
        or not callable(verify)
    ):
        raise ReleaseExecutionSurfaceV2Error(
            "source closure verifier origin or Python API is invalid"
        )
    return verify


def _module_name(path: str) -> str:
    relative = PurePosixPath(path).relative_to("src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_index(python_paths: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    module_to_path = {_module_name(path): path for path in python_paths}
    if len(module_to_path) != len(python_paths):
        raise ReleaseExecutionSurfaceV2Error("package module inventory is ambiguous")
    return module_to_path, {path: module for module, path in module_to_path.items()}


def _module_and_parents(module: str, module_to_path: Mapping[str, str]) -> set[str]:
    result: set[str] = set()
    parts = module.split(".")
    for size in range(1, len(parts) + 1):
        candidate = ".".join(parts[:size])
        if candidate in module_to_path:
            result.add(candidate)
    return result


def _resolve_imports(
    tree: ast.AST,
    *,
    current_module: str,
    current_is_package: bool,
    module_to_path: Mapping[str, str],
) -> set[str]:
    resolved: set[str] = set()
    package = current_module if current_is_package else current_module.rpartition(".")[0]
    for node in ast.walk(tree):
        candidates: set[str] = set()
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                trim = node.level - 1
                if trim > len(package_parts):
                    continue
                base_parts = package_parts[: len(package_parts) - trim]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            candidates.add(base)
            candidates.update(
                f"{base}.{alias.name}" if base else alias.name for alias in node.names
            )
        for candidate in candidates:
            if candidate in module_to_path:
                resolved.update(_module_and_parents(candidate, module_to_path))
    return resolved


def _parse_python(payload: bytes, *, path: str) -> ast.Module:
    try:
        return ast.parse(_text(payload, field=path), filename=path)
    except SyntaxError as exc:
        raise ReleaseExecutionSurfaceV2Error(f"{path} is not valid Python") from exc


def _runtime_import_closure(
    *,
    root_modules: Sequence[str],
    module_to_path: Mapping[str, str],
    payloads: Mapping[str, bytes],
) -> tuple[list[str], dict[str, ast.Module]]:
    unknown = sorted(set(root_modules) - set(module_to_path))
    if unknown:
        raise ReleaseExecutionSurfaceV2Error(f"runtime root modules are absent: {unknown}")
    queue: deque[str] = deque()
    for module in root_modules:
        queue.extend(sorted(_module_and_parents(module, module_to_path)))
    visited: set[str] = set()
    trees: dict[str, ast.Module] = {}
    while queue:
        module = queue.popleft()
        if module in visited:
            continue
        visited.add(module)
        path = module_to_path[module]
        tree = _parse_python(payloads[path], path=path)
        trees[path] = tree
        is_package = path.endswith("/__init__.py") or path == "src/pii_zh/__init__.py"
        for imported in sorted(
            _resolve_imports(
                tree,
                current_module=module,
                current_is_package=is_package,
                module_to_path=module_to_path,
            )
        ):
            if imported not in visited:
                queue.append(imported)
    return sorted(trees), trees


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return names


def _dynamic_imports(tree: ast.Module) -> list[str | None]:
    importlib_aliases = {"importlib"}
    direct_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    direct_aliases.add(alias.asname or alias.name)
    values: list[str | None] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dynamic = False
        if isinstance(node.func, ast.Name):
            dynamic = node.func.id in direct_aliases or node.func.id == "__import__"
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
            dynamic = isinstance(node.func.value, ast.Name) and (
                node.func.value.id in importlib_aliases
            )
        if not dynamic:
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            values.append(node.args[0].value)
        else:
            values.append(None)
    return values


def _pattern_counts(
    files: Mapping[str, bytes], patterns: Sequence[tuple[str, re.Pattern[str]]]
) -> tuple[Counter[str], set[str]]:
    counts: Counter[str] = Counter()
    affected: set[str] = set()
    for path, payload in files.items():
        content = _text(payload, field=path)
        for pattern_id, pattern in patterns:
            matches = list(pattern.finditer(content))
            if matches:
                counts[pattern_id] += len(matches)
                affected.add(path)
    return counts, affected


def _package_data_configuration(pyproject: Mapping[str, Any]) -> dict[str, list[str]]:
    try:
        raw = pyproject["tool"]["setuptools"]["package-data"]
    except (KeyError, TypeError) as exc:
        raise ReleaseExecutionSurfaceV2Error("setuptools package-data table is absent") from exc
    if not isinstance(raw, Mapping):
        raise ReleaseExecutionSurfaceV2Error("setuptools package-data must be a table")
    result: dict[str, list[str]] = {}
    for package, patterns in raw.items():
        if (
            not isinstance(package, str)
            or not isinstance(patterns, list)
            or not all(isinstance(pattern, str) and pattern for pattern in patterns)
        ):
            raise ReleaseExecutionSurfaceV2Error("setuptools package-data is malformed")
        result[package] = list(patterns)
    return result


def _discover_wheel_inventory(
    root: Path, *, package_root: str, package_data: Mapping[str, Sequence[str]]
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[str], list[str]]:
    safe_root = _safe_relative_path(package_root, field="whole_wheel.package_root")
    base = root.joinpath(*PurePosixPath(safe_root).parts)
    if base.is_symlink() or not base.is_dir():
        raise ReleaseExecutionSurfaceV2Error("package root is not a regular directory")
    python_paths: list[str] = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        _read_repository_file(root, relative, field=f"wheel Python file {relative}")
        python_paths.append(relative)

    data_paths: set[str] = set()
    for package, patterns in sorted(package_data.items()):
        if package != "pii_zh" and not package.startswith("pii_zh."):
            raise ReleaseExecutionSurfaceV2Error("package-data escapes the pii_zh package")
        package_directory = root / "src" / Path(*package.split("."))
        if package_directory.is_symlink() or not package_directory.is_dir():
            raise ReleaseExecutionSurfaceV2Error("package-data package directory is unavailable")
        for pattern in patterns:
            pure = PurePosixPath(pattern)
            if pure.is_absolute() or "\\" in pattern or ".." in pure.parts:
                raise ReleaseExecutionSurfaceV2Error("package-data pattern is unsafe")
            matches = sorted(package_directory.glob(pattern))
            if not matches:
                raise ReleaseExecutionSurfaceV2Error("package-data pattern matches no files")
            for path in matches:
                relative = path.relative_to(root).as_posix()
                _read_repository_file(root, relative, field=f"wheel package-data {relative}")
                data_paths.add(relative)

    overlap = set(python_paths) & data_paths
    if overlap:
        raise ReleaseExecutionSurfaceV2Error("Python and package-data inventories overlap")
    payloads: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    for kind, paths in (("python", python_paths), ("package_data", sorted(data_paths))):
        for path in paths:
            payload = _read_repository_file(root, path, field=f"wheel inventory {path}")
            payloads[path] = payload
            inventory.append(
                {
                    "path": path,
                    "kind": kind,
                    "size_bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
    inventory.sort(key=lambda item: item["path"])
    return inventory, payloads, python_paths, sorted(data_paths)


def wheel_inventory_sha256(inventory: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_hash({"files": list(inventory)})


def _pass_check(**values: Any) -> dict[str, Any]:
    return {"status": "PASS", "reason_codes": [], **values}


def _blocked_check(reason_codes: Sequence[str], **values: Any) -> dict[str, Any]:
    reasons = sorted(set(reason_codes))
    if not reasons:
        raise ReleaseExecutionSurfaceV2Error("BLOCKED checks require a reason code")
    return {"status": "BLOCKED", "reason_codes": reasons, **values}


def build_report(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    schema_path: Path = DEFAULT_SCHEMA,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    contract_path = _absolute_without_symlink_resolution(contract_path)
    schema_path = _absolute_without_symlink_resolution(schema_path)
    try:
        contract_relative = contract_path.relative_to(root).as_posix()
        schema_relative = schema_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseExecutionSurfaceV2Error(
            "contract and schema must be regular files inside the repository"
        ) from exc
    contract_payload = _read_repository_file(
        root, contract_relative, field="contract", required_mode=0o444
    )
    contract = _json_object(contract_payload, field="contract")
    schema_payload = _read_repository_file(root, schema_relative, field="schema")
    schema = _json_object(schema_payload, field="schema")
    Draft202012Validator.check_schema(schema)
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.path)
    )
    if schema_errors:
        locations = ["/".join(str(part) for part in error.absolute_path) for error in schema_errors]
        raise ReleaseExecutionSurfaceV2Error(f"contract schema validation failed at {locations}")
    observed_contract_hash = canonical_json_hash(contract, remove="contract_sha256")
    if contract.get("contract_sha256") != observed_contract_hash:
        raise ReleaseExecutionSurfaceV2Error("contract canonical hash does not match")

    implementation = contract["implementation"]
    validator_path, validator_payload = _verify_binding(
        root, implementation["validator"], field="implementation.validator"
    )
    schema_binding = implementation["schema"]
    if schema_binding["path"] != schema_relative:
        raise ReleaseExecutionSurfaceV2Error("schema path does not match its binding")
    if _sha256_bytes(schema_payload) != schema_binding["file_sha256"]:
        raise ReleaseExecutionSurfaceV2Error("schema hash does not match its binding")

    checks: dict[str, dict[str, Any]] = {}
    checks["contract_identity"] = _pass_check(
        contract_canonical_sha256=observed_contract_hash,
        contract_file_sha256=_sha256_bytes(contract_payload),
        validator_path=validator_path,
        validator_file_sha256=_sha256_bytes(validator_payload),
        schema_file_sha256=_sha256_bytes(schema_payload),
    )

    manifest_path, pyproject_payload = _verify_binding(
        root, contract["package_manifest"], field="package_manifest"
    )
    pyproject = _parse_toml(pyproject_payload)
    try:
        raw_scripts = pyproject["project"]["scripts"]
    except (KeyError, TypeError) as exc:
        raise ReleaseExecutionSurfaceV2Error("project.scripts is absent") from exc
    if not isinstance(raw_scripts, Mapping) or not all(
        isinstance(name, str) and isinstance(target, str) for name, target in raw_scripts.items()
    ):
        raise ReleaseExecutionSurfaceV2Error("project.scripts is malformed")
    project_scripts = dict(raw_scripts)
    declared_entrypoints = {
        item["name"]: item["target"] for item in contract["installed_entrypoints"]
    }
    entrypoint_reasons: list[str] = []
    if project_scripts != declared_entrypoints:
        entrypoint_reasons.append("project_scripts_not_exactly_enumerated")
    parsed_entrypoints: list[tuple[str, str, str]] = []
    for name, target in sorted(project_scripts.items()):
        match = _ENTRYPOINT_TARGET.fullmatch(target)
        if match is None:
            entrypoint_reasons.append("installed_entrypoint_target_is_not_static")
            continue
        parsed_entrypoints.append((name, match["module"], match["callable"]))

    package_data = _package_data_configuration(pyproject)
    configured_package_data = contract["package_data_boundary"]["declared_patterns"]
    if package_data != configured_package_data:
        entrypoint_reasons.append("package_data_declaration_changed")

    inventory, wheel_payloads, python_paths, package_data_paths = _discover_wheel_inventory(
        root,
        package_root=contract["whole_wheel"]["package_root"],
        package_data=package_data,
    )
    module_to_path, _ = _module_index(python_paths)
    entrypoint_callable_failures = 0
    for _, module, callable_name in parsed_entrypoints:
        path = module_to_path.get(module)
        if path is None or callable_name not in _top_level_names(
            _parse_python(wheel_payloads[path], path=path)
        ):
            entrypoint_callable_failures += 1
    if entrypoint_callable_failures:
        entrypoint_reasons.append("installed_entrypoint_callable_not_statically_present")
    checks["installed_entrypoints"] = (
        _pass_check(
            package_manifest_path=manifest_path,
            project_script_count=len(project_scripts),
            declared_entrypoint_count=len(declared_entrypoints),
            missing_or_extra_entrypoint_count=0,
            callable_failure_count=0,
        )
        if not entrypoint_reasons
        else _blocked_check(
            entrypoint_reasons,
            package_manifest_path=manifest_path,
            project_script_count=len(project_scripts),
            declared_entrypoint_count=len(declared_entrypoints),
            missing_or_extra_entrypoint_count=len(
                set(project_scripts.items()) ^ set(declared_entrypoints.items())
            ),
            callable_failure_count=entrypoint_callable_failures,
        )
    )

    service_reasons: list[str] = []
    service_modules: list[str] = []
    for api in contract["selected_service_apis"]:
        module = api["module"]
        path = module_to_path.get(module)
        if path != api["path"]:
            service_reasons.append("selected_service_api_module_path_changed")
            continue
        names = _top_level_names(_parse_python(wheel_payloads[path], path=path))
        if not set(api["required_exports"]).issubset(names):
            service_reasons.append("selected_service_api_export_missing")
        service_modules.append(module)
    checks["selected_service_apis"] = (
        _pass_check(
            selected_api_module_count=len(service_modules),
            required_export_count=sum(
                len(api["required_exports"]) for api in contract["selected_service_apis"]
            ),
            missing_export_count=0,
        )
        if not service_reasons
        else _blocked_check(
            service_reasons,
            selected_api_module_count=len(service_modules),
            required_export_count=sum(
                len(api["required_exports"]) for api in contract["selected_service_apis"]
            ),
            missing_export_count=1,
        )
    )

    inventory_hash = wheel_inventory_sha256(inventory)
    expected_wheel = contract["whole_wheel"]
    wheel_reasons: list[str] = []
    if inventory_hash != expected_wheel["inventory_sha256"]:
        wheel_reasons.append("whole_wheel_inventory_hash_changed")
    if len(inventory) != expected_wheel["inventory_file_count"]:
        wheel_reasons.append("whole_wheel_inventory_count_changed")
    if len(python_paths) != expected_wheel["python_file_count"]:
        wheel_reasons.append("whole_wheel_python_count_changed")
    if len(package_data_paths) != expected_wheel["package_data_file_count"]:
        wheel_reasons.append("whole_wheel_package_data_count_changed")
    checks["whole_wheel_inventory"] = (
        _pass_check(
            inventory_sha256=inventory_hash,
            inventory_file_count=len(inventory),
            python_file_count=len(python_paths),
            package_data_file_count=len(package_data_paths),
        )
        if not wheel_reasons
        else _blocked_check(
            wheel_reasons,
            inventory_sha256=inventory_hash,
            inventory_file_count=len(inventory),
            python_file_count=len(python_paths),
            package_data_file_count=len(package_data_paths),
        )
    )

    runtime_roots = [module for _, module, _ in parsed_entrypoints] + service_modules
    runtime_paths, runtime_trees = _runtime_import_closure(
        root_modules=runtime_roots,
        module_to_path=module_to_path,
        payloads=wheel_payloads,
    )
    entrypoint_modules = {module for _, module, _ in parsed_entrypoints}
    second_entrypoint = "pii_zh.evaluation.cascade_ablation_cli"
    second_path = module_to_path.get(second_entrypoint)
    closure_reasons: list[str] = []
    if second_entrypoint not in entrypoint_modules or second_path not in runtime_paths:
        closure_reasons.append("ablation_entrypoint_not_in_runtime_import_closure")
    checks["runtime_import_closure"] = (
        _pass_check(
            root_module_count=len(set(runtime_roots)),
            installed_entrypoint_root_count=len(entrypoint_modules),
            runtime_import_closure_file_count=len(runtime_paths),
            ablation_entrypoint_included=True,
            conservative_parent_package_imports_included=True,
        )
        if not closure_reasons
        else _blocked_check(
            closure_reasons,
            root_module_count=len(set(runtime_roots)),
            installed_entrypoint_root_count=len(entrypoint_modules),
            runtime_import_closure_file_count=len(runtime_paths),
            ablation_entrypoint_included=False,
            conservative_parent_package_imports_included=True,
        )
    )

    dynamic_targets: Counter[str] = Counter()
    opaque_dynamic_count = 0
    project_dynamic_paths: set[str] = set()
    for path in python_paths:
        tree = _parse_python(wheel_payloads[path], path=path)
        for target in _dynamic_imports(tree):
            if target is None:
                opaque_dynamic_count += 1
            else:
                dynamic_targets[target] += 1
                if target == "pii_zh" or target.startswith("pii_zh."):
                    project_dynamic_paths.add(path)
    dynamic_policy = contract["dynamic_import_boundary"]
    allowed_external = set(dynamic_policy["allowed_constant_external_targets"])
    observed_external = {
        target
        for target in dynamic_targets
        if target != "pii_zh" and not target.startswith("pii_zh.")
    }
    dynamic_reasons: list[str] = []
    if opaque_dynamic_count:
        dynamic_reasons.append("opaque_dynamic_import_present")
    if observed_external != allowed_external:
        dynamic_reasons.append("constant_external_dynamic_import_set_changed")
    for target in dynamic_targets:
        if target == "pii_zh" or target.startswith("pii_zh."):
            path = module_to_path.get(target)
            if path is None:
                dynamic_reasons.append("project_dynamic_import_target_missing")
            if any(callsite in runtime_paths for callsite in project_dynamic_paths) and (
                path not in runtime_paths
            ):
                dynamic_reasons.append("runtime_project_dynamic_import_not_closure_bound")
    checks["dynamic_import_boundary"] = (
        _pass_check(
            whole_wheel_python_file_count=len(python_paths),
            constant_dynamic_import_target_count=len(dynamic_targets),
            constant_dynamic_import_call_count=sum(dynamic_targets.values()),
            opaque_dynamic_import_count=0,
            project_dynamic_import_target_count=sum(
                target == "pii_zh" or target.startswith("pii_zh.") for target in dynamic_targets
            ),
            external_dynamic_import_targets=sorted(observed_external),
        )
        if not dynamic_reasons
        else _blocked_check(
            dynamic_reasons,
            whole_wheel_python_file_count=len(python_paths),
            constant_dynamic_import_target_count=len(dynamic_targets),
            constant_dynamic_import_call_count=sum(dynamic_targets.values()),
            opaque_dynamic_import_count=opaque_dynamic_count,
            project_dynamic_import_target_count=sum(
                target == "pii_zh" or target.startswith("pii_zh.") for target in dynamic_targets
            ),
            external_dynamic_import_targets=sorted(observed_external),
        )
    )

    required_resources = contract["package_data_boundary"]["entrypoint_required_resources"]
    resource_reasons: list[str] = []
    for resource in required_resources:
        path = resource["path"]
        consumer_path = module_to_path.get(resource["consumer_module"])
        if path not in package_data_paths:
            resource_reasons.append("entrypoint_resource_not_declared_as_package_data")
        if consumer_path not in runtime_paths:
            resource_reasons.append("entrypoint_resource_consumer_not_in_import_closure")
            continue
        resource_name = PurePosixPath(path).name
        consumer_text = _text(wheel_payloads[consumer_path], field=consumer_path)
        if resource_name not in consumer_text:
            resource_reasons.append("entrypoint_resource_not_statically_named_by_consumer")
    unreferenced_data_count = len(package_data_paths) - len(required_resources)
    if unreferenced_data_count < 0:
        resource_reasons.append("entrypoint_resource_count_exceeds_package_data_inventory")
    checks["package_data_boundary"] = (
        _pass_check(
            declared_pattern_group_count=len(package_data),
            package_data_file_count=len(package_data_paths),
            entrypoint_required_resource_count=len(required_resources),
            distributed_not_entrypoint_required_count=unreferenced_data_count,
            all_package_data_in_whole_wheel_scan=True,
            resource_access_is_static_filename_evidence_not_runtime_trace=True,
        )
        if not resource_reasons
        else _blocked_check(
            resource_reasons,
            declared_pattern_group_count=len(package_data),
            package_data_file_count=len(package_data_paths),
            entrypoint_required_resource_count=len(required_resources),
            distributed_not_entrypoint_required_count=max(unreferenced_data_count, 0),
            all_package_data_in_whole_wheel_scan=True,
            resource_access_is_static_filename_evidence_not_runtime_trace=True,
        )
    )

    wheel_metadata_payloads: dict[str, bytes] = {manifest_path: pyproject_payload}
    duplicate_metadata_paths = 0
    for index, binding in enumerate(contract["wheel_metadata_sources"]):
        path, payload = _verify_binding(root, binding, field=f"wheel_metadata_sources[{index}]")
        if path in wheel_metadata_payloads or path in wheel_payloads:
            duplicate_metadata_paths += 1
        wheel_metadata_payloads[path] = payload
    checks["wheel_metadata_source_binding"] = (
        _pass_check(
            metadata_source_file_count=len(wheel_metadata_payloads),
            duplicate_binding_count=0,
            actual_built_wheel_bound=False,
        )
        if duplicate_metadata_paths == 0
        else _blocked_check(
            ["wheel_metadata_source_binding_overlap"],
            metadata_source_file_count=len(wheel_metadata_payloads),
            duplicate_binding_count=duplicate_metadata_paths,
            actual_built_wheel_bound=False,
        )
    )

    support_payloads: dict[str, bytes] = {}
    duplicate_support_paths = 0
    for index, binding in enumerate(contract["support_surfaces"]):
        path, payload = _verify_binding(root, binding, field=f"support_surfaces[{index}]")
        if path in support_payloads:
            duplicate_support_paths += 1
        support_payloads[path] = payload
    checks["support_surface_binding"] = (
        _pass_check(
            support_surface_file_count=len(support_payloads),
            duplicate_binding_count=0,
        )
        if duplicate_support_paths == 0
        else _blocked_check(
            ["support_surface_binding_overlap"],
            support_surface_file_count=len(support_payloads),
            duplicate_binding_count=duplicate_support_paths,
        )
    )

    private_scan = {**wheel_payloads, **wheel_metadata_payloads, **support_payloads}
    private_counts, private_paths = _pattern_counts(private_scan, _PRIVATE_ABSOLUTE_PATH_PATTERNS)
    private_match_count = sum(private_counts.values())
    checks["private_absolute_paths"] = (
        _pass_check(
            declared_wheel_payload_source_file_count=len(inventory),
            wheel_metadata_source_file_count=len(wheel_metadata_payloads),
            support_surface_file_count=len(support_payloads),
            scanned_file_count=len(private_scan),
            match_count=0,
            affected_file_count=0,
        )
        if private_match_count == 0
        else _blocked_check(
            ["private_absolute_path_in_declared_wheel_inputs_or_installed_surface"],
            declared_wheel_payload_source_file_count=len(inventory),
            wheel_metadata_source_file_count=len(wheel_metadata_payloads),
            support_surface_file_count=len(support_payloads),
            scanned_file_count=len(private_scan),
            match_count=private_match_count,
            affected_file_count=len(private_paths),
        )
    )

    required_resource_paths = {resource["path"] for resource in required_resources}
    runtime_scan = {
        **{path: wheel_payloads[path] for path in runtime_paths},
        **{path: wheel_payloads[path] for path in required_resource_paths},
        **wheel_metadata_payloads,
        **support_payloads,
    }
    gpu_counts, gpu_paths = _pattern_counts(runtime_scan, _FIXED_PHYSICAL_GPU_PATTERNS)
    gpu_match_count = sum(gpu_counts.values())
    checks["fixed_physical_gpu"] = (
        _pass_check(
            installed_surface_scanned_file_count=len(runtime_scan),
            match_count=0,
            affected_file_count=0,
            process_local_logical_cuda_zero_allowed=True,
        )
        if gpu_match_count == 0
        else _blocked_check(
            ["fixed_physical_gpu_in_installed_entrypoint_surface"],
            installed_surface_scanned_file_count=len(runtime_scan),
            match_count=gpu_match_count,
            affected_file_count=len(gpu_paths),
            process_local_logical_cuda_zero_allowed=True,
        )
    )

    source_policy = contract["source_closure"]
    artifact_binding = source_policy["artifact"]
    closure_path, closure_payload = _verify_binding(
        root, artifact_binding, field="source_closure.artifact"
    )
    closure = _json_object(closure_payload, field="source_closure.artifact")
    verifier_path, _ = _verify_binding(
        root, source_policy["verifier"], field="source_closure.verifier"
    )
    verify_closure = _load_source_closure_verifier(
        root.joinpath(*PurePosixPath(verifier_path).parts),
        expected_repository_path=verifier_path,
    )
    closure_replay_reasons: list[str] = []
    try:
        replayed = verify_closure(
            root.joinpath(*PurePosixPath(closure_path).parts), repository_root=root
        )
    except Exception:
        replayed = None
        closure_replay_reasons.append("successor_source_closure_strict_replay_failed")
    observed_closure_canonical = canonical_json_hash(closure, remove="source_closure_sha256")
    if (
        closure.get("closure_id") != source_policy["required_closure_id"]
        or closure.get("closure_id") != artifact_binding["closure_id"]
        or closure.get("source_closure_sha256") != artifact_binding["canonical_sha256"]
        or observed_closure_canonical != artifact_binding["canonical_sha256"]
    ):
        closure_replay_reasons.append("successor_source_closure_identity_mismatch")
    if replayed != closure:
        closure_replay_reasons.append("successor_source_closure_replay_result_mismatch")
    closure_repository_files = closure.get("repository_files")
    source_bound_payloads = {
        **wheel_payloads,
        **wheel_metadata_payloads,
        **support_payloads,
    }
    source_binding_groups = {
        "package_inventory": wheel_payloads,
        "wheel_metadata": wheel_metadata_payloads,
        "support_surface": support_payloads,
    }
    mismatch_counts = {name: 0 for name in source_binding_groups}
    if not isinstance(closure_repository_files, Mapping):
        closure_replay_reasons.append("successor_source_closure_inventory_missing")
        mismatch_counts = {name: len(payloads) for name, payloads in source_binding_groups.items()}
    else:
        for group_name, payloads in source_binding_groups.items():
            for path, payload in payloads.items():
                binding = closure_repository_files.get(path)
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("sha256") != _sha256_bytes(payload)
                    or binding.get("size_bytes") != len(payload)
                ):
                    mismatch_counts[group_name] += 1
        if sum(mismatch_counts.values()):
            closure_replay_reasons.append("release_surface_inputs_not_source_closure_bound")
    checks["successor_source_closure_binding"] = (
        _pass_check(
            closure_id=artifact_binding["closure_id"],
            closure_path=closure_path,
            closure_canonical_sha256=observed_closure_canonical,
            verifier_path=verifier_path,
            strict_replay_passed=True,
            release_surface_source_bound_file_count=len(source_bound_payloads),
            release_surface_source_binding_mismatch_count=0,
            package_inventory_source_bound_file_count=len(wheel_payloads),
            package_inventory_source_binding_mismatch_count=0,
            wheel_metadata_source_bound_file_count=len(wheel_metadata_payloads),
            wheel_metadata_source_binding_mismatch_count=0,
            support_surface_source_bound_file_count=len(support_payloads),
            support_surface_source_binding_mismatch_count=0,
        )
        if not closure_replay_reasons
        else _blocked_check(
            closure_replay_reasons,
            closure_id=closure.get("closure_id"),
            closure_path=closure_path,
            closure_canonical_sha256=observed_closure_canonical,
            verifier_path=verifier_path,
            strict_replay_passed=False,
            release_surface_source_bound_file_count=len(source_bound_payloads),
            release_surface_source_binding_mismatch_count=sum(mismatch_counts.values()),
            package_inventory_source_bound_file_count=len(wheel_payloads),
            package_inventory_source_binding_mismatch_count=mismatch_counts["package_inventory"],
            wheel_metadata_source_bound_file_count=len(wheel_metadata_payloads),
            wheel_metadata_source_binding_mismatch_count=mismatch_counts["wheel_metadata"],
            support_surface_source_bound_file_count=len(support_payloads),
            support_surface_source_binding_mismatch_count=mismatch_counts["support_surface"],
        )
    )

    blocked_ids = sorted(
        check_id for check_id, check in checks.items() if check["status"] != PASS_STATUS
    )
    status = PASS_STATUS if not blocked_ids else "BLOCKED"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "status": status,
        "policy": {
            "policy_id": contract["policy_id"],
            "contract_path": contract_relative,
            "contract_canonical_sha256": observed_contract_hash,
            "contract_file_sha256": _sha256_bytes(contract_payload),
        },
        "release_target": contract["release_target"],
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(blocked_ids),
            "blocked_check_count": len(blocked_ids),
            "blocked_check_ids": blocked_ids,
            "installed_entrypoint_count": len(project_scripts),
            "selected_service_api_count": len(contract["selected_service_apis"]),
            "runtime_import_closure_file_count": len(runtime_paths),
            "declared_wheel_payload_source_file_count": len(inventory),
            "wheel_metadata_source_file_count": len(wheel_metadata_payloads),
        },
        "privacy": {
            "aggregate_and_digest_only": True,
            "raw_pii_literal_persisted": False,
            "record_level_data_read": False,
            "hidden_gold_read": False,
            "model_weights_read": False,
            "gpu_queried_or_used": False,
            "network_used": False,
        },
        "limitations": [
            "PASS proves only the bound local execution-surface-v2 policy.",
            (
                "The wheel-input boundary is the complete setuptools Python/package-data "
                "source inventory plus bound pyproject, README, and license metadata inputs."
            ),
            (
                "No built wheel artifact is bound here, so byte-level whole-wheel privacy "
                "and clean-wheel execution remain separate gates."
            ),
            (
                "The AST closure is conservative for static imports; constant dynamic "
                "imports and package-resource filenames are audited separately, not runtime traced."
            ),
            (
                "The exact successor source closure is strictly replayed against the current "
                "tree; this policy does not replace its broader source-governance claims."
            ),
            (
                "Hosted CI, quality evidence, release artifacts, signatures, approvals, "
                "and publication remain separate gates."
            ),
        ],
        "claims": {
            "local_execution_surface_v2_policy_passed": status == PASS_STATUS,
            "all_project_scripts_enumerated": (
                checks["installed_entrypoints"]["status"] == PASS_STATUS
            ),
            "declared_wheel_source_and_metadata_inputs_private_path_count_zero": (
                checks["private_absolute_paths"]["status"] == PASS_STATUS
            ),
            "successor_source_closure_bound": (
                checks["successor_source_closure_binding"]["status"] == PASS_STATUS
            ),
            "clean_wheel_execution_passed": False,
            "hosted_ci_passed": False,
            "production_ready": False,
            "release_ready": False,
            "published": False,
        },
    }
    report["receipt_sha256"] = canonical_json_hash(report)
    return report


def _receipt_payload(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReleaseExecutionSurfaceV2Error(
                "refusing to overwrite an existing receipt"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _verify_receipt(path: Path, expected: Mapping[str, Any]) -> None:
    metadata = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReleaseExecutionSurfaceV2Error("receipt is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise ReleaseExecutionSurfaceV2Error("receipt mode must be 0444")
    actual, payload = _load_json_path(path, field="receipt")
    if actual != expected or payload != _receipt_payload(expected):
        raise ReleaseExecutionSurfaceV2Error("receipt does not match the current report")
    if actual.get("receipt_sha256") != canonical_json_hash(actual, remove="receipt_sha256"):
        raise ReleaseExecutionSurfaceV2Error("receipt canonical hash does not match")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(contract_path=args.contract, schema_path=args.schema)
        if args.output is not None:
            _write_no_clobber(args.output, _receipt_payload(report))
        else:
            _verify_receipt(args.verify_receipt, report)
    except (OSError, ReleaseExecutionSurfaceV2Error, ValueError) as exc:
        print(f"release execution-surface-v2 validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "blocked_check_count": report["summary"]["blocked_check_count"],
                "receipt_sha256": report["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
