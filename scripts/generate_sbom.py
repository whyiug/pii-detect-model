#!/usr/bin/env python3
"""Generate a deterministic CycloneDX 1.6 SBOM from a committed uv lockfile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import tomllib

CYCLONEDX_NAMESPACE = "http://cyclonedx.org/schema/bom-1.6.schema.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _purl(name: str, version: str) -> str:
    normalized = name.lower().replace("_", "-")
    quoted_name = urllib.parse.quote(normalized, safe="")
    quoted_version = urllib.parse.quote(version, safe=".+-")
    return f"pkg:pypi/{quoted_name}@{quoted_version}"


def _component_ref(package: dict[str, Any]) -> str:
    name = str(package["name"])
    version = str(package.get("version", "0+local"))
    marker = str(package.get("resolution-markers", package.get("marker", "")))
    discriminator = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:12] if marker else "default"
    return f"{_purl(name, version)}?lock-variant={discriminator}"


def _hashes(package: dict[str, Any]) -> list[dict[str, str]]:
    values: set[str] = set()
    candidates: list[dict[str, Any]] = []
    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        candidates.append(sdist)
    wheels = package.get("wheels")
    if isinstance(wheels, list):
        candidates.extend(item for item in wheels if isinstance(item, dict))
    for candidate in candidates:
        value = candidate.get("hash")
        if isinstance(value, str) and value.startswith("sha256:"):
            digest = value.removeprefix("sha256:")
            if len(digest) == 64:
                values.add(digest.upper())
    return [{"alg": "SHA-256", "content": value} for value in sorted(values)]


def _license_expression(package: dict[str, Any]) -> str | None:
    metadata = package.get("metadata")
    if not isinstance(metadata, dict):
        return None
    license_value = metadata.get("license")
    return license_value if isinstance(license_value, str) and license_value.strip() else None


def _sanitized_source(source: Any) -> dict[str, Any]:
    """Keep source provenance without copying credentials or local absolute paths."""

    if not isinstance(source, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in source.items():
        if key in {"editable", "path", "virtual"}:
            sanitized[key] = "<local>"
        elif key in {"git", "registry", "url"} and isinstance(value, str):
            parsed = urllib.parse.urlsplit(value)
            hostname = parsed.hostname or ""
            netloc = hostname
            if parsed.port is not None:
                netloc += f":{parsed.port}"
            sanitized[key] = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        elif isinstance(value, (bool, int, float, str)):
            sanitized[key] = value
    return sanitized


def build_sbom(lockfile: Path, pyproject: Path) -> dict[str, Any]:
    lock = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    project_document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = project_document.get("project")
    packages = lock.get("package")
    if not isinstance(project, dict) or not isinstance(packages, list):
        raise ValueError("pyproject or uv.lock has an unsupported structure")
    project_name = project.get("name")
    project_version = project.get("version")
    if not isinstance(project_name, str) or not isinstance(project_version, str):
        raise ValueError("project name/version must be declared in pyproject.toml")

    components: list[dict[str, Any]] = []
    refs_by_name: dict[str, list[str]] = {}
    package_by_ref: dict[str, dict[str, Any]] = {}
    root_package: dict[str, Any] | None = None
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock package entries must be tables")
        name = package.get("name")
        version = package.get("version", project_version if name == project_name else None)
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("every locked package must have a name and resolved version")
        normalized_package = dict(package)
        normalized_package["version"] = version
        if name == project_name:
            if root_package is not None:
                raise ValueError(f"uv.lock contains duplicate root package {project_name}")
            root_package = normalized_package
            continue
        reference = _component_ref(normalized_package)
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": reference,
            "name": name,
            "version": version,
            "purl": _purl(name, version),
            "properties": [
                {
                    "name": "pii-zh-qwen:source",
                    "value": json.dumps(_sanitized_source(package.get("source")), sort_keys=True),
                },
            ],
        }
        hashes = _hashes(package)
        if hashes:
            component["hashes"] = hashes
        license_expression = _license_expression(package)
        if license_expression:
            component["licenses"] = [{"expression": license_expression}]
        components.append(component)
        refs_by_name.setdefault(name, []).append(reference)
        package_by_ref[reference] = normalized_package

    components.sort(key=lambda item: item["bom-ref"])
    dependencies: list[dict[str, Any]] = []
    for reference in sorted(package_by_ref):
        package = package_by_ref[reference]
        dependency_refs: set[str] = set()
        for dependency in package.get("dependencies", []):
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                continue
            candidates = refs_by_name.get(dependency["name"], [])
            requested_version = dependency.get("version")
            if isinstance(requested_version, str):
                candidates = [
                    candidate
                    for candidate in candidates
                    if package_by_ref[candidate].get("version") == requested_version
                ]
            dependency_refs.update(candidates)
        dependencies.append({"ref": reference, "dependsOn": sorted(dependency_refs)})

    if root_package is None:
        raise ValueError(f"uv.lock does not contain root package {project_name}")
    root_ref = _purl(project_name, project_version)
    root_dependency_refs: set[str] = set()
    root_dependency_entries = list(root_package.get("dependencies", []))
    optional_dependencies = root_package.get("optional-dependencies", {})
    if isinstance(optional_dependencies, dict):
        for entries in optional_dependencies.values():
            if isinstance(entries, list):
                root_dependency_entries.extend(entries)
    for dependency in root_dependency_entries:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
            continue
        candidates = refs_by_name.get(dependency["name"], [])
        requested_version = dependency.get("version")
        if isinstance(requested_version, str):
            candidates = [
                candidate
                for candidate in candidates
                if package_by_ref[candidate].get("version") == requested_version
            ]
        root_dependency_refs.update(candidates)
    root_dependencies = sorted(root_dependency_refs)
    dependencies.append({"ref": root_ref, "dependsOn": root_dependencies})
    identity_material = json.dumps(
        {
            "lock_sha256": sha256_file(lockfile),
            "project_sha256": sha256_file(pyproject),
            "components": [component["bom-ref"] for component in components],
        },
        sort_keys=True,
    )
    serial = uuid.uuid5(uuid.NAMESPACE_URL, identity_material)
    return {
        "$schema": CYCLONEDX_NAMESPACE,
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": project_name,
                "version": project_version,
                "purl": root_ref,
            },
            "properties": [
                {"name": "pii-zh-qwen:generator", "value": "scripts/generate_sbom.py"},
                {"name": "pii-zh-qwen:lock-sha256", "value": sha256_file(lockfile)},
                {"name": "pii-zh-qwen:pyproject-sha256", "value": sha256_file(pyproject)},
            ],
        },
        "components": components,
        "dependencies": sorted(dependencies, key=lambda item: item["ref"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--check", action="store_true", help="fail if output differs instead of writing"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        sbom = build_sbom(args.lockfile, args.pyproject)
        rendered = json.dumps(sbom, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
                print(f"SBOM is missing or stale: {args.output}", file=sys.stderr)
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"SBOM generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"CycloneDX SBOM verified: {args.output}"
        if args.check
        else f"Wrote CycloneDX SBOM: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
