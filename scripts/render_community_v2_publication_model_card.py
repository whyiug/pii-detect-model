#!/usr/bin/env python3
"""Render the community-v2 Model Card after the source commit is frozen.

The repository template cannot contain the hash of the commit which contains
that template.  This offline renderer resolves that unavoidable self-reference
outside the source tree, validates the complete GitHub release binding, and
publishes one no-clobber UTF-8 artifact for the publication-successor builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Final

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPOSITORY_ROOT / "model_cards/PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md"
PACKAGE_VERSION: Final = "0.2.0rc1"
EXPECTED_RELEASE_TAG: Final = "v0.2.0rc1"
EXPECTED_GITHUB_REPOSITORY: Final = "whyiug/pii-detect-model"
EXPECTED_HUGGING_FACE_REPOSITORY: Final = (
    "Forrest20231206/pii-zh-qwen3-0.6b-24class"
)
GIT_SOURCE_COMMIT_PLACEHOLDER: Final = "{{GIT_SOURCE_COMMIT}}"
RELEASE_TAG_PLACEHOLDER: Final = "{{RELEASE_TAG}}"
GITHUB_RELEASE_URL_PLACEHOLDER: Final = "{{GITHUB_RELEASE_URL}}"
PLACEHOLDERS: Final = (
    GIT_SOURCE_COMMIT_PLACEHOLDER,
    RELEASE_TAG_PLACEHOLDER,
    GITHUB_RELEASE_URL_PLACEHOLDER,
)

_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPOSITORY_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)
_MAX_TEMPLATE_BYTES = 8 * 1024 * 1024
_EXPECTED_FRONTMATTER: Final[dict[str, Any]] = {
    "language": ["zh"],
    "library_name": "transformers",
    "pipeline_tag": "token-classification",
    "license": "apache-2.0",
    "base_model": "ZJUICSR/AIguard-pii-detection-fast",
    "base_model_revision": "677a5ebc1600fef61e8973cafd3026be322b3a73",
    "tags": ["pii", "chinese", "token-classification"],
    "model_name": "pii-zh-qwen3-0.6b-24class",
    "package_version": PACKAGE_VERSION,
    "publication_state": "release_candidate",
}
_STALE_REMOTE_STATE_MARKERS = (
    "staged_not_uploaded",
    "尚未上传",
    "两个 private 暂存仓库",
)


class ModelCardRenderError(RuntimeError):
    """Raised when a final Model Card cannot be rendered safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ModelCardRenderError(f"Model Card frontmatter repeats key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_frontmatter(text: str) -> None:
    lines = text.splitlines(keepends=True)
    delimiters = [index for index, line in enumerate(lines) if line == "---\n"]
    if len(delimiters) != 2 or delimiters[0] != 0:
        raise ModelCardRenderError("Model Card must contain one unique YAML frontmatter block")
    closing = delimiters[1]
    if closing <= 1:
        raise ModelCardRenderError("Model Card YAML frontmatter is empty")
    try:
        document = yaml.load("".join(lines[1:closing]), Loader=_UniqueKeyLoader)
    except ModelCardRenderError:
        raise
    except (TypeError, yaml.YAMLError) as exc:
        raise ModelCardRenderError("Model Card YAML frontmatter is invalid") from exc
    if document != _EXPECTED_FRONTMATTER:
        raise ModelCardRenderError("Model Card YAML frontmatter differs from the release contract")


def _read_regular_template(path: Path) -> str:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ModelCardRenderError("safe Model Card reads require O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ModelCardRenderError("Model Card template is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > (
            _MAX_TEMPLATE_BYTES
        ):
            raise ModelCardRenderError("Model Card template must be a regular file of safe size")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > _MAX_TEMPLATE_BYTES:
                raise ModelCardRenderError("Model Card template exceeds the size limit")
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_before != identity_current
            or total != before.st_size
        ):
            raise ModelCardRenderError("Model Card template changed while it was read")
        payload = b"".join(chunks)
    except OSError as exc:
        raise ModelCardRenderError("Model Card template could not be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelCardRenderError("Model Card template must be UTF-8") from exc
    if "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ModelCardRenderError(
            "Model Card template must be NUL-free LF-only text ending in one newline"
        )
    _validate_frontmatter(text)
    for placeholder in PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ModelCardRenderError(
                f"Model Card template must contain {placeholder} exactly once"
            )
    stale = next(
        (marker for marker in _STALE_REMOTE_STATE_MARKERS if marker.casefold() in text.casefold()),
        None,
    )
    if stale is not None:
        raise ModelCardRenderError(
            f"Model Card template retains stale remote-state marker {stale!r}"
        )
    return text


def _binding_values(text: str, *, key: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?[`\"']?{re.escape(key)}[`\"']?\s*[:=]\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    values: list[str] = []
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is not None:
            values.append(match.group(1).strip().strip("`\"'").strip())
    return values


def render_model_card(
    template: Path,
    *,
    git_source_commit: str,
    release_tag: str,
    github_repository: str,
    github_release_url: str,
) -> bytes:
    """Return a fully bound final Model Card without writing it."""

    if _GIT_COMMIT.fullmatch(git_source_commit) is None:
        raise ModelCardRenderError("Git source commit must be a full lowercase hex object ID")
    if release_tag != EXPECTED_RELEASE_TAG:
        raise ModelCardRenderError(f"release tag must equal {EXPECTED_RELEASE_TAG}")
    if _REPOSITORY_ID.fullmatch(github_repository) is None:
        raise ModelCardRenderError("GitHub repository must be an owner/repository ID")
    if github_repository != EXPECTED_GITHUB_REPOSITORY:
        raise ModelCardRenderError(
            f"GitHub repository must equal {EXPECTED_GITHUB_REPOSITORY}"
        )
    expected_url = f"https://github.com/{github_repository}/releases/tag/{release_tag}"
    if github_release_url != expected_url:
        raise ModelCardRenderError("GitHub Release URL does not match the repository and tag")

    text = _read_regular_template(template)
    replacements = {
        GIT_SOURCE_COMMIT_PLACEHOLDER: git_source_commit,
        RELEASE_TAG_PLACEHOLDER: release_tag,
        GITHUB_RELEASE_URL_PLACEHOLDER: github_release_url,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    if any(placeholder in text for placeholder in PLACEHOLDERS):
        raise ModelCardRenderError("Model Card retains an unresolved release placeholder")
    expected_bindings = {
        "github_repository": EXPECTED_GITHUB_REPOSITORY,
        "hugging_face_repository": EXPECTED_HUGGING_FACE_REPOSITORY,
        "git_source_commit": git_source_commit,
        "release_tag": release_tag,
        "github_release_url": github_release_url,
    }
    for key, expected in expected_bindings.items():
        if _binding_values(text, key=key) != [expected]:
            raise ModelCardRenderError(f"rendered Model Card {key} binding is not exact")
    return text.encode("utf-8")


def publish_no_clobber(path: Path, payload: bytes) -> None:
    """Create one output without following a symlink or replacing caller data."""

    try:
        parent_metadata = os.lstat(path.parent)
    except OSError as exc:
        raise ModelCardRenderError("Model Card output parent is unavailable") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ModelCardRenderError("Model Card output parent must be a real directory")
    parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ModelCardRenderError("refusing to overwrite existing Model Card output") from exc
    except OSError as exc:
        raise ModelCardRenderError("Model Card output could not be created") from exc
    created_metadata = os.fstat(descriptor)
    created_identity = (created_metadata.st_dev, created_metadata.st_ino)
    complete = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        final_descriptor_metadata = os.fstat(descriptor)
        final_path_metadata = os.lstat(path)
        final_parent_metadata = os.lstat(path.parent)
        if (
            not stat.S_ISREG(final_descriptor_metadata.st_mode)
            or not stat.S_ISREG(final_path_metadata.st_mode)
            or (final_descriptor_metadata.st_dev, final_descriptor_metadata.st_ino)
            != (final_path_metadata.st_dev, final_path_metadata.st_ino)
            or final_descriptor_metadata.st_size != len(payload)
            or final_path_metadata.st_size != len(payload)
            or (final_parent_metadata.st_dev, final_parent_metadata.st_ino) != parent_identity
        ):
            raise ModelCardRenderError("Model Card output identity changed while it was written")
        complete = True
    except OSError as exc:
        raise ModelCardRenderError("Model Card output could not be written") from exc
    finally:
        os.close(descriptor)
        if not complete:
            try:
                current_metadata = os.lstat(path)
            except OSError:
                pass
            else:
                if (
                    stat.S_ISREG(current_metadata.st_mode)
                    and not stat.S_ISLNK(current_metadata.st_mode)
                    and (current_metadata.st_dev, current_metadata.st_ino) == created_identity
                ):
                    path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-source-commit", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--github-release-url", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = render_model_card(
            args.template,
            git_source_commit=args.git_source_commit,
            release_tag=args.release_tag,
            github_repository=args.github_repository,
            github_release_url=args.github_release_url,
        )
        publish_no_clobber(args.output, payload)
    except ModelCardRenderError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "remote_write_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "RENDERED",
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "remote_write_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
