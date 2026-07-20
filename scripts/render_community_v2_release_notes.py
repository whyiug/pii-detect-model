#!/usr/bin/env python3
"""Render the RC1 GitHub Release body with immutable source and HF identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts import build_community_v2_publication_successor as successor
    from scripts import render_community_v2_publication_model_card as model_card_renderer
except ImportError:  # pragma: no cover - direct script execution
    import build_community_v2_publication_successor as successor  # type: ignore[no-redef]
    import render_community_v2_publication_model_card as model_card_renderer  # type: ignore[no-redef]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPOSITORY_ROOT / "release/community-v2-rc1/RELEASE_NOTES.md"
GITHUB_REPOSITORY = "whyiug/pii-detect-model"
HUGGING_FACE_REPOSITORY = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
RELEASE_TAG = "v0.2.0rc1"
SOURCE_PLACEHOLDER = "{{GIT_SOURCE_COMMIT}}"
HF_COMMIT_PLACEHOLDER = "{{HF_IMMUTABLE_SHA}}"
HF_URL_PLACEHOLDER = "{{HF_IMMUTABLE_URL}}"
PLACEHOLDERS = (SOURCE_PLACEHOLDER, HF_COMMIT_PLACEHOLDER, HF_URL_PLACEHOLDER)
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_TEMPLATE_BYTES = 8 * 1024 * 1024


class ReleaseNotesRenderError(RuntimeError):
    """Raised when a final GitHub Release body cannot be rendered safely."""


def _read_template(path: Path) -> str:
    try:
        payload = successor._read_regular_payload(
            path,
            field="community-v2 Release notes template",
            maximum_bytes=MAX_TEMPLATE_BYTES,
        ).payload
    except successor.PublicationSuccessorError as exc:
        raise ReleaseNotesRenderError("Release notes template is unavailable or unsafe") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseNotesRenderError("Release notes template must be UTF-8") from exc
    if "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ReleaseNotesRenderError(
            "Release notes template must be NUL-free LF-only text ending in a newline"
        )
    for placeholder in PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ReleaseNotesRenderError(
                f"Release notes template must contain {placeholder} exactly once"
            )
    return text


def render_release_notes(
    template: Path, *, git_source_commit: str, hugging_face_commit: str
) -> bytes:
    if GIT_SHA_RE.fullmatch(git_source_commit) is None:
        raise ReleaseNotesRenderError("Git source commit must be a full lowercase SHA-1")
    if GIT_SHA_RE.fullmatch(hugging_face_commit) is None:
        raise ReleaseNotesRenderError("Hugging Face commit must be a full lowercase SHA-1")
    hf_url = (
        f"https://huggingface.co/{HUGGING_FACE_REPOSITORY}/tree/{hugging_face_commit}"
    )
    text = _read_template(template)
    replacements = {
        SOURCE_PLACEHOLDER: git_source_commit,
        HF_COMMIT_PLACEHOLDER: hugging_face_commit,
        HF_URL_PLACEHOLDER: hf_url,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    if any(placeholder in text for placeholder in PLACEHOLDERS):
        raise ReleaseNotesRenderError("Release notes retain an unresolved identity placeholder")
    expected_lines = (
        f"- `git_source_commit: {git_source_commit}`",
        f"- `release_tag: {RELEASE_TAG}`",
        f"- `github_repository: {GITHUB_REPOSITORY}`",
        f"- `hugging_face_repository: {HUGGING_FACE_REPOSITORY}`",
        f"- `hugging_face_immutable_commit: {hugging_face_commit}`",
        f"- `hugging_face_immutable_url: {hf_url}`",
    )
    lines = text.splitlines()
    for expected in expected_lines:
        if lines.count(expected) != 1:
            raise ReleaseNotesRenderError(
                f"rendered Release notes do not contain exact binding {expected!r}"
            )
    return text.encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-source-commit", required=True)
    parser.add_argument("--hugging-face-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = render_release_notes(
            args.template,
            git_source_commit=args.git_source_commit,
            hugging_face_commit=args.hugging_face_commit,
        )
        model_card_renderer.publish_no_clobber(args.output, payload)
    except (
        OSError,
        ReleaseNotesRenderError,
        model_card_renderer.ModelCardRenderError,
    ) as exc:
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
    raise SystemExit(main())
