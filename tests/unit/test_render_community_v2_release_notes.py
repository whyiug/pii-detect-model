from __future__ import annotations

from pathlib import Path

import pytest
from scripts import render_community_v2_release_notes as renderer

SOURCE_COMMIT = "a" * 40
HF_COMMIT = "b" * 40


def test_renders_exact_source_and_hf_bindings() -> None:
    payload = renderer.render_release_notes(
        renderer.DEFAULT_TEMPLATE,
        git_source_commit=SOURCE_COMMIT,
        hugging_face_commit=HF_COMMIT,
    )
    text = payload.decode("utf-8")
    hf_url = (
        f"https://huggingface.co/{renderer.HUGGING_FACE_REPOSITORY}/tree/{HF_COMMIT}"
    )
    assert f"`git_source_commit: {SOURCE_COMMIT}`" in text
    assert f"`hugging_face_immutable_commit: {HF_COMMIT}`" in text
    assert f"`hugging_face_immutable_url: {hf_url}`" in text
    assert "{{" not in text


@pytest.mark.parametrize(
    ("source_commit", "hf_commit"),
    [
        ("A" * 40, HF_COMMIT),
        ("a" * 39, HF_COMMIT),
        (SOURCE_COMMIT, "main"),
        (SOURCE_COMMIT, "B" * 40),
    ],
)
def test_rejects_non_immutable_identity(source_commit: str, hf_commit: str) -> None:
    with pytest.raises(renderer.ReleaseNotesRenderError):
        renderer.render_release_notes(
            renderer.DEFAULT_TEMPLATE,
            git_source_commit=source_commit,
            hugging_face_commit=hf_commit,
        )


def test_rejects_missing_or_repeated_placeholder(tmp_path: Path) -> None:
    original = renderer.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    for index, text in enumerate(
        (
            original.replace(renderer.SOURCE_PLACEHOLDER, "missing"),
            original.replace(
                renderer.HF_COMMIT_PLACEHOLDER,
                renderer.HF_COMMIT_PLACEHOLDER * 2,
            ),
        )
    ):
        template = tmp_path / f"bad-{index}.md"
        template.write_text(text, encoding="utf-8")
        with pytest.raises(renderer.ReleaseNotesRenderError, match="exactly once"):
            renderer.render_release_notes(
                template,
                git_source_commit=SOURCE_COMMIT,
                hugging_face_commit=HF_COMMIT,
            )


def test_output_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "release-notes.final.md"
    payload = renderer.render_release_notes(
        renderer.DEFAULT_TEMPLATE,
        git_source_commit=SOURCE_COMMIT,
        hugging_face_commit=HF_COMMIT,
    )
    renderer.model_card_renderer.publish_no_clobber(output, payload)
    with pytest.raises(renderer.model_card_renderer.ModelCardRenderError, match="overwrite"):
        renderer.model_card_renderer.publish_no_clobber(output, payload)
