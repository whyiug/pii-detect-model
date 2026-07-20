from __future__ import annotations

import os
from pathlib import Path

import pytest
from scripts import render_community_v2_publication_model_card as renderer

GITHUB_REPOSITORY = "whyiug/pii-detect-model"
GIT_SOURCE_COMMIT = "b" * 40
RELEASE_TAG = "v0.2.0rc1"
RELEASE_URL = (
    "https://github.com/whyiug/pii-detect-model/releases/tag/v0.2.0rc1"
)


def _render(template: Path = renderer.DEFAULT_TEMPLATE) -> bytes:
    return renderer.render_model_card(
        template,
        git_source_commit=GIT_SOURCE_COMMIT,
        release_tag=RELEASE_TAG,
        github_repository=GITHUB_REPOSITORY,
        github_release_url=RELEASE_URL,
    )


def test_repository_template_renders_one_exact_release_binding() -> None:
    payload = _render()
    text = payload.decode("utf-8")
    assert text.endswith("\n") and "\r" not in text
    assert all(placeholder not in text for placeholder in renderer.PLACEHOLDERS)
    assert text.count(f"`git_source_commit: {GIT_SOURCE_COMMIT}`") == 1
    assert text.count(f"`release_tag: {RELEASE_TAG}`") == 1
    assert text.count(f"`github_release_url: {RELEASE_URL}`") == 1
    for stale in renderer._STALE_REMOTE_STATE_MARKERS:
        assert stale.casefold() not in text.casefold()


def test_publish_is_no_clobber_and_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "README.publication.final.md"
    renderer.publish_no_clobber(output, _render())
    original = output.read_bytes()

    with pytest.raises(renderer.ModelCardRenderError, match="overwrite"):
        renderer.publish_no_clobber(output, b"replacement\n")
    assert output.read_bytes() == original


def test_publish_refuses_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "caller-owned.md"
    target.write_text("caller data\n", encoding="utf-8")
    output = tmp_path / "README.publication.final.md"
    output.symlink_to(target)

    with pytest.raises(renderer.ModelCardRenderError, match="overwrite"):
        renderer.publish_no_clobber(output, _render())
    assert target.read_text(encoding="utf-8") == "caller data\n"


def test_renderer_refuses_symlink_template(tmp_path: Path) -> None:
    template = tmp_path / "template.md"
    template.symlink_to(renderer.DEFAULT_TEMPLATE)
    with pytest.raises(renderer.ModelCardRenderError, match="unavailable"):
        _render(template)


def test_renderer_fails_closed_if_template_path_inode_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.md"
    template.write_bytes(renderer.DEFAULT_TEMPLATE.read_bytes())
    decoy = tmp_path / "decoy.md"
    decoy.write_bytes(renderer.DEFAULT_TEMPLATE.read_bytes())
    real_lstat = os.lstat

    def swapped_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        if Path(path) == template:
            return real_lstat(decoy)
        return real_lstat(path)

    monkeypatch.setattr(renderer.os, "lstat", swapped_lstat)
    with pytest.raises(renderer.ModelCardRenderError, match="changed while it was read"):
        _render(template)


def test_publish_fails_closed_if_final_output_inode_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "README.publication.final.md"
    decoy = tmp_path / "decoy.md"
    decoy.write_text("different inode\n", encoding="utf-8")
    real_lstat = os.lstat

    def swapped_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        candidate = Path(path)
        if candidate == output and output.exists():
            return real_lstat(decoy)
        return real_lstat(path)

    monkeypatch.setattr(renderer.os, "lstat", swapped_lstat)
    with pytest.raises(renderer.ModelCardRenderError, match="identity changed"):
        renderer.publish_no_clobber(output, _render())


@pytest.mark.parametrize(
    ("commit", "tag", "repository", "url"),
    [
        ("abc123", RELEASE_TAG, GITHUB_REPOSITORY, RELEASE_URL),
        (GIT_SOURCE_COMMIT, "v0.2.0", GITHUB_REPOSITORY, RELEASE_URL),
        (GIT_SOURCE_COMMIT, RELEASE_TAG, "not-a-repository", RELEASE_URL),
        (
            GIT_SOURCE_COMMIT,
            RELEASE_TAG,
            GITHUB_REPOSITORY,
            "https://github.com/other/repository/releases/tag/v0.2.0rc1",
        ),
    ],
)
def test_invalid_release_identity_fails_closed(
    commit: str, tag: str, repository: str, url: str
) -> None:
    with pytest.raises(renderer.ModelCardRenderError):
        renderer.render_model_card(
            renderer.DEFAULT_TEMPLATE,
            git_source_commit=commit,
            release_tag=tag,
            github_repository=repository,
            github_release_url=url,
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "stale"])
def test_template_placeholder_or_remote_state_drift_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    text = renderer.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    if mutation == "missing":
        text = text.replace(renderer.GIT_SOURCE_COMMIT_PLACEHOLDER, "")
    elif mutation == "duplicate":
        text += renderer.RELEASE_TAG_PLACEHOLDER + "\n"
    else:
        text += "Current state: staged_not_uploaded\n"
    template = tmp_path / "template.md"
    template.write_text(text, encoding="utf-8", newline="\n")

    with pytest.raises(renderer.ModelCardRenderError):
        _render(template)


@pytest.mark.parametrize(
    "mutation",
    [
        "package_version",
        "publication_state",
        "pipeline_tag",
        "duplicate_key",
        "second_frontmatter",
        "github_target",
        "hugging_face_target",
    ],
)
def test_frontmatter_and_fixed_target_drift_fails_even_when_body_keeps_version(
    tmp_path: Path, mutation: str
) -> None:
    text = renderer.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    if mutation == "package_version":
        text = text.replace("package_version: 0.2.0rc1", "package_version: 0.2.0")
    elif mutation == "publication_state":
        text = text.replace("publication_state: release_candidate", "publication_state: public")
    elif mutation == "pipeline_tag":
        text = text.replace(
            "pipeline_tag: token-classification", "pipeline_tag: text-classification"
        )
    elif mutation == "duplicate_key":
        text = text.replace(
            "publication_state: release_candidate\n",
            "publication_state: release_candidate\npublication_state: release_candidate\n",
        )
    elif mutation == "second_frontmatter":
        text += "---\nextra: block\n---\n"
    elif mutation == "github_target":
        text = text.replace(
            "github_repository: whyiug/pii-detect-model",
            "github_repository: other/pii-detect-model",
        )
    else:
        text = text.replace(
            "hugging_face_repository: Forrest20231206/pii-zh-qwen3-0.6b-24class",
            "hugging_face_repository: other/pii-zh-qwen3-0.6b-24class",
        )
    template = tmp_path / "template.md"
    template.write_text(text, encoding="utf-8", newline="\n")

    with pytest.raises(renderer.ModelCardRenderError):
        _render(template)
