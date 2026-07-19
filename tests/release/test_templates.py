from __future__ import annotations

from pathlib import Path


def test_docker_templates_are_offline_weightless_and_do_not_expose_ports(
    repository_root: Path,
) -> None:
    for filename in ("train.Dockerfile", "inference.Dockerfile"):
        text = (repository_root / "docker" / filename).read_text(encoding="utf-8")
        lowered = text.lower()
        instructions = [
            line.strip().lower()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert "hf_hub_offline=1" in lowered
        assert "transformers_offline=1" in lowered
        assert "uv sync --frozen" in lowered
        assert not any(line.startswith("expose ") for line in instructions)
        assert "from_pretrained" not in lowered
        assert "huggingface-cli" not in lowered
        assert "hf download" not in lowered
        assert "/data1" not in lowered
        assert "copy . " not in lowered
        assert all(
            ".safetensors" not in line.lower()
            for line in text.splitlines()
            if line.startswith("COPY ")
        )


def test_inference_template_defaults_to_lightweight_rules_profile(
    repository_root: Path,
) -> None:
    text = (repository_root / "docker" / "inference.Dockerfile").read_text(encoding="utf-8")

    assert "ARG UV_IMAGE=ghcr.io/astral-sh/uv:" in text
    assert "@sha256:" in text
    assert "FROM ${UV_IMAGE} AS uv-bin" in text
    assert "COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uv" in text
    assert "COPY --chmod=0644 docker/PACKAGE_README.md ./README.md" in text
    assert "COPY pyproject.toml uv.lock README.md" not in text
    assert "UV_PYTHON_DOWNLOADS=never" in text
    assert "pip install" not in text.casefold()
    assert "ARG RUNTIME_PROFILE=rules-only" in text
    assert "rules-only)" in text
    assert "uv sync --frozen --no-dev --extra service --no-editable" in text
    assert "--extra presidio --extra service" not in text
    assert "local-model)" in text
    assert "--extra cascade --extra service" in text
    assert "unsupported RUNTIME_PROFILE" in text


def test_docker_build_context_is_an_explicit_allowlist(repository_root: Path) -> None:
    patterns = (repository_root / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert patterns[0] == "**"
    assert "!pyproject.toml" in patterns
    assert "!uv.lock" in patterns
    assert "!README.md" not in patterns
    assert "!docker/PACKAGE_README.md" in patterns
    assert "!src/**" in patterns
    assert "src/**/__pycache__/" in patterns
    assert "src/**/*.py[cod]" in patterns
    assert "src/*.egg-info/" in patterns
    assert "src/**/*.egg-info/" in patterns
    assert "!scripts/train.py" in patterns
    assert not any("model" in pattern.casefold() for pattern in patterns)

    package_readme = (repository_root / "docker" / "PACKAGE_README.md").read_text(encoding="utf-8")
    lowered = package_readme.casefold()
    assert "/data" not in lowered
    assert "/home" not in lowered
    assert "token=" not in lowered
    assert "api_key" not in lowered


def test_ci_runs_core_tests_lint_and_package_smoke(repository_root: Path) -> None:
    workflow = (repository_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv run ruff check src tests scripts" in workflow
    assert "uv run python -m scripts.run_current_community_rc_tests" in workflow
    assert "PUBLIC_CLEAN_CHECKOUT_TEST_PATHS" in (
        repository_root / "scripts" / "run_current_community_rc_tests.py"
    ).read_text(encoding="utf-8")
    assert "python -m build --wheel" in workflow
    assert "scripts/generate_sbom.py" in workflow
