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


def test_ci_runs_core_tests_lint_and_package_smoke(repository_root: Path) -> None:
    workflow = (repository_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv run ruff check src tests scripts" in workflow
    assert "uv run pytest tests/unit tests/release" in workflow
    assert "python -m build --wheel" in workflow
    assert "scripts/generate_sbom.py" in workflow
