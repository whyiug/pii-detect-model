PYTHON ?= python
ARTIFACT ?= release/hf_model/local-rc
SOURCE_REGISTRY ?= configs/data/source_registry.yaml
DEPENDENCY_SCAN ?=
DEPENDENCY_EXCEPTIONS ?= release/dependency-security-exceptions.example.json

.PHONY: help test test-release lint package-smoke sbom build-release release-gate docker-train docker-inference

help:
	@$(PYTHON) -c "print('Targets: test, test-release, lint, package-smoke, sbom, build-release, release-gate, docker-train, docker-inference')"

test:
	$(PYTHON) -m pytest tests/unit tests/release

test-release:
	$(PYTHON) -m pytest tests/release

lint:
	$(PYTHON) -m ruff check src tests scripts

package-smoke:
	rm -rf release/package-smoke
	$(PYTHON) -m pip wheel . --no-deps --wheel-dir release/package-smoke
	$(PYTHON) -c "import zipfile; from pathlib import Path; wheel=next(Path('release/package-smoke').glob('*.whl')); names=zipfile.ZipFile(wheel).namelist(); assert 'pii_zh/taxonomy/taxonomy.yaml' in names; assert 'pii_zh/taxonomy/presidio_mapping.yaml' in names"

sbom:
	$(PYTHON) scripts/generate_sbom.py --lockfile uv.lock --pyproject pyproject.toml --output release/sbom.cdx.json

build-release:
	@test -n "$(CHECKPOINT_DIR)" || (echo "CHECKPOINT_DIR is required" >&2; exit 2)
	@test -n "$(EVIDENCE_DIR)" || (echo "EVIDENCE_DIR is required" >&2; exit 2)
	$(PYTHON) scripts/build_release.py --checkpoint-dir "$(CHECKPOINT_DIR)" --evidence-dir "$(EVIDENCE_DIR)" --output-dir "$(ARTIFACT)"

release-gate:
	@test -n "$(DEPENDENCY_SCAN)" || (echo "DEPENDENCY_SCAN is required" >&2; exit 2)
	$(PYTHON) scripts/release_gate.py --artifact "$(ARTIFACT)" --source-registry "$(SOURCE_REGISTRY)" --dependency-scan "$(DEPENDENCY_SCAN)" --dependency-exceptions "$(DEPENDENCY_EXCEPTIONS)"

docker-train:
	docker build -f docker/train.Dockerfile -t pii-zh-qwen-train:local .

docker-inference:
	docker build -f docker/inference.Dockerfile -t pii-zh-qwen-inference:local .

