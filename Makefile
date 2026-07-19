PYTHON ?= python
ARTIFACT ?= release/hf_model/local-rc
SOURCE_REGISTRY ?= configs/data/source_registry.yaml
DEPENDENCY_SCAN ?=
DEPENDENCY_EXCEPTIONS ?= release/dependency-security-exceptions.example.json

.PHONY: help test test-release lint package-smoke sbom build-release release-gate docker-train docker-inference

help:
	@$(PYTHON) -c "print('Targets: test, test-release, lint, package-smoke, sbom, build-release, release-gate, docker-train, docker-inference')"

test:
	PYTHONPATH=src $(PYTHON) -m scripts.run_current_community_rc_tests

test-release:
	$(PYTHON) -m pytest tests/release

lint:
	$(PYTHON) -m ruff check src tests scripts

package-smoke:
	rm -rf release/package-smoke
	$(PYTHON) -m pip wheel . --no-deps --wheel-dir release/package-smoke
	$(PYTHON) -c "import zipfile; from pathlib import Path; wheels=list(Path('release/package-smoke').glob('*.whl')); assert len(wheels)==1, f'expected one wheel, found {len(wheels)}'; names=set(zipfile.ZipFile(wheels[0]).namelist()); required={'pii_zh/taxonomy/taxonomy.yaml','pii_zh/taxonomy/presidio_mapping.yaml','pii_zh/data/validators/cn_vehicle_plate.py','pii_zh/rules/cn_common_v6.py','pii_zh/cascade/service_profiles.py','pii_zh/cascade/routing.py','pii_zh/cli.py','pii_zh/service/app.py'}; missing=sorted(required-names); leaked=sorted(name for name in names if name.startswith(('tests/','reports/','configs/','scripts/','examples/'))); assert not missing, f'missing wheel entries: {missing}'; assert not leaked, f'repository-only payloads leaked into wheel: {leaked}'"

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
