# Local release engineering

The current trained artifact has passed its preregistered three-seed validation gate and has been
evaluated once on the frozen public suites. It is a **community research release candidate**, not a
production approval and not evidence that an artifact has been uploaded. Frozen evaluation
provenance is documented in [`evaluation_provenance.md`](evaluation_provenance.md).

The release tools are local-only. They do not train, download a model, contact the Hugging Face Hub,
or upload an artifact. Release-mode evaluation and evidence assembly fail closed unless model,
dataset, prediction, calibration, and implementation identities are cryptographically bound.

All commands below use repository-relative example paths under `runs/research-rc` and
`release/staging`. Keep actual large checkpoints outside Git if needed, but stage only a clean,
read-only release input with the same file contract.

## 1. Prepare a clean checkpoint

The selected seed-42 checkpoint directory must contain exactly the release checkpoint inputs:

```text
runs/research-rc/seed-42/checkpoint/
├── added_tokens.json
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

`added_tokens.json` is optional only when the tokenizer did not emit it. When present, its digest
must match `training_manifest.json.output_artifact.files`. Only safetensors weights are accepted.
The builder rejects pickle weights, optimizer or scheduler state, gradients, caches, raw datasets,
API prompts or responses, customer material, and private logs anywhere in the checkpoint tree,
even if such a file is not on the copy allowlist. Create a clean export; do not delete provenance
from the original run.

The selected training manifest must bind the immutable base revision, tokenizer files, taxonomy,
train and validation manifests, seed 42 recipe, JPT initializer, full-attention output, and final
artifact hashes.

## 2. Assemble release evidence

`scripts/assemble_release_evidence.py` is the canonical bridge from attested run artifacts to the
publication-safe evidence set. It consumes aggregate, self-hashed reports—not dataset or prediction
rows—and enforces all of the following before writing output:

- the exact validation seeds 13, 42, and 97 all passed the preregistered gate;
- seed 42 is the preselected full-attention system;
- synthetic test, PII Bench formal, and PII Bench chat identities and frozen metrics are complete;
- calibration was fitted on validation and only applied to holdouts;
- model weights, tokenizer, training manifest, calibration, source provenance, and system identity
  agree;
- emitted JSON/YAML is deterministic, path-free, and contains no raw text, entity values, document
  identifiers, or record-level data.

Run it with relative paths:

```bash
python scripts/assemble_release_evidence.py \
  --checkpoint-dir runs/research-rc/seed-42/checkpoint \
  --taxonomy src/pii_zh/taxonomy/taxonomy.yaml \
  --training-manifest runs/research-rc/seed-42/training_manifest.json \
  --calibration runs/research-rc/seed-42/calibration.json \
  --calibration-diagnostics runs/research-rc/seed-42/calibration.diagnostics.json \
  --community-report runs/research-rc/reports/community_rc_validation_report.json \
  --system-summary runs/research-rc/reports/system_frozen_evaluation_summary.json \
  --data-provenance runs/research-rc/provenance/data_provenance.json \
  --teacher-provenance runs/research-rc/provenance/teacher_provenance.json \
  --output-dir release/staging/evidence \
  --release-name zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1
```

This writes `taxonomy.yaml`, `id2label.json`, `calibration.json`, `thresholds.yaml`,
`training_manifest.json`, `data_provenance.json`, `teacher_provenance.json`,
`evaluation_report.json`, and `model-index.yml`. Generate the tenth required file, a deterministic
CycloneDX SBOM, from the committed lock:

```bash
python scripts/generate_sbom.py \
  --lockfile uv.lock \
  --pyproject pyproject.toml \
  --output release/staging/evidence/sbom.cdx.json
```

Every training source referenced by `training_manifest.json`, `data_provenance.json`, or
`teacher_provenance.json` must exist in `configs/data/source_registry.yaml`, have an immutable
revision, a compatible license, a completed review, and
`public_weight_training_allowed: true`. Evaluation-only sources belong only in evaluation
metadata; they must not appear as training, prompting, distillation, calibration-fit, or threshold
selection sources.

## 3. Preserve the tokenizer/runtime contract

The attested inference runtime is Transformers **5.13.1** with
`serialized_fast_tokenizer_v1`. Load the tokenizer's serialized backend graph directly:

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1",
    fix_mistral_regex=False,
)
```

Do not replace this with `AutoTokenizer`. Reconstructing the model-specific tokenizer can discard
the serialized Unicode-codepoint boundary graph and change offsets even when model weights are
unchanged. A release smoke test must load with Transformers 5.13.1, verify the boundary probe, and
exercise offline source-checkpoint/release-package logit parity. Other Transformers versions
require a separate compatibility audit; they inherit no release claim automatically.

## 4. Record validation and frozen evaluation honestly

The preregistered synthetic validation gate passed for seeds 13, 42, and 97. Their strict micro F1
values were 0.9888, 0.9800, and 0.9830; all per-seed strict micro/macro, PII-free FPR, Tier-0/Tier-1
recall, and calibration criteria passed.

After that gate unlocked holdout access, the unchanged seed-42 system was evaluated once with its
frozen validation calibration applied without refitting:

| Frozen suite | Documents | Strict micro F1 (95% bootstrap CI) | Strict macro F1 | PII-free document FPR |
| --- | ---: | ---: | ---: | ---: |
| Synthetic v1.3 test | 2,000 | 0.9072 (0.8973–0.9167) | 0.8803 | **23.83% (143/600)** |
| PII Bench zh formal | 5,000 | 0.8265 (0.8201–0.8324) | 0.4969 | not estimable |
| PII Bench zh chat | 3,000 | 0.7406 (0.7320–0.7496) | 0.4078 | not estimable |

`evaluation_report.json` must retain the three validation seed records, aggregate quality gate,
release scope, selected seed, full frozen-suite results, confidence intervals, and their source
hashes. A metric JSON without those bindings is development evidence only.

The synthetic PII-free FPR is a release-critical limitation. It must not be hidden by the strong
positive-span scores, by validation metrics, or by the absence of negative documents in PII Bench.

## 5. Build the local Hugging Face directory

Start from candidate research-RC model-card and release-specific legal/security files. Human
publication review remains pending until the checklist records it:

```bash
python scripts/build_release.py \
  --checkpoint-dir runs/research-rc/seed-42/checkpoint \
  --evidence-dir release/staging/evidence \
  --model-card model_cards/README.md \
  --third-party-notices THIRD_PARTY_NOTICES.md \
  --security-policy SECURITY.md \
  --output-dir release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1
```

The builder copies the license and notice files, generates the Qwen3Bi `auto_map`, extracts the
reviewed config/model implementation into Hugging Face remote-code files, and writes deterministic
SHA-256 entries for every file except `checksums.txt` itself. Staging is atomic. `--force` may only
replace the named child directory under `release/`; it does not mutate checkpoint or evidence
inputs.

Verify the checksums independently:

```bash
(cd release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1 && sha256sum --check checksums.txt)
```

## 6. Scan dependencies and public artifacts

Generate a fresh dependency scan from the environment represented by the immutable lock. Normalize
its result to `release/dependency-scan.schema.json`; the normalized document must name the scanner
and version, mark the scan complete, bind the exact packaged SBOM SHA-256, and list each finding's
component, advisory ID, severity, and status.

High, critical, medium/moderate, and unknown unresolved findings block. An exception must match both
the component and advisory and include an approver, substantive justification, expiry, and
compensating controls. Expired, broad, or incomplete exceptions fail closed.

Run the redacting scanner separately:

```bash
python scripts/scan_public_artifacts.py \
  --json-output release/staging/public-scan.json \
  release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1
```

The public report contains only file, line, and finding type—never the matched value or its
low-entropy hash. Internally, a deliberately synthetic PII canary may be matched against a reviewed
SHA-256 allowlist following `release/synthetic-canaries.example.json`; secret findings can never be
suppressed.

## 7. Run the machine gate

```bash
python scripts/release_gate.py \
  --artifact release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1 \
  --source-registry configs/data/source_registry.yaml \
  --dependency-scan release/staging/dependency-scan.json \
  --dependency-exceptions release/dependency-security-exceptions.example.json \
  --json-output release/staging/gate-report.json
```

Local machine-verifiable downstream steps may proceed only when the command exits zero and reports
`status: PASS`. That result covers machine-checkable integrity, provenance, privacy scanning,
evidence, and dependency policy; it is not production approval and it does not authorize an
upload. Publication remains blocked until `SECURITY.md` names a tested private route and the human
checklist is approved.

## 8. Generate the aggregate research-RC report artifact

Only after the schema-v2 gate receipt passes, generate the path-free aggregate report input from
the exact staged package. The generator independently verifies package checksums, packaged
evidence bytes, SBOM and dependency-scan binding, provenance self-hashes, and the gate receipt's
content identity:

```bash
python scripts/generate_release_report.py \
  --evaluation-report release/staging/evidence/evaluation_report.json \
  --training-manifest release/staging/evidence/training_manifest.json \
  --data-provenance release/staging/evidence/data_provenance.json \
  --teacher-provenance release/staging/evidence/teacher_provenance.json \
  --dependency-scan release/staging/dependency-scan.json \
  --dependency-exceptions release/dependency-security-exceptions.example.json \
  --release-gate release/staging/gate-report.json \
  --artifact-dir release/hf_model/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1 \
  --output release/staging/reports/artifact.json
```

`artifact.json` contains aggregate counts, metrics, provenance conclusions, source hashes, and
machine-gate results only. It must not contain raw text, entity values, document identifiers,
absolute paths, or record-level data. Rendered HTML is a local decision artifact; it does not
authorize publication or production deployment.

## Research RC and production human gate

The current evidence supports only a community research RC. Production approval remains a separate
human decision and is currently withheld because there is no private cross-domain evaluation, no
long-document evaluation, no production Presidio end-to-end evaluation, and no customer-data or
canary deployment evidence. The measured 23.83% synthetic PII-free document FPR is itself a strong
reason not to deploy this RC as an automatic redaction control.

Before any production claim, authorized reviewers must define deployment-specific precision and
recall guardrails, evaluate representative private data under the privacy policy, test the complete
Presidio/policy path, assess long documents and operational failure modes, review model-card and
legal claims, and approve rollback and monitoring. These human checks cannot be waived by a machine
gate result.

Complete [`../release/release_checklist.md`](../release/release_checklist.md), reproduce checksums
from a clean checkout, and publish only through a separately authorized workflow at an immutable
revision. No repository release tool uploads files or creates/signs a tag.

## Docker templates

Training and inference Dockerfiles are offline by default: `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; no model weight is baked in and no port is exposed. Mount an approved
local checkpoint read-only as described in [`../docker/README.md`](../docker/README.md). Never bake
credentials or a shared model cache into an image.
