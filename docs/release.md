# Local release engineering

Frozen evaluation provenance and publication-safe rule-baseline evidence are documented in
[`evaluation_provenance.md`](evaluation_provenance.md). Model evaluation in release mode must bind
the dataset, prediction, and model training manifests; a metric JSON without those identities is
development evidence only.

The release tools are deliberately local-only. They never train, download a base model, contact
the Hugging Face Hub, or upload an artifact. A model release remains blocked until the immutable
checkpoint and all evidence are supplied by an approved training/evaluation run.

## 1. Prepare isolated inputs

Use a checkpoint directory containing exactly the release checkpoint inputs:

```text
checkpoint/
├── added_tokens.json        # optional; copied when emitted by the tokenizer
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

When `added_tokens.json` is present, the builder copies it and the release gate requires its hash
to match `training_manifest.json.output_artifact.files`. It remains optional for tokenizers that do
not emit it; no unbound tokenizer metadata is admitted to the release file set.

Only safetensors are accepted. If the checkpoint tree also contains a pickle weight, optimizer or
scheduler state, gradient/cache artifact, raw data table, API prompt/response, or customer data,
the builder fails even though that file is not on the copy allowlist. Prepare a clean export rather
than deleting evidence from the original training run.

The evidence directory must contain:

```text
evidence/
├── taxonomy.yaml
├── id2label.json
├── calibration.json
├── thresholds.yaml
├── training_manifest.json
├── data_provenance.json
├── teacher_provenance.json
├── evaluation_report.json
├── model-index.yml
└── sbom.cdx.json
```

Generate the deterministic CycloneDX SBOM from the committed dependency lock:

```bash
python scripts/generate_sbom.py --lockfile uv.lock --pyproject pyproject.toml \
  --output evidence/sbom.cdx.json
```

`data_provenance.json.sources[]` records `source_id`, the admitted pool, and a positive
`sample_count`/`sampled_records`. `training_manifest.json` records every base/data/teacher source ID.
`teacher_provenance.json` records every teacher source ID or explicitly states that no teacher was
used. Every referenced ID must exist in `configs/data/source_registry.yaml`, have an immutable
revision, explicit compatible license, completed review, and
`public_weight_training_allowed: true`.

An `evaluation_only` source may appear in evaluation metadata, but it must never appear in any of
those three training/provenance files as a training pool, training path, or referenced source.

## 2. Record quality evidence without best-seed selection

`evaluation_report.json` needs at least three unique seed runs. Each run must contain non-empty
`metrics` and `quality_gate_passed: true`. The aggregate gate must name explicit criteria:

```json
{
  "release_decision": "passed",
  "seeds": [
    {"metrics": {"strict_span_f1": 0.0}, "quality_gate_passed": true, "seed": 11},
    {"metrics": {"strict_span_f1": 0.0}, "quality_gate_passed": true, "seed": 22},
    {"metrics": {"strict_span_f1": 0.0}, "quality_gate_passed": true, "seed": 33}
  ],
  "quality_gate": {
    "criteria": [
      {
        "name": "replace-with-approved-criterion",
        "operator": ">=",
        "passed": true,
        "threshold": 0.0,
        "value": 0.0
      }
    ],
    "status": "passed"
  }
}
```

The zeroes above are schema examples, not passing project thresholds. Approved thresholds and real
measurements must come from the frozen evaluation plan. Missing, incomplete, one-seed, or failed
evidence always produces an `RC_BLOCKED_*` result; there is no `allow-incomplete` flag.

## 3. Build the local HF directory

Start from a completed model card, not the placeholder template:

```bash
python scripts/build_release.py \
  --checkpoint-dir /path/to/clean/checkpoint \
  --evidence-dir /path/to/release/evidence \
  --model-card /path/to/completed/README.md \
  --third-party-notices /path/to/release-specific/THIRD_PARTY_NOTICES.md \
  --security-policy /path/to/release-ready/SECURITY.md \
  --output-dir release/hf_model/zh-pii-qwen3-0.6b-bi-v1.0.0
```

The builder copies LICENSE/NOTICE/security files, generates the Qwen3Bi `auto_map`, and extracts
the reviewed `Qwen3BiConfig` and model implementation into Hugging Face remote-code files. It then
writes deterministic SHA-256 entries for every file except `checksums.txt` itself. Staging is
atomic. `--force` only replaces the named output directory; it does not alter checkpoint/evidence
inputs, and it is restricted to a child of this repository's `release/` directory. LICENSE,
NOTICE, third-party notices, and SECURITY default to the repository files, but each has an explicit
override flag. The current repository third-party/security templates intentionally block a real RC
until release-specific attribution and a tested private reporting route are supplied.

The checksum file uses the standard `sha256sum` two-space format and is independently reproducible:

```bash
(cd release/hf_model/<release-name> && sha256sum --check checksums.txt)
```

The release gate performs the same verification and additionally rejects missing, duplicate,
absolute, traversal, stale, and uncovered entries.

## 4. Scan and gate

Run the redacting public-artifact scanner independently when diagnosing a privacy failure:

```bash
python scripts/scan_public_artifacts.py release/hf_model/<release-name>
```

The scanner reports only a SHA-256 fingerprint, file, line, and finding type—never the matched
value. A deliberately synthetic PII canary can be allowlisted by that fingerprint using a reviewed
copy of `release/synthetic-canaries.example.json`. The allowlist requires `synthetic: true` and a
purpose. It cannot suppress secret findings.

Normalize the dependency scanner output as follows (pip-audit JSON is also accepted):

```json
{
  "findings": [
    {"component": "package-name", "id": "CVE-or-GHSA", "severity": "high", "status": "open"}
  ],
  "generated_at": "UTC timestamp",
  "scan_complete": true,
  "scanner": "scanner name and version",
  "sbom_sha256": "SHA-256 of the exact packaged sbom.cdx.json"
}
```

High, critical, medium/moderate, and unknown unresolved findings block. A narrow exception must
match both vulnerability and component and provide approver, substantive justification, expiry,
and compensating controls. The JSON schema and empty default are under `release/`; expired or
blanket exceptions fail closed.

```bash
python scripts/release_gate.py \
  --artifact release/hf_model/<release-name> \
  --source-registry configs/data/source_registry.yaml \
  --dependency-scan /path/to/dependency-scan.json \
  --dependency-exceptions release/dependency-security-exceptions.example.json \
  --json-output release/gate-report.json
```

Publication automation may proceed only on exit code 0 and `status: PASS`. That result covers
machine-checkable evidence only. Complete the human/legal/privacy checks in
`release/release_checklist.md`, verify `checksums.txt` from a clean checkout, use an immutable Hub
revision, and publish through a separately authorized workflow. These repository tools contain no
upload command.

## Docker templates

Training and inference Dockerfiles are offline-by-default: `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, no weight is copied or downloaded, and no port is exposed. Mount an
approved local checkpoint read-only at runtime and override the command as documented in
`docker/README.md`. Do not bake `/data1/models` or credentials into an image.
