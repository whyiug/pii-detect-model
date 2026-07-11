# pii-zh-qwen

`pii-zh-qwen` is a reproducibility-first Simplified Chinese PII detection project. The current
artifact is a **community research release candidate**, not a production release: it combines a
Qwen3-0.6B token classifier using padding-aware full attention with deterministic Chinese format
rules, frozen validation calibration, deterministic fusion, and structured refinement.

## Current research RC

The trained checkpoint starts from `Qwen/Qwen3-0.6B-Base`, initializes full attention through the
audited JPT-to-Full path, and predicts the 24 core entity types in the versioned taxonomy. Training
used 16,000 licensed deterministic-synthetic documents; 2,000 disjoint synthetic documents were
used for validation, early stopping, and per-seed calibration. Seeds 13, 42, and 97 all passed the
preregistered validation gate. Seed 42 was selected by the fixed conventional-seed rule before any
holdout access, not by choosing the best holdout result.

The frozen system is bound to the following runtime and post-processing contract:

- Transformers 5.13.1 with the serialized fast-tokenizer graph loaded directly;
- `cn_common_v5` deterministic rules and `deterministic_fusion_v1`;
- validation-fitted calibration applied to holdouts without refitting;
- `structured_refinement_v4`;
- exact model, tokenizer, dataset, prediction, calibration, and implementation hashes in the
  release evidence.

### Frozen results

These are blind, apply-only results for the complete seed-42 system. Confidence intervals are
95% document-level percentile bootstrap intervals with 1,000 samples. PII Bench contains no
PII-free documents, so a false-positive document rate cannot be estimated for those two subsets.

| Frozen suite | Documents | Strict micro F1 (95% CI) | Strict macro F1 | Character F1 | Relaxed micro F1 | PII-free document FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic v1.3 test | 2,000 | 0.9072 (0.8973–0.9167) | 0.8803 | 0.9464 | 0.9311 | **23.83% (143/600)** |
| PII Bench zh formal | 5,000 | 0.8265 (0.8201–0.8324) | 0.4969 | 0.8702 | 0.8543 | not estimable |
| PII Bench zh chat | 3,000 | 0.7406 (0.7320–0.7496) | 0.4078 | 0.7314 | 0.7685 | not estimable |

Across all 10,000 frozen documents, the count-pooled strict micro F1 is 0.8148. Bootstrap
intervals are reported per suite and are not naively pooled.

The three-seed synthetic validation strict micro F1 values were 0.9888, 0.9800, and 0.9830 for
seeds 13, 42, and 97 respectively; every seed passed all preregistered validation criteria.

> **Non-production warning:** the held-out synthetic PII-free document false-positive rate is
> **23.83%**, far above a reasonable production guardrail. No private cross-domain, long-document,
> production Presidio end-to-end, or customer-data evaluation has been completed. This RC must not
> be presented as production-ready, a compliance guarantee, or a drop-in redaction control.

## Model and data boundaries

- Canonical truth is a character-level, left-closed/right-open span. BIO labels are derived
  training views, never the sole stored truth.
- Detection, risk, and policy are separate layers. A detected entity is not automatically an
  instruction to redact it.
- Structured identifiers are checked with format or checksum rules where possible; the neural
  model does not replace those validators.
- Public, private-enterprise, evaluation-only, and quarantined data are physically and logically
  isolated. Evaluation-only data never enters training, prompting, distillation, calibration, or
  threshold selection.
- Real customer PII must not be committed, logged, sent to third-party APIs, or included in public
  artifacts.
- The 24-label taxonomy does not imply coverage of every identity-bearing phrase, dialect, OCR
  artifact, nested entity, document length, or deployment domain.

## Repository layout

```text
configs/                 frozen data, training, evaluation, and release contracts
docs/                    annotation, provenance, isolation, and release documentation
src/pii_zh/data/         data contracts and synthetic/alignment workflows
src/pii_zh/models/       Qwen3 full-attention token-classification implementation
src/pii_zh/rules/        deterministic Chinese PII rules and validators
src/pii_zh/fusion/       fusion, calibration application, and refinement
src/pii_zh/taxonomy/     24-label taxonomy and Presidio mapping
scripts/                 reproducible training, evaluation, evidence, and release tools
tests/                   unit, release, and integration contracts
release/                 local release schemas and human checklist
pii-detect-plan.md       implementation and evaluation plan
```

## Reproducible workflows

The command-line entry points are intentionally composable and expose their full arguments through
`--help`:

```bash
python scripts/materialize_synthetic_data.py --help
python scripts/train.py --help
python scripts/predict.py --help
python scripts/predict_rules.py --help
python scripts/fuse_predictions.py --help
python scripts/calibrate_predictions.py --help
python scripts/apply_calibration.py --help
python scripts/refine_predictions.py --help
python scripts/evaluate.py --help
python scripts/summarize_community_rc.py --help
python scripts/summarize_system_evaluations.py --help
python scripts/assemble_release_evidence.py --help
```

The preregistered decision contract is
`configs/evaluation/synthetic_v1_3_community_rc_amendment_v6.yaml`. Release-mode evaluation binds
the dataset, prediction, and training manifests; an unbound metric file is development evidence
only. See [`docs/release.md`](docs/release.md) for a complete, relative-path example that assembles
evidence, builds a local Hugging Face directory, scans it, and runs the machine gate.

## Development setup

Python 3.10 or newer is required. The attested release runtime uses Transformers 5.13.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[core,dev]"
pytest
ruff check src tests
mypy src/pii_zh
```

Training and Presidio integrations remain optional dependency groups:

```bash
python -m pip install -e ".[training]"
python -m pip install -e ".[core,presidio]"
```

For the release tokenizer, load the serialized backend directly with
`PreTrainedTokenizerFast.from_pretrained(..., fix_mistral_regex=False)`. Do not substitute
`AutoTokenizer`: model-specific reconstruction does not preserve the attested Unicode codepoint
boundary graph. Exact loading guidance is in [`docs/release.md`](docs/release.md).

## Taxonomy API

```python
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy

taxonomy = load_taxonomy()
mapping = load_presidio_mapping(taxonomy=taxonomy)

assert len(taxonomy.core_label_names) == 24
assert "CN_RESIDENT_ID" in taxonomy.core_label_names
assert mapping.model_to_presidio["CN_RESIDENT_ID"] == "CN_ID_CARD"
```

The packaged loader rejects duplicate labels, unknown risk tiers, incomplete mappings, auxiliary
labels that leak into output mappings, and version mismatches.

## Security, privacy, and license

Do not report a suspected vulnerability in a public issue; follow [`SECURITY.md`](SECURITY.md).
Examples and tests must use clearly synthetic values, and logs must contain hashes or counts rather
than raw text. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing code,
data, or weights.

Repository-authored code and documentation are provided under Apache-2.0 unless a file states
otherwise. That license does not grant rights to third-party datasets, model weights, generated
artifacts, or customer data; each source requires an independent provenance and license decision.
