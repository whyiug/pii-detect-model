# pii-zh-qwen

`pii-zh-qwen` is a reproducibility-first project for Simplified Chinese
personally identifiable information (PII) detection. The intended system combines a Qwen
token-classification recognizer, deterministic Chinese format validators, an existing ERNIE
recognizer, and tenant policy through Presidio.

## Current status

This repository is **pre-alpha**. It defines the phase-A taxonomy, annotation contract, data
isolation rules, and configuration templates, alongside early data/model implementation work. It
does not contain trained weights or a validated ERNIE/Qwen benchmark. No production, internal-gold,
customer, shadow, or canary result is claimed. The engineering plan is maintained in
[`pii-detect-plan.md`](pii-detect-plan.md).

The intended public checkpoint is `Qwen/Qwen3-0.6B-Base` adapted for bidirectional/full attention
token classification. A 1.7B quality checkpoint is conditional on measured end-to-end gains. These
are design targets, not artifacts currently present in this repository.

## Design boundaries

- The canonical truth is a character-level, left-closed/right-open span. BIO labels are derived
  training views, never the sole stored truth.
- Detection, risk, and policy are separate layers. A detected entity is not automatically an
  instruction to redact it.
- Structured identifiers should be verified by format/checksum rules where possible. The semantic
  model is not intended to replace all rules.
- Public, private-enterprise, evaluation-only, and quarantined data are physically and logically
  isolated. Evaluation-only data must never be used for training, prompting, distillation,
  calibration, or threshold selection.
- Real customer PII must not be committed, logged, sent to third-party APIs, or included in public
  artifacts.
- This software is a technical component, not a legal-compliance guarantee.

## Repository layout

```text
configs/                 versioned configuration templates
docs/                    annotation, data-contract, and test-isolation specifications
src/pii_zh/data/         data contracts and synthetic/alignment building blocks
src/pii_zh/models/       experimental model architecture building blocks
src/pii_zh/taxonomy/     machine-readable taxonomy and Presidio mapping
tests/unit/              phase-A contract tests
pii-detect-plan.md       full implementation and release plan
```

Remaining phases add complete conversion, training, evaluation, calibration, rule, fusion, and
Presidio workflows without placing private data in Git.

## Development setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[core,dev]"
pytest
ruff check src tests
mypy src/pii_zh
```

Training and Presidio dependencies are deliberately optional:

```bash
python -m pip install -e ".[training]"
python -m pip install -e ".[core,presidio]"
```

Dependency ranges are compatibility bounds, not a final release lock. A publishable experiment
must additionally record an immutable environment lock, base-model revision, tokenizer hash,
dataset manifest hashes, random seeds, and teacher provenance.

## Taxonomy API

```python
from pii_zh.taxonomy import load_presidio_mapping, load_taxonomy

taxonomy = load_taxonomy()
mapping = load_presidio_mapping(taxonomy=taxonomy)

assert "CN_RESIDENT_ID" in taxonomy.core_label_names
assert mapping.model_to_presidio["CN_RESIDENT_ID"] == "CN_ID_CARD"
```

The packaged loader rejects duplicate labels, unknown risk tiers, incomplete mappings, auxiliary
labels that leak into output mappings, and version mismatches.

## Data pools

Only samples that pass both a quality gate and an explicit public-weight-training license gate may
enter `data/processed/public_release_pool`. Customer/internal material belongs in
`private_enterprise_pool`; frozen benchmarks belong in `evaluation_only`; unresolved provenance,
license, or quality belongs in `quarantined`. See `docs/data_contract.md` and
`docs/test_isolation.md` before adding any data workflow.

## Security and responsible use

Do not report a suspected vulnerability in a public issue. Follow [`SECURITY.md`](SECURITY.md).
Examples and tests must use clearly synthetic values, and logs must contain hashes/counts rather
than raw text. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing code,
data, or weights.

## License

Repository-authored code and documentation are provided under Apache-2.0 unless a file states
otherwise. That license does not grant rights to third-party datasets, model weights, API output,
customer data, or future generated artifacts; each source requires an independent provenance and
license decision.
