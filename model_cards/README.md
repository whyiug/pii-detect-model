---
language:
  - zh
pipeline_tag: token-classification
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
datasets: []
tags:
  - pii
  - privacy
  - qwen3
  - bidirectional-attention
library_name: transformers
---

# `<org>/zh-pii-qwen3-0.6b-bi-v1.0.0` release-candidate template

> RC blocker: replace every placeholder and attach measured three-seed evidence before building a
> publishable package. This template makes no trained-model or quality claim.

## Model description

This is a Simplified Chinese PII token-classification model based on an immutable Qwen3 Base
revision. `Qwen3BiForTokenClassification` replaces causal attention with padding-aware full
attention and predicts BIO labels. Record the exact base revision, taxonomy version, training
recipe, selected seed, tokenizer hash, and release tag here.

## Intended use and out-of-scope use

Describe approved offline and service use cases, the expected Presidio integration, and the
supported languages/domains. This component is not a legal-compliance guarantee. It must not be
used as the sole control for high-impact decisions, identity verification, or processing outside
the evaluated domains.

## Labels, calibration, and thresholds

The authoritative label table is `id2label.json`; definitions are in `taxonomy.yaml`. Calibration
parameters and policy-independent score thresholds are in `calibration.json` and
`thresholds.yaml`. Add the Presidio mapping version and explain the `high_recall`, `balanced`, and
`high_precision` policies.

## Training data and teachers

Summarize source categories and sampling proportions without including rows, entity values,
customer templates, prompts, or responses. Cite `data_provenance.json`, `teacher_provenance.json`,
and `training_manifest.json`. State explicitly that `evaluation_only` sources were never used for
training, prompting, distillation, calibration, or threshold selection.

## Evaluation

Report all required public/private, hard-negative, noise, long-document, memorization, and
Presidio end-to-end results with confidence intervals. Include at least three independently seeded
runs and link every model-index result to `evaluation_report.json`. Do not publish a best-seed-only
claim.

## Limitations and privacy risks

Document measured limitations for dialects, OCR noise, nested entities, implicit identity,
non-text inputs, domain-shifted formats, long documents, and calibration drift. Summarize canary
extraction and memorization tests without revealing canaries or test rows.

## Loading custom code safely

Loading this architecture executes the repository's Python files. Review them, pin an immutable
release revision, and require safetensors:

```python
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

model_id = "<org>/zh-pii-qwen3-0.6b-bi"
revision = "v1.0.0"
# Load the serialized backend graph directly. Model-specific AutoTokenizer
# reconstruction does not preserve this model's character-boundary contract.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    model_id,
    revision=revision,
    fix_mistral_regex=False,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    use_safetensors=True,
)
```

Do not replace the generic tokenizer loader above with `AutoTokenizer`: exact per-character offsets
are part of the trained and attested model contract.

Record the tested lower/upper Transformers versions. The remote code performs no network, shell,
or dynamic-download operation; `checksums.txt` covers the code and every public artifact.

## License, citation, and security reporting

The declared model license applies only after the release-specific source registry and legal
review pass. See `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and the provenance files. Report
security, privacy, memorization, and provenance issues through the private route in `SECURITY.md`.
