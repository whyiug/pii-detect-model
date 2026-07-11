---
language:
  - zh
pipeline_tag: token-classification
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-0.6B-Base
tags:
  - pii
  - privacy
  - qwen3
  - token-classification
  - bidirectional-attention
  - research-release-candidate
production_ready: false
---

# zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1

This is a research release candidate for Simplified Chinese PII token classification. It adapts
`Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` to
padding-aware, non-causal full attention and a 49-ID BIO classification head covering 24 core
entity types plus `O`.

**Release status: `production_ready=false`.** The intended Hub repository is
`whyiug/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1`, and the planned immutable release revision is
`synthetic-v1.3-rc1`. The checkpoint is stored as `model.safetensors`. This RC is suitable for
authorized research, reproducibility work, and locally evaluated prototypes; it is not a
production privacy control or a compliance guarantee.

## What is in this release

The staged release checkpoint is a `Qwen3BiForTokenClassification` model. Its custom architecture
replaces Qwen3 causal attention with full attention over non-padding tokens. The final training
stage used LoRA and was initialized from an attested JPT token-classification stage. Training used
a maximum sequence length of 128. The release contains custom Python code for the architecture,
so loading the model requires reviewing and opting into that code with `trust_remote_code=True`.

The core taxonomy contains these 24 output entity types:

`PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`, `DATE_OF_BIRTH`,
`CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`,
`SOCIAL_SECURITY_NUMBER`, `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`,
`VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER`,
`WECHAT_ID`, `QQ_NUMBER`, `ALIPAY_ACCOUNT`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`,
`DEVICE_ID`, `GEO_COORDINATE`, and `SECRET`.

`SECRET` covers security-sensitive material and is not necessarily personal information. The
authoritative definitions and risk tiers are in [taxonomy.yaml](./taxonomy.yaml); the exact 49-ID
mapping (`O` plus `B-`/`I-` for each core entity) is in [id2label.json](./id2label.json).

## A critical distinction: weights versus the evaluated system

The frozen test results below belong to one fixed system:

`seed-42 model + cn_common_v5 rules + deterministic_fusion_v1 + frozen validation calibration + structured_prediction_refinement_v4`.

The rules, fusion, calibration, and refinement stages materially affect the output. Consequently,
the figures in this card and in [model-index.yml](./model-index.yml) are **system-level results,
not standalone checkpoint results**. Loading only the Hugging Face weights produces model-only
scores and must not be expected to reproduce these figures. The calibration identity, component
hashes, selected seed, and apply-only holdout policy are bound in
[evaluation_report.json](./evaluation_report.json). The packaged calibration and thresholds are
also recorded in [calibration.json](./calibration.json) and
[thresholds.yaml](./thresholds.yaml).

No model, rule, fusion, calibration, or refinement behavior was changed after the validation gate
unlocked the holdouts. Calibration was fitted on synthetic validation and applied to every test
suite without refitting.

## Training and evaluation data

All gradient-training records were deterministically generated synthetic records admitted to the
public release pool. No private, tenant, customer, or production records were used.

| Split or source | Records | Permitted use in this release |
|---|---:|---|
| Synthetic v1.3 train | 16,000 | Gradient training |
| Synthetic v1.3 validation | 2,000 | Early stopping, per-seed validation gate, calibration, and threshold selection |
| Synthetic v1.3 frozen test | 2,000 | Evaluation only; zero training use |
| PII Bench zh formal | 5,000 | Evaluation only |
| PII Bench zh chat | 3,000 | Evaluation only |

PII Bench zh was not used for gradient training, early stopping, calibration, threshold selection,
prompting, or distillation. It remained evaluation-only.

A locally run Qwen3-8B model generated placeholder-only, variable-slot template candidates
upstream. Humans reviewed 70 candidates and accepted 53 before deterministic synthetic value
rendering. Qwen3-8B did not
annotate spans, did not pseudo-label records, and its unreviewed raw output was not used for
training. `external_api=false`: no external API teacher was used. Detailed source admission,
review counts, roles, hashes, and split usage are in
[data_provenance.json](./data_provenance.json),
[teacher_provenance.json](./teacher_provenance.json), and
[training_manifest.json](./training_manifest.json).

## Frozen system evaluation

The selected seed was fixed as seed 42 before holdout access. Metrics use strict span matching:
both the entity label and character boundaries must match. Per-suite 95% intervals are
document-level percentile bootstrap intervals with 1,000 samples. The pooled figures aggregate
counts over all 10,000 frozen test documents; suite intervals were not naively pooled.

| Frozen suite | Documents | Strict span precision | Strict span recall | Strict span micro F1 | 95% F1 interval | PII-free document FPR |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic v1.3 test | 2,000 | 0.9203149606 | 0.8943985308 | **0.9071716858** | [0.8973296575, 0.9166545105] | **0.2383333333** |
| PII Bench zh formal | 5,000 | 0.9553601340 | 0.7283233304 | **0.8265343091** | [0.8200758498, 0.8323885963] | Not available: no PII-free documents |
| PII Bench zh chat | 3,000 | 0.9439425051 | 0.6093584305 | **0.7406154342** | [0.7319756938, 0.7496436645] | Not available: no PII-free documents |
| Pooled frozen suites | 10,000 | **0.9470102577** | **0.7149170853** | **0.8147574153** | Not pooled | Not pooled |

The synthetic test contained 600 PII-free documents; the frozen system emitted at least one false
positive on 143 of them. Its PII-free false-positive rate was 0.2383333333, with a 95% interval of
[0.2049686874, 0.2720092598]. This high false-positive rate is a material blocker for production
use. Formal and chat recall also show substantial domain-dependent misses.

The complete aggregate counts, per-suite macro/relaxed/character metrics, bootstrap settings,
dataset revisions, and system hashes are in [evaluation_report.json](./evaluation_report.json).
The machine-readable leaderboard entries are in [model-index.yml](./model-index.yml).

### Validation gate only — not test performance

Three independently trained seeds were evaluated on the 2,000-record synthetic **validation**
split to enforce the preregistered release gate. These numbers supported model/system selection and
holdout unlocking. They are not frozen-test results and must not be presented as generalization
performance.

| Seed | Validation strict micro F1 | Validation strict macro F1 | Validation PII-free FPR |
|---:|---:|---:|---:|
| 13 | 0.9887766554 | 0.9887176214 | 0.0000000000 |
| 42 | 0.9799860042 | 0.9831315969 | 0.0000000000 |
| 97 | 0.9829894032 | 0.9854827110 | 0.0000000000 |

All three seeds passed the validation gate. Seed 42 was selected by a fixed conventional-seed rule,
not by choosing the best observed validation score. The gate criteria and all three provenance
chains are in [evaluation_report.json](./evaluation_report.json).

## Intended use

Appropriate uses are limited to:

- authorized research on Chinese PII detection and character-boundary token classification;
- reproducibility checks against the attached manifests and checksums;
- offline prototyping where users perform their own domain, tenant, threshold, and error analysis;
- evaluation of a locally assembled rules-and-model system on non-production or properly governed
  data.

The model may support a Presidio recognizer in the accompanying source project, but this RC does
not include evidence for production Presidio end-to-end quality or latency.

## Out-of-scope and unsafe uses

Do not use this RC as the sole mechanism for automatic disclosure prevention, anonymization,
regulatory compliance, authentication, authorization, identity verification, fraud decisions, or
high-impact decisions about people. Do not treat a non-detection as proof that text is free of
personal or security-sensitive information. Human review and defense-in-depth controls remain
necessary wherever missed or over-redacted content can cause harm.

This release does not authorize processing data that the operator is not permitted to access. It
also does not authorize probing public services or third-party systems.

## Known limitations and unmeasured coverage

The following evidence is absent, so `production_ready` remains `false`:

- private-enterprise gold evaluation;
- tenant holdout and time holdout evaluation;
- a dedicated long-document quality benchmark;
- production Presidio end-to-end quality and latency measurement;
- demographic fairness evaluation.

Additional limitations include:

- Training is synthetic-only. Real formatting, OCR artifacts, dialects, code switching, and domain
  conventions may differ substantially.
- The 128-token training length and missing long-document benchmark make chunking behavior and
  cross-window entities unverified.
- Flat BIO output cannot represent overlapping or nested entities in a single sequence.
- Calibration can drift under changes in domain, class prevalence, upstream normalization, model
  runtime, or system components.
- The synthetic PII-free false-positive rate is high, while formal/chat recall is materially below
  synthetic-test recall.
- `SECRET` is broader than PII; policy decisions must remain separate from detection.
- Synthetic-only training and the lack of private records reduce direct training-data exposure,
  but they do not prove absence of base-model memorization or eliminate inference-time privacy
  risk.

Operators must establish task-specific acceptance thresholds and rerun end-to-end tests on their
own authorized data before any deployment decision.

## Loading the checkpoint safely

The frozen system evaluation used Transformers 5.13.1. Before publication, the final staged
package must also pass an offline 5.13.1 load, Unicode-boundary probe, and source/package logit
parity check. Its custom model code targets Transformers 5.13 through the 5.x series, but versions
other than 5.13.1 are not quality-attested by this release. Pin a content-addressed revision,
require safetensors, and review
`configuration_qwen3_bi.py` and `modeling_qwen3_bi.py` before enabling remote code.

The serialized tokenizer backend is part of the model's character-boundary contract. Load it
directly with `PreTrainedTokenizerFast`. **Do not use `AutoTokenizer`**: model-specific tokenizer
reconstruction does not preserve the attested per-codepoint boundary graph.

```python
from pathlib import Path

from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast

# Use the independently verified local package. After an authorized Hub
# publication, replace this path with the repository and pin its immutable revision.
model_path = Path("/path/to/zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1")

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    model_path,
    local_files_only=True,
    fix_mistral_regex=False,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    use_safetensors=True,
)
model.eval()
```

The remote code is required for full-attention inference. The attached
[checksums.txt](./checksums.txt) binds all files in the built release package. Dependency inventory
is recorded in [sbom.cdx.json](./sbom.cdx.json).

## Reproducibility, provenance, and security

Use the following release files as the source of truth:

- [training_manifest.json](./training_manifest.json): seed, recipe, runtime, dataset hashes,
  tokenizer contract, and checkpoint binding;
- [data_provenance.json](./data_provenance.json) and
  [teacher_provenance.json](./teacher_provenance.json): source admission and upstream template
  generation;
- [evaluation_report.json](./evaluation_report.json): validation gate, frozen system identity,
  aggregate metrics, uncertainty, and `production_ready=false` scope;
- [calibration.json](./calibration.json) and [thresholds.yaml](./thresholds.yaml): selected-seed
  validation calibration and apply-only test policy;
- [model-index.yml](./model-index.yml): public test result index;
- [checksums.txt](./checksums.txt): content-addressed package file hashes;
- [sbom.cdx.json](./sbom.cdx.json): dependency inventory.

The Apache-2.0 declaration is accompanied by [LICENSE](./LICENSE), [NOTICE](./NOTICE), and
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md). Public upload is withheld until
[SECURITY.md](./SECURITY.md) lists a tested private route. A recipient of the local package may use
an existing trusted channel only to request such a route, without sending report details first.
