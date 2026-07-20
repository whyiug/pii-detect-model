# Third-Party Notices — pii-zh-qwen 0.2.0rc1 community candidate

This notice records the sources admitted to, evaluated with, or otherwise relevant to the current
community candidate and its repository history. Maintainer `whyiug` reviewed this inventory and
approved public GitHub and Hugging Face distribution on 2026-07-20 with no exceptions. A
self-hashed external receipt binds that decision to the exact `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md` files used by the publication builder.

## AIguard PII detection model

- Source: `ZJUICSR/AIguard-pii-detection-fast`
- Revision: `677a5ebc1600fef61e8973cafd3026be322b3a73`
- Upstream-declared license: Apache-2.0
- Release role: fixed initialization checkpoint for the current community token classifier

The current training manifest binds the upstream `config.json`, `model.safetensors`,
`tokenizer.json`, and `tokenizer_config.json` by SHA-256. It records a strict backbone copy, an
outside-label row copy, and 12 verified source-to-target classification-head projections. The
resulting candidate changes attention from causal to full bidirectional attention and trains the
24-entity, 49-token-label taxonomy. The upstream checkpoint is not redistributed as a separate
artifact; the Hugging Face candidate contains the resulting merged weights.

## Historical Qwen3-0.6B-Base research candidate

- Source: `Qwen/Qwen3-0.6B-Base`
- Revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- License: Apache License 2.0
- Attribution: Qwen; Copyright 2024 Alibaba Cloud
- Release role: base checkpoint for the fine-tuned token-classification model

The archived `synthetic-v1.3-rc1` checkpoint was derived from this base. It is not the current
AIguard24 seed-97 candidate. The upstream Qwen license and attribution remain applicable to the
historical Qwen-derived materials.

## Qwen3-8B template candidate generator

- Source: `Qwen/Qwen3-8B`
- Local model-index SHA-256 fingerprint:
  `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`
- License: Apache License 2.0
- Attribution: Qwen; Copyright 2024 Alibaba Cloud
- Release role: local, placeholder-only synthetic-template candidate generator

This model was not a span-label or pseudo-label teacher. Reviewers examined 70 generated
template candidates and accepted 53. Only the 53 reviewed, accepted template skeletons were
admitted to the repository template asset. The Qwen3-8B checkpoint, raw model outputs, rejected
outputs, and model-produced span labels are not included in the training data or redistributed
with this release candidate.

## Repository synthetic templates

- Source: `src/pii_zh/data/synthetic/assets/curated_templates_v1.json`
- Revision:
  `sha256:d65c7b50a21c48ce217a4d44ea6f7333bdb0c385847f8c1d4d067507f4b2563f`
- License: Apache License 2.0
- Attribution: Copyright 2026 pii-zh-qwen contributors
- Release role: reviewed template skeletons used by the deterministic synthetic-data generator

These templates contain abstract field markers rather than customer records. Training examples
were materialized by the repository's deterministic synthetic-data pipeline.

## pii-bench-zh

- Source: `wan9yu/pii-bench-zh`
- Revision: `c350b94897af668517ff5de237d89f2ce2eaa6f0`
- License: Apache License 2.0
- Release role: frozen, evaluation-only formal and chat suites

This dataset was excluded from training, calibration, threshold selection, and template
development. Dataset rows are not redistributed with this release candidate; only aggregate
evaluation evidence is reported.

## Runtime and development dependencies

The exact packaged dependency inventory is recorded in `sbom.cdx.json`; the reproducible source
environment resolution is recorded in `uv.lock`. Those records are authoritative for package
names and resolved versions. Runtime and development dependencies are not vendored into this
repository or the model checkpoint.

## Excluded inputs and review boundary

No external API output and no customer or internal production data were used to train, calibrate,
or evaluate this release candidate. The immutable typed license report retains its pre-approval
status `COMPLETE_HUMAN_APPROVAL_PENDING` as historical mechanical evidence. Human clearance comes
from the separately validated approval receipt, not from that automated status or this notice
alone.
