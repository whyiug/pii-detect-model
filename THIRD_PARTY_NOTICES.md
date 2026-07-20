# Third-Party Notices — pii-zh-qwen 0.2.0rc1

This document records third-party models, datasets and tools used to build or evaluate this
version. Each upstream license and attribution applies independently.

## AIguard PII detection model

- Source: <code>ZJUICSR/AIguard-pii-detection-fast</code>
- Revision: <code>677a5ebc1600fef61e8973cafd3026be322b3a73</code>
- Upstream-declared license: Apache-2.0
- Role: initialization checkpoint for the released 24-class token classifier

The training manifest binds the upstream <code>config.json</code>,
<code>model.safetensors</code>, <code>tokenizer.json</code> and
<code>tokenizer_config.json</code> by SHA-256. It records a strict backbone copy, an outside-label
row copy and 12 source-to-target classification-head projections. The upstream checkpoint is not
redistributed as a separate artifact.

## Qwen research lineage

### Qwen3-0.6B-Base

- Source: <code>Qwen/Qwen3-0.6B-Base</code>
- Revision: <code>da87bfb608c14b7cf20ba1ce41287e8de496c0cd</code>
- License: Apache-2.0
- Attribution: Qwen; Copyright 2024 Alibaba Cloud
- Role: base checkpoint for an archived research checkpoint, not the current released weights

### Qwen3-8B

- Source: <code>Qwen/Qwen3-8B</code>
- License: Apache-2.0
- Attribution: Qwen; Copyright 2024 Alibaba Cloud
- Role: local generator for placeholder-only synthetic template candidates

Reviewers accepted 53 of 70 generated template candidates. Qwen3-8B was not used as a span-label
or pseudo-label teacher. Its checkpoint, raw outputs and rejected outputs are not redistributed.

## Repository synthetic templates

- Source: <code>src/pii_zh/data/synthetic/assets/curated_templates_v1.json</code>
- SHA-256: <code>d65c7b50a21c48ce217a4d44ea6f7333bdb0c385847f8c1d4d067507f4b2563f</code>
- License: Apache-2.0
- Attribution: Copyright 2026 pii-zh-qwen contributors

The templates contain abstract field markers rather than customer records. Training examples were
created by the repository's deterministic synthetic-data pipeline.

## pii-bench-zh

- Source: <code>wan9yu/pii-bench-zh</code>
- Revision: <code>c350b94897af668517ff5de237d89f2ce2eaa6f0</code>
- License: Apache-2.0
- Role: evaluation-only formal and chat suites

This dataset was excluded from training, calibration, threshold selection and template
development. Dataset rows are not redistributed; only aggregate results are reported.

## Runtime and development dependencies

Runtime and development dependencies are not vendored into the model repository. Resolved package
versions are recorded in the
[source lock](https://github.com/whyiug/pii-detect-model/blob/v0.2.0rc1/uv.lock), and the release
page provides downloadable checksums and supply-chain artifacts:
[GitHub v0.2.0rc1](https://github.com/whyiug/pii-detect-model/releases/tag/v0.2.0rc1).

## Data boundary

No customer data, internal production data or external API output was used to train, calibrate or
evaluate this version. Training and development data are deterministic synthetic or open-source
derived data as described in the Model Card.
