# Model cards

## Current community candidate

- [`pii-zh-qwen3-0.6b-24class` 0.2.0rc1](PII_ZH_QWEN3_0_6B_24CLASS_RC1.md)
- [Publication-successor card](PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md)
- Structured model-only results: [`model-index.yml`](model-index.yml)
- Publication-successor state: `staged_not_uploaded`
- Production ready: `false`

The RC1 card is byte-identical to the README in the immutable local `model-package-v2r2`.
It documents the 24-label model, local usage, synthetic Open-24 comparison, public-test-exposed
PII Bench ZH post-hoc result, training provenance and limitations. The full cascade result remains
a system-level claim and must not be attributed to the weights alone.

Actual Hugging Face publication uses the separate publication-successor card and requires a
successor package because the immutable local package still contains preauthorization language
and the historical security policy. See
[`release/community-v2-rc1/PUBLISHING.md`](../release/community-v2-rc1/PUBLISHING.md).

## Historical archive

The previous `zh-pii-qwen3-0.6b-bi-synthetic-v1.3-rc1` card is historical, synthetic-only and not
the current flagship. Its exact tracked version remains available in Git history before the
community-v2 release-preparation commit (for example, `git show 15986a8:model_cards/README.md`).
Its package-relative links are intentionally not copied into this index because those archived
package files are not part of the current GitHub source release.

No model card in this directory authorizes a first/best/SOTA, real-world, production or compliance
claim.
