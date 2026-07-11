# Release checklist

This checklist records human review; it cannot override `scripts/release_gate.py`. The repository
is currently **RC_BLOCKED** until trained weights and release-specific evidence exist.

## Immutable inputs

- [ ] Release tag and Git commit are immutable and signed according to project policy.
- [ ] Base model, tokenizer, taxonomy, data manifests, teacher(s), and recipe have exact revisions
  or SHA-256 digests.
- [ ] The source registry records an explicit license/terms conclusion for every sampled source.
- [ ] No `evaluation_only`, quarantined, or private-enterprise source entered training, prompting,
  distillation, calibration, or threshold selection.

## Quality and privacy

- [ ] At least three unique seeds have complete metrics and all published quality criteria pass.
- [ ] Public, private cross-domain, hard-negative, noisy, long-document, and Presidio end-to-end
  evaluations are represented honestly, including confidence intervals and deployment limits.
- [ ] Memorization, canary extraction, duplicate/template leakage, secret, and likely-real-PII scans
  pass; any synthetic canary is explicitly allowlisted by hash and reviewed.
- [ ] A human reviewed the model card, examples, provenance, and remote code.

## Supply chain and package

- [ ] `model.safetensors` is the only weight file; no pickle, optimizer, scheduler, gradient, cache,
  raw data, prompt/response, customer value, or private log is present.
- [ ] `checksums.txt` verifies every file and the CycloneDX SBOM matches the immutable dependency
  lock.
- [ ] A fresh dependency vulnerability scan is complete. Every unresolved blocking finding is
  fixed or has a narrow, approved, unexpired exception with compensating controls.
- [ ] `SECURITY.md` has a tested private reporting route; NOTICE and third-party attributions are
  release-specific and complete.
- [ ] `scripts/release_gate.py` reports `PASS` from a clean checkout. No waiver may convert a failed
  quality/source/privacy gate into green.

