# Release checklist

This checklist records the current community research RC state and the remaining human and
publication work. It cannot override `scripts/release_gate.py`. Training and blind public-suite
evaluation are complete; production approval, final package verification, upload, and tag signing
are not complete.

## Immutable inputs and isolation

- [ ] The intended release tag and Git commit have been reviewed, made immutable, and signed under
  project policy.
- [x] The base model, tokenizer, 24-label taxonomy, train/validation/test manifests, teacher/source
  lineage, recipe, three seeds, selected seed, and output artifact have exact revisions or SHA-256
  digests in the attested manifests.
- [x] Every sampled training source in `configs/data/source_registry.yaml` has an immutable
  revision, explicit license conclusion, completed review, and permission for public-weight
  training.
- [x] Manifest and provenance checks show that evaluation-only, quarantined, and
  private-enterprise sources did not enter training, prompting, distillation, calibration fitting,
  or threshold selection.

## Quality and privacy evidence

- [x] Seeds 13, 42, and 97 have complete synthetic-validation metrics and each passed all
  preregistered strict micro/macro F1, PII-free FPR, Tier-0/Tier-1 recall, and calibration criteria.
- [x] The unchanged seed-42 system was reported on all frozen public suites after the validation
  unlock: synthetic test strict micro F1 0.9072, PII Bench formal 0.8265, and PII Bench chat 0.7406,
  each with a 95% document-bootstrap interval.
- [x] The research-RC evidence reports the synthetic PII-free document FPR of 23.83% (143/600) and
  explicitly marks the artifact non-production.
- [ ] Run and review private cross-domain, noisy-private, long-document, production Presidio
  end-to-end, shadow, customer, and canary evaluations. These are currently absent, so production
  approval remains withheld.
- [ ] Memorization, canary extraction, duplicate/template leakage, secret, and likely-real-PII
  checks have been completed together and reviewed for the final package.
- [ ] A human has reviewed the final model card, examples, provenance, evaluation limitations,
  remote code, and the non-production claim.

## Supply chain and package

- [ ] The final staged package has been inspected to confirm `model.safetensors` is its only weight
  file and contains no pickle, optimizer, scheduler, gradient, cache, raw data, prompt/response,
  customer value, or private log.
- [ ] `checksums.txt` has been regenerated and independently verified for every final package file.
- [ ] The final CycloneDX SBOM has been regenerated from `uv.lock` and its digest matches both the
  packaged evidence and normalized dependency scan.
- [ ] A fresh `pip-audit` run has completed against the release dependency environment; every
  unresolved blocking finding is fixed or has a narrow, approved, unexpired exception with
  compensating controls.
- [ ] `scripts/scan_public_artifacts.py` reports no unreviewed secret or likely-real-PII finding for
  the final package; every synthetic canary exception is fingerprint-bound and human-reviewed.
- [ ] `SECURITY.md` has a tested private reporting route, and NOTICE plus third-party attribution
  have been reviewed as release-specific and complete.
- [ ] Offline Transformers 5.13.1 loading uses the serialized fast-tokenizer graph, passes the
  Unicode boundary probe, and preserves source-checkpoint/release-package logit parity.
- [ ] `scripts/release_gate.py` exits zero with `status: PASS` for the final artifact from a clean
  checkout. No waiver may turn a failed quality, source, privacy, or dependency gate green.

## Human production and publication decisions

- [ ] Authorized privacy, security, model-risk, legal, and product owners have reviewed the measured
  23.83% synthetic PII-free FPR and explicitly approved the intended use. This is required for any
  production claim and is currently not granted.
- [ ] Representative private-domain and full production Presidio/policy-path evidence meets
  deployment-specific guardrails, including long-document and failure-mode tests.
- [ ] Rollback, monitoring, drift detection, false-positive review, and incident-response ownership
  are approved for the intended deployment.
- [ ] The exact final package has been published through an authorized workflow at an immutable Hub
  revision; no upload is currently claimed.
- [ ] The corresponding release tag has been signed and independently verified; no signed tag is
  currently claimed.
