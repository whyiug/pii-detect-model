# Private security channel test waiver for `0.2.0rc1`

## Recorded decision

- Release: `pii-zh-qwen 0.2.0rc1`
- GitHub repository: `whyiug/pii-detect-model`
- Hugging Face repository: `Forrest20231206/pii-zh-qwen3-0.6b-24class`
- Authorizing maintainer: `whyiug`
- Authorization date: 2026-07-20
- Authorization record time: `2026-07-20T08:29:44Z`
- Waiver ID: `pii-zh-qwen-0.2.0rc1-private-channel-test-waiver-20260720`

Maintainer `whyiug` explicitly waives, for this release candidate only, the publication gate that
would otherwise require an independent account without repository administration privileges to
submit a synthetic GitHub Private Vulnerability Report and require the project to record its
triage-to-closure lifecycle in a tested-channel receipt.

## Machine-readable evidence

The receipt generator binds the following exact statements to this file and `SECURITY.md`:

```text
Waiver-ID: pii-zh-qwen-0.2.0rc1-private-channel-test-waiver-20260720
Authorized-by: whyiug
Authorized-at: 2026-07-20T08:29:44Z
Package-version: 0.2.0rc1
GitHub-repository: whyiug/pii-detect-model
Hugging-Face-repository: Forrest20231206/pii-zh-qwen3-0.6b-24class
Provider: github_private_vulnerability_reporting
Enabled: true
Independent-test-completed: false
Decision: maintainer_waived_for_release_candidate
Evidence-basis: explicit_human_maintainer_attestation_and_bound_waiver_file
```

## Facts retained by the waiver

- GitHub Private Vulnerability Reporting is enabled and must remain enabled.
- The independent synthetic end-to-end test remains outstanding.
- This waiver does not state or imply that an independent report was received, accepted, triaged,
  drafted, or closed successfully.
- A draft created by a repository owner, if any, is not independent test evidence. This public
  record contains no private advisory body, report payload, or private GHSA identifier.

## Scope and limits

The waiver covers only the missing independent private-report-channel test and its tested-channel
receipt for `0.2.0rc1`. It allows the release process to use a separately generated, self-hashed
`private-security-channel-waiver-receipt.json` in place of that tested-channel receipt.

It does **not**:

- disable or weaken the private reporting route or the guidance in `SECURITY.md`;
- provide a production security, privacy, availability, or service-level guarantee;
- waive source, hosted CI, signed-tag, license, artifact, checksum, SBOM, model-package, private
  Hugging Face re-download, or remote-evidence gates;
- authorize publication by itself or replace the maintainer's separate final confirmation before
  changing Hugging Face visibility and publishing the GitHub Release; or
- apply to any later release, tag, repository, or model identity.

This committed source policy locks `0.2.0rc1` to the waiver path. Changing to tested-channel
evidence requires a new source commit that removes this policy and repeats CI, signing,
and artifact construction; evidence types must never be swapped beneath the same source commit or
tag. This run must retain the alternative test path as unperformed.
