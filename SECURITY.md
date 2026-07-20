# Security Policy

## Supported version

`pii-zh-qwen 0.2.0rc1` is a community research release candidate. GitHub Private Vulnerability
Reporting is enabled for this public source repository, but the independent synthetic end-to-end
test remains outstanding. On 2026-07-20, maintainer `whyiug` explicitly waived only that
independent-test release gate for `0.2.0rc1`; the waiver is recorded in
[`release/community-v2-rc1/PRIVATE_SECURITY_CHANNEL_WAIVER.md`](release/community-v2-rc1/PRIVATE_SECURITY_CHANNEL_WAIVER.md).
It retains no independent tested-channel evidence. Reports concerning software
security, model behavior, data provenance, privacy, or supply-chain integrity are in scope for this
version. This release candidate is not presented as a production security warranty or service-level
commitment.

## Private reporting

Do not disclose a suspected vulnerability, leaked personal datum, model-memorization case,
unsafe remote-code behavior, or supply-chain issue in a public issue, discussion, or post.

[GitHub Private Vulnerability Reporting](https://github.com/whyiug/pii-detect-model/security/advisories/new)
remains enabled. Its independent synthetic end-to-end check remains outstanding. The scoped
`0.2.0rc1` waiver permits release staging to continue without a tested-channel receipt, but does not
disable Private Vulnerability Reporting, validate production readiness, attest that any report was
independently received or triaged, or provide the separate final authorization required to make the
Hugging Face repository and GitHub Release public. If the authenticated private-report form is
unavailable, use an existing trusted channel only to request a secure reporting route; do not include
vulnerability details in that initial request.

Identify `pii-zh-qwen 0.2.0rc1` and give only a short impact category. Do not attach sensitive data,
real PII, secrets, frozen evaluation rows, exploit details, or reproduction archives to the first
message.

After a private route is agreed, include only the material needed to assess the report:

- the affected revision, package, model, or dataset-manifest hash;
- the impact and the security or privacy boundary crossed;
- a minimal local reproduction using synthetic data;
- relevant logs with credentials, tokens, and PII removed;
- a suggested remediation or regression assertion, if available.

Do not test against public or third-party deployments. Do not include real secrets, customer data,
or weaponized payloads. Coordinated reports will be assessed privately, with remediation and
disclosure timing discussed with the reporter.

## Model and data safety reports

False negatives, false positives, domain regressions, memorization, provenance errors, and
evaluation leakage are treated as safety-quality reports even when they are not software
vulnerabilities. Use the private-report route whenever evidence could contain PII or disclose a
frozen evaluation set. Because the independent test remains outstanding, an existing trusted
channel may be used only to report that the form is unavailable; that initial message must contain
no report details. The release-gate waiver does not alter this reporting guidance.
