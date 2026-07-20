# Security Policy

## Supported version

`pii-zh-qwen 0.2.0rc1` is a community research release candidate. GitHub Private Vulnerability
Reporting is enabled for this public source repository, but an independent synthetic end-to-end
check has not yet been recorded. Reports concerning software security, model behavior, data
provenance, privacy, or supply-chain integrity are in scope for this version. This release
candidate is not presented as a production security warranty or service-level commitment.

## Private reporting

Do not disclose a suspected vulnerability, leaked personal datum, model-memorization case,
unsafe remote-code behavior, or supply-chain issue in a public issue, discussion, or post.

[GitHub Private Vulnerability Reporting](https://github.com/whyiug/pii-detect-model/security/advisories/new)
is enabled. Its independent synthetic end-to-end check and receipt are still pending, so the
Hugging Face model upload and GitHub Release remain blocked. If the authenticated private-report
form is unavailable, use an existing trusted channel only to request a secure reporting route; do
not include vulnerability details in that initial request.

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
frozen evaluation set. Until its independent test is recorded, an existing trusted channel may be
used only to report that the form is unavailable; that initial message must contain no report
details.
