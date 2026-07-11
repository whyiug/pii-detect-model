# Security Policy

## Supported version

`synthetic-v1.3-rc1` is a research release candidate. Once a tested private route is established,
reports concerning software security, model behavior, data provenance, privacy, or supply-chain
integrity will be in scope for this version. This release candidate is not presented as a
production security warranty or service-level commitment.

## Private reporting

Do not disclose a suspected vulnerability, leaked personal datum, model-memorization case,
unsafe remote-code behavior, or supply-chain issue in a public issue, discussion, or post.

No tested private reporting route exists for this local RC package yet. Public upload is therefore
withheld. Before publication, an authorized maintainer must create the intended repository, enable
and test GitHub private vulnerability reporting, and update this file with the verified route. If
you received this package through an existing trusted private channel, use that same channel only
to request a secure reporting route; do not include vulnerability details in the initial request.

Identify `synthetic-v1.3-rc1` and give only a short impact category. Do not attach sensitive data,
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
vulnerabilities. Once the verified private route exists, use it whenever evidence could contain
PII or disclose a frozen evaluation set. Until then, an existing trusted channel may be used only
to request a route; the initial request must contain no report details.
