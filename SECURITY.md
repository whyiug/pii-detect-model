# Security Policy

## Supported versions

The project is pre-alpha and has no supported production release yet. Once releases exist, this
section will list the maintained versions and security-update policy.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability, leaked personal datum, model memorization case,
unsafe remote-code behavior, or supply-chain issue in a public issue or discussion.

Use the repository host's private vulnerability-reporting feature when it is enabled. If it is not
enabled, contact the repository owner through a private channel shown on the owner's profile and
request a secure reporting route without including exploit or personal-data details in that first
message. A dedicated security address has not yet been configured; configuring and testing one is
a release blocker.

Include, where safe:

- affected revision, package, model, or dataset manifest hash;
- impact and the security/privacy boundary crossed;
- a minimal local reproduction using synthetic data;
- relevant logs with credentials, tokens, and PII removed;
- suggested remediation or regression assertion.

Do not test against public or third-party deployments. Do not include real secrets, customer data,
or weaponized payloads. Maintainers should acknowledge a complete private report, coordinate a
fix and disclosure timeline, and credit the reporter if requested.

## Model and data safety reports

False negatives, false positives, demographic/domain regressions, memorization, provenance errors,
and evaluation leakage are treated as safety-quality reports even when they are not software
vulnerabilities. Submit them privately whenever samples might contain PII or reveal a frozen test
set.

