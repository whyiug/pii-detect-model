# Security Policy

## Supported versions

Security fixes are provided for the latest public release of `pii-zh-qwen`.

| Version | Supported |
|---|---|
| `0.2.0rc1` | Yes |
| Earlier versions | No |

## Report a security issue

Please report suspected vulnerabilities through
[GitHub Private Vulnerability Reporting](https://github.com/whyiug/pii-detect-model/security/advisories/new).
Do not disclose the issue in a public issue, discussion, pull request, or post before coordinated
disclosure is complete.

A useful report includes:

- the affected version, commit, package, or model revision;
- a concise description of the impact and affected security boundary;
- a minimal local reproduction using synthetic data;
- redacted logs and relevant code locations; and
- a suggested fix or regression test, when available.

Do not upload real personal information, customer data, passwords, API tokens, cookies, private
keys, credentials, or other secrets. Do not test against public or third-party deployments. Use a
local environment and synthetic inputs, and remove sensitive values from logs and attachments.

## Scope

Reports may cover the source code, Python package, CLI, HTTP service, Presidio integration, model
loading and remote-code boundary, dependency or build integrity, unintended data disclosure, model
memorization, and provenance issues with a security or privacy impact.

Accuracy or documentation issues that can be demonstrated entirely with synthetic, non-sensitive
examples may be filed as regular GitHub issues. Use the private reporting channel whenever the
evidence itself could expose sensitive information or enable abuse.

## What to expect

We aim to acknowledge complete reports promptly. Confirmed issues are handled through coordinated
remediation and disclosure, and reporters are kept informed when scope, severity, remediation, or
the disclosure timeline changes.
