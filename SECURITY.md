# Security Policy

## Supported Versions

Security fixes are applied to the current version of the repository.

| Version | Supported |
| ------- | :-------: |
| Current repository version | ✅ |
| Older releases | ❌ |

Older releases are preserved for archival and reproducibility purposes but may
not receive security updates.

## Reporting a Security Vulnerability

If you discover a security vulnerability, please **do not open a public GitHub
issue containing sensitive details**.

Instead, contact the project maintainer privately using the contact information
available through the repository owner's GitHub profile.

Please include, when possible:

- A clear description of the vulnerability.
- Steps required to reproduce the issue.
- The affected component or file.
- Potential security impact.
- Relevant environment or dependency information.
- A suggested mitigation, if available.

Please avoid including credentials, API keys, private datasets, personal data,
or other unnecessary sensitive information in the report.

## Response Process

Security reports will be reviewed as reasonably possible.

The maintainer may:

1. Confirm and reproduce the reported issue.
2. Assess its severity and potential impact.
3. Develop and test an appropriate fix.
4. Update affected code, configuration, dependencies, or workflows.
5. Publish a patched version or security-related update when appropriate.
6. Credit the reporter when appropriate, unless anonymity is requested.

No specific response or resolution time is guaranteed.

## Scope

This security policy applies to vulnerabilities involving repository components
such as:

- Source code.
- Dependency usage.
- Configuration files.
- GitHub Actions workflows.
- Handling of environment variables, credentials, or secrets.
- Unsafe file or input handling introduced by project code.

## Research and Correctness Issues

Scientific, methodological, or reproducibility problems are important, but they
are normally not security vulnerabilities.

Examples include:

- incorrect evaluation metrics,
- data leakage between calibration and held-out evaluation,
- inconsistent experiment configuration,
- incorrect statistical analysis,
- documentation errors,
- non-reproducible experimental results.

Unless such an issue also creates a security risk or exposes sensitive
information, it may be reported through the repository's normal public issue
process.

When reporting a research-correctness issue publicly, do not include private
data, credentials, or other sensitive information.

## Third-Party Components

This project depends on third-party Python packages, pretrained models, and
datasets.

A vulnerability originating entirely in a third-party component may need to be
reported to that component's maintainers. However, project-specific insecure
usage or configuration of a dependency is within the scope of this policy.

Thank you for helping keep the project secure.