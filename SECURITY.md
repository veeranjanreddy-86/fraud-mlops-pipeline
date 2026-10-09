# Security Policy

## Scope

This is a portfolio/reference project that runs entirely on **synthetic data** with a **local**
MLflow store. It contains no credentials, customer data or proprietary code. It is not intended
to be deployed as-is to score real payment traffic.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/veeranjanreddy-86/fraud-mlops-pipeline/security/advisories/new)
rather than opening a public issue. Include steps to reproduce and the affected version/commit.
You can expect an acknowledgement within a few days.

## Hardening notes for anyone adapting this project

- **Model deserialisation:** MLflow 3 stores the sklearn pipeline with `skops`, and only the
  explicitly allow-listed types in `train.SKOPS_TRUSTED_TYPES` are loaded. The exported bundle
  used by the container is a `joblib` pickle; only load bundles you built yourself.
- **Secrets:** none are required. Keep `.env` out of version control (it is gitignored); use a
  secret manager for any remote tracking-server or registry credentials.
- **API surface:** request bodies are strictly validated (`extra="forbid"`, bounded numeric
  ranges, batch size <= 1000). Add authentication, rate limiting and TLS termination at the
  gateway before exposing the service.
- **Container:** the runtime image runs as a non-root user and contains no build tooling.
- **Data:** logs contain request metadata and latency only, never transaction payloads.
