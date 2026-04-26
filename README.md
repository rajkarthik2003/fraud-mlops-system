# Fraud Detection MLOps System

Production-oriented fraud detection project built around model serving, explainability, monitoring, and deployment workflows.

## Project Summary

This repository packages a fraud detection workflow as a system rather than only a model notebook. The strongest public signals in the repo are:

- FastAPI inference service
- SHAP-based explanation endpoint
- Redis-backed caching
- PostgreSQL logging
- Prometheus-style monitoring
- Docker and Kubernetes deployment assets
- test coverage for API and inference logic

## Problem Context

The project is built around an extreme class-imbalance problem where fraud cases represent roughly `0.17%` of the dataset. Because of that, the design focuses on threshold tuning, precision-recall tradeoffs, and operational monitoring instead of accuracy alone.

## Main Features

- fraud prediction endpoint
- batch prediction support
- explanation support for model outputs
- drift-check endpoint
- health and readiness probes
- async prediction workflow
- CI-oriented test structure

## Repo Structure

```text
app/
  api/          FastAPI service
  ui/           dashboard layer
training/       training and export scripts
tests/          API and inference tests
k8s/            Kubernetes manifests
monitoring/     metrics and drift helpers
Dockerfile
docker-compose.yml
```

## Why This Repo Matters

This is one of the stronger public repos in the account because it demonstrates applied ML engineering across:

- model packaging
- operational APIs
- monitoring
- deployment setup
- testing

It supports the MLOps and production-ML claims in your resume well.

## Notes

- The project is best read as a portfolio-grade systems repo, not just a benchmark report.
- A cleanup was applied so the repository behaves more cleanly on Windows-based Git workflows.

## Related Projects

For adjacent work in LLM systems and backend delivery, see:

- [grounded-llm-system](https://github.com/rajkarthik2003/grounded-llm-system)
- [EduvisionMVC](https://github.com/rajkarthik2003/EduvisionMVC)
- [Next-Word-Prediction-Model-Using-Deep-Learning](https://github.com/rajkarthik2003/Next-Word-Prediction-Model-Using-Deep-Learning)
