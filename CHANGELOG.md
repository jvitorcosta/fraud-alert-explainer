# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com) and this
project uses [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Project tooling from `python-uv-claude-template` v0.3.0: Claude Code agents, commands, hooks
  and rules, Dependabot, issue and PR templates, contributing guide.

## [0.1.0] - 2026-10-29

### Added

- LightGBM and logistic-regression fraud models on the Sparkov dataset, with a time split and
  past-only card-history features.
- Alerts-per-day vs fraud-caught analysis, with thresholds set on validation and checked on test.
- SHAP top reasons per alert, explained by a local Ollama model, with a deterministic check
  that each explanation cites only the ranked features and no invented numbers.
