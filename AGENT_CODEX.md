# AGENT_CODEX.md
## InfraWatch™ — Codex Agent Rules (Engineering Contract)

This document defines strict rules for Codex (and any AI agents) working on the InfraWatch™ repository.

InfraWatch™ is a decision-support Earth Observation (EO) analytics application focused on:
vegetation risk monitoring near traction and power networks using satellite imagery (Copernicus Sentinel).

The agent must optimize for reliability, auditability, and business value — not novelty.

---

## 1. Mission & Product Boundary

### Mission
Build an EO analytics system that:
- ingests Sentinel satellite data,
- computes vegetation and change metrics,
- assigns risk per network segment,
- generates alerts, dashboards, and audit-ready PDF reports.

### Hard boundaries (DO NOT implement)
- No rocket, propulsion, or launch logic
- No surveillance of individuals
- No personal data processing
- No black-box ML in MVP phases
- No autonomous actions (emails, purchases, control systems)

---

## 2. Decision-First Definition of Done

A feature is considered DONE only if:
1. It enables a clear operational decision (e.g. “trim vegetation within 14 days”)
2. Outputs are explainable in plain language
3. Results are reproducible (same input → same output)
4. Minimal tests exist or lack of tests is explicitly justified
5. Documentation is updated (docstring / README / docs)

If any point is missing → the feature is NOT done.

---

## 3. Mandatory Development Workflow

For every task, Codex must:

1. Restate the objective in business terms
2. Propose the minimal technical solution (MVP first)
3. Implement in small, isolated steps
4. Add or update tests where feasible
5. Update documentation
6. Summarize what was done, why, and how to validate it

No large, monolithic commits.
No refactors unrelated to the task.

---

## 4. Repository Structure Rules

Codex must strictly respect the following structure:

src/infrawatch/
- ingestion/        EO data discovery, download, preprocessing, caching
- analytics/        vegetation indices, time-series, change detection
- scoring/          VEI and risk score logic
- reporting/        PDF / CSV generation
- ui/               Streamlit dashboards (presentation only)
- api/              optional CLI / REST interfaces
- utils/            shared helpers (paths, logging, validation)

tests/
scripts/
docs/

### Separation of concerns (non-negotiable)
- No business logic in ui/
- scoring/ must not depend on Streamlit
- ingestion/ must not contain scoring logic
- Shared logic goes to utils/
- No circular imports

---

## 5. Coding Standards

- Language: Python 3.11+
- Style: PEP8
- Prefer pure functions over classes unless state is required
- Use type hints where practical
- Avoid side effects
- Keep functions small and readable

### Naming
- snake_case for variables and functions
- PascalCase for classes
- Abbreviations only if domain-standard (NDVI, SAR, VEI)

### Error handling
- Fail explicitly
- Raise meaningful exceptions
- Never silently ignore missing data
- Missing data must be treated as a data-quality event

---

## 6. Determinism & Auditability

InfraWatch™ must be auditable by design.

- Every metric must be documented
- Every threshold must be:
  - a named constant
  - documented
  - configurable (config / env / file)

No magic numbers.

BAD:
if vei > 0.7: ...

GOOD:
VEI_WARNING_THRESHOLD = 0.70
if vei >= VEI_WARNING_THRESHOLD: ...

Ensure deterministic behavior:
- consistent time-series ordering
- documented resampling rules
- processing metadata stored (date, version, source)

---

## 7. Data & Licensing Rules

### Allowed data (MVP)
- Copernicus Sentinel-1 and Sentinel-2
- Public vector layers with compatible licenses
- Client-provided infrastructure geometry

### Not allowed
- Scraping restricted sources
- Using private datasets without permission
- Any personal data

All new data sources must have license and origin documented.

---

## 8. Security Rules

- Never commit secrets or credentials
- Use .env and .env.example
- External APIs must be optional and documented
- No hardcoded tokens or keys

---

## 9. Quality Gates (Before Commit / Merge)

Before finalizing changes, Codex must ensure:
- Tests pass (pytest) or justification is documented
- No obvious lint issues
- Documentation updated
- Interfaces are stable or changes are documented

Breaking changes require documentation updates.

---

## 10. Performance Rules (Practical)

EO data can be large:
- Avoid loading full rasters when unnecessary
- Prefer windowed or tiled reads
- Cache intermediate results
- Correctness > optimization in MVP

---

## 11. Output Requirements

### Alerts must include
- segment identifier
- risk score
- recommended action window
- data availability / confidence flag

### Dashboard must show
- segment-level risk map
- time trends
- filters by time and risk threshold

### PDF Report must include
- executive summary
- data sources
- methodology (plain language)
- results (tables / plots)
- recommendations
- limitations and assumptions
- versioning (app version, date)

---

## 12. Git Discipline

- Small, atomic commits only
- One feature or fix per commit
- Commit messages must express business intent

Examples:
- feat: compute NDVI time-series per traction segment
- feat: add VEI-based vegetation risk scoring
- fix: handle missing Sentinel dates deterministically
- docs: add operator-facing report description

No “misc” or “cleanup” commits.

---

## 13. Agent Self-Check (Mandatory)

Before submitting code, Codex must answer:
1. What operational decision does this enable?
2. Is the result explainable to a non-technical user?
3. Can the output be reproduced?
4. Are thresholds documented and configurable?
5. Are tests present or justified?
6. Is this the minimal necessary solution?

If any answer is “no” → revise.

---

## 14. Implementation Priority Order

Codex should implement features strictly in this order:
1. Repository skeleton and configuration
2. Sentinel-2 ingestion (NDVI-ready)
3. Vegetation analytics (NDVI, growth rate)
4. Risk scoring (VEI, thresholds)
5. PDF reporting
6. Streamlit dashboard
7. Sentinel-1 / SAR extensions (later)

---

## 15. Final Rule

If a feature does not improve a real operational decision,
it does not belong in InfraWatch™.

InfraWatch™ is an industrial, decision-support system.
Keep it minimal, explainable, and auditable.
