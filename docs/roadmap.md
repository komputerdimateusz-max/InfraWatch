\# ROADMAP.md

\## InfraWatch™ — Vegetation Risk Monitoring for Traction \& Power Networks



InfraWatch™ is an EO (Earth Observation) analytics product that converts satellite data into

\*\*actionable vegetation risk alerts\*\* for linear infrastructure (traction \& power networks).



This roadmap prioritizes: \*\*decision-first outputs, explainability, auditability, and MVP speed\*\*.



---



\## North Star (Product Vision)



Deliver a system that:

\- monitors vegetation growth along infrastructure corridors,

\- detects changes and anomalies,

\- assigns \*\*risk scores per segment\*\*,

\- generates \*\*alerts + audit-ready reports\*\*,

\- integrates with operator workflows (GIS / CMMS) in later phases.



---



\## Milestones Overview



\### M0 — Repo \& Foundations (Done / In Progress)

\*\*Outcome:\*\* repository ready for controlled development (Codex-ready).

\- \[x] GitHub repo connected

\- \[x] README.md (product description)

\- \[x] AGENT\_CODEX.md (development contract)

\- \[x] Project skeleton (src/, tests/, scripts/)

\- \[ ] CI pipeline (optional, later)



---



\## Phase 1 — MVP (Decision Outputs from Open EO Data)



\### M1 — Ingestion (Sentinel-2 first)

\*\*Outcome:\*\* deterministic ingestion + caching + metadata for Sentinel-2.

\- \[ ] Search Sentinel-2 scenes for bbox + date range

\- \[ ] Dry-run listing

\- \[ ] Download with cache (skip existing)

\- \[ ] Store metadata JSON per scene

\- \[ ] Minimal tests for validation/sorting/path logic

\- \[ ] Update `.env.example` (optional credentials)



\*\*DoD:\*\* reproducible scene list and cache folder structure.



---



\### M2 — Vegetation Analytics (NDVI Time-Series)

\*\*Outcome:\*\* compute NDVI and derive vegetation intensity per pixel / corridor.

\- \[ ] Read required bands (e.g. Red, NIR)

\- \[ ] Compute NDVI raster

\- \[ ] Normalize / mask invalid pixels

\- \[ ] Save derived NDVI products under `satellite\_data/processed/`

\- \[ ] Tests for NDVI computation (pure functions)



\*\*DoD:\*\* NDVI computed for at least 1 downloaded scene and stored deterministically.



---



\### M3 — Corridor \& Segmentation

\*\*Outcome:\*\* model network geometry and compute metrics per segment.

\- \[ ] Accept network geometry input (client file or demo GeoJSON)

\- \[ ] Create corridor buffer around line (configurable width)

\- \[ ] Segment lines (fixed length, e.g., 100–500m)

\- \[ ] Aggregate NDVI per segment (mean, p90, trend)

\- \[ ] Data quality flags (coverage, cloud/mask ratio)



\*\*DoD:\*\* per-segment vegetation metrics table generated for a date.



---



\### M4 — Risk Scoring v0 (Explainable Rules)

\*\*Outcome:\*\* deterministic risk scoring and priorities (no ML).

\- \[ ] Define VEI (Vegetation Encroachment Index) from metrics

\- \[ ] Risk Score 0–100 (rule-based)

\- \[ ] Threshold configuration (warning/critical)

\- \[ ] Forecast horizon (7/14/30) using trend extrapolation

\- \[ ] Minimal tests for scoring rules



\*\*DoD:\*\* segment-level risk scores with clear thresholds and reasons.



---



\### M5 — Reporting v0 (Audit-ready PDF)

\*\*Outcome:\*\* generate a report usable by operators and auditors.

\- \[ ] PDF report template (exec summary → method → results → recommendations)

\- \[ ] Tables: top-risk segments, trends

\- \[ ] Basic plots: risk trend, VEI trend

\- \[ ] Include assumptions + limitations + version stamp



\*\*DoD:\*\* one PDF report generated from local data with deterministic formatting.



---



\### M6 — UI v0 (Streamlit Dashboard)

\*\*Outcome:\*\* thin UI layer for exploration and exports.

\- \[ ] Choose region/date range

\- \[ ] Show segment risk table with filters

\- \[ ] Show trend plots

\- \[ ] “Generate report” button

\- \[ ] Export CSV



\*\*DoD:\*\* operator can identify top-risk segments and export actions.



---



\## Phase 2 — Pilot \& Commercial Readiness



\### P1 — Pilot Execution Pack

\*\*Outcome:\*\* package product for first operator pilot (Lubuskie target possible).

\- \[ ] Pilot configuration: corridor width, segmentation, thresholds

\- \[ ] Standard operating procedure: alerts → field verification

\- \[ ] Feedback loop \& calibration checklist

\- \[ ] Case study format (ROI + avoided costs)



---



\### P2 — Coverage \& Robustness

\*\*Outcome:\*\* reliability improvements and operational hardening.

\- \[ ] Sentinel-1 integration for cloud-independent change detection

\- \[ ] Better masking / cloud handling (Sentinel-2 quality layers)

\- \[ ] Data completeness scoring per segment

\- \[ ] Backfill historical time-series



---



\### P3 — Alerts \& Integrations

\*\*Outcome:\*\* connect into operator workflows.

\- \[ ] Alert export: email text template / webhook payload

\- \[ ] CMMS integration (export tasks)

\- \[ ] GIS integration (GeoJSON outputs, WMS/WFS optional)



---



\## Phase 3 — Scale \& Differentiation



\### S1 — Predictive Modeling (Optional ML)

\*\*Outcome:\*\* improved forecasts and fewer false positives.

\- \[ ] ML only after strong baseline and labels exist

\- \[ ] Explainability requirements remain mandatory

\- \[ ] Model monitoring + drift handling



\### S2 — Multi-tenant SaaS

\*\*Outcome:\*\* commercial platform with access control and billing.

\- \[ ] Tenant separation

\- \[ ] User roles

\- \[ ] Usage-based pricing hooks (km/area/update frequency)

\- \[ ] SLA / monitoring



---



\## Backlog (Ideas, Not MVP)



\- InSAR deformation risk (terrain stability)

\- Flood risk layers (SAR-based)

\- Storm/wind correlation modules

\- “Action recommendations library” (playbooks)

\- Operator-specific thresholds and compliance packs



---



\## Success Metrics (Product)



\- Reduction in outage-causing vegetation events (tracked in pilots)

\- False-positive rate (operator-confirmed)

\- Time-to-decision (from alert to action)

\- Estimated avoided cost (ROI for management)



---



\## Governance



\- Keep commits small and auditable

\- No scope creep beyond current milestone

\- Always follow rules in `AGENT\_CODEX.md`



---



\## Next Step (Current Focus)

\*\*M1 — Ingestion (Sentinel-2 deterministic pipeline)\*\*



