# InfraWatch™
### Satellite-based vegetation risk monitoring for traction & power networks

InfraWatch™ is an Earth Observation (EO) analytics application designed to **detect, assess, and predict vegetation-related risks** for traction networks and linear infrastructure using satellite imagery.

The system transforms raw satellite data into **actionable risk indicators, alerts, and audit-ready reports** for infrastructure operators.

> We do not sell satellite images.  
> We sell early warnings and operational decisions.

---

## 🎯 Project Goal

The primary goal of InfraWatch™ is to **reduce outages, failures, and maintenance costs** caused by uncontrolled vegetation growth near traction and power networks.

The application focuses on:
- early detection of vegetation encroachment,
- risk prioritization of network segments,
- decision support for maintenance planning.

---

## 🧠 Core Use Case (v1)

**Vegetation risk monitoring along traction / power lines**

The system analyzes vegetation growth within a defined corridor around:
- railway traction networks,
- power transmission and distribution lines,
- other linear infrastructure.

Typical risks addressed:
- tree and shrub encroachment,
- seasonal vegetation growth,
- increased outage probability during wind or storms.

---

## 🛰️ Data Sources

InfraWatch™ is built on **open and legally reusable satellite data**.

### Primary EO data
- **Sentinel-2 (multispectral, optical)**  
  - vegetation indices (NDVI, EVI),
  - seasonal and temporal trends,
  - spatial analysis at 10 m resolution.

- **Sentinel-1 (SAR radar)**  
  - change detection independent of clouds and daylight,
  - continuity of monitoring in all weather conditions.

All data is provided by the **Copernicus Programme** and can be used for **commercial services**.

---

## ⬇️ Downloading NDVI data

The Streamlit UI now includes a **NDVI Downloader** sidebar section for pulling additional
Sentinel-2 scenes and generating `ndvi.tif` files automatically.

### Workflow
1. **Draw a line** on the map (or enter a manual bbox) so the downloader knows the AOI.
2. In the sidebar, open **NDVI Downloader** and choose:
   - AOI source (drawn buffer bbox or manual bbox)
   - date range
   - maximum cloud cover
   - backend (CDSE or AWS Open Data)
3. Click **Search**, then **Download** on a scene to fetch data and compute NDVI.
4. The app stores outputs in:
   ```
   C:\\InfraWatch\\satellite_data\\raw\\s2\\YYYYMMDD\\<scene_folder>\\ndvi.tif
   ```
   and immediately rescans available NDVI dates.

### Credentials (optional)
CDSE downloads require credentials. Copy `.env.example` to `.env` and set:
```
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password
```
If CDSE fails, switch to **AWS Open Data (Earth Search)** which does not require credentials.

### Windows setup (PowerShell)
```
Copy-Item .env.example .env
notepad .env
```

### Windows setup (cmd)
```
copy .env.example .env
notepad .env
```

---

## 📊 Key Analytics & Metrics

### Vegetation Analytics
- NDVI / vegetation density indicators
- vegetation growth rate
- seasonal normalization

### Change Detection
- comparison of time-series imagery
- detection of abnormal vegetation growth patterns

### Risk Scoring
- **Vegetation Encroachment Index (VEI)**
- **Risk Score (0–100)** per network segment
- forecasted risk horizons (7 / 14 / 30 days)

---

## 🚨 Product Outputs

InfraWatch™ produces **decision-oriented outputs**, not raw imagery.

### Outputs include:
- 🚨 automated alerts (email / API)
- 📊 risk dashboards (map + trends)
- 📄 PDF reports (audit & ESG ready)
- 🔌 future API for CMMS / GIS integration

Example alert:
> *“Segment L-17: vegetation risk score 82/100 – maintenance recommended within 14 days.”*

---

## 🏗️ System Architecture (high level)
Satellite EO Data
↓
Ingestion & Preprocessing
↓
Vegetation & Change Analytics
↓
Risk Scoring Engine
↓
Alerts • Dashboard • Reports


The architecture is modular, auditable, and scalable.

---

## 🔬 MVP Scope

The initial MVP intentionally avoids over-engineering.

**Included:**
- rule-based vegetation metrics,
- deterministic risk scoring,
- batch processing,
- PDF reporting and simple dashboard.

**Explicitly excluded (for MVP):**
- machine learning / black-box models,
- real-time guarantees,
- hardware or field sensors.

---

## 🧪 Key KPIs

- vegetation growth rate near infrastructure
- number of high-risk segments detected
- false positive rate
- estimated avoided outage cost
- time from detection to decision

---

## ⚙️ Technology Stack (initial)

- Python
- NumPy / Pandas
- GeoPandas / RasterIO
- Streamlit (dashboard)
- PDF report generation
- Docker (future)

---

## 🧭 Design Principles

- **Decision-first** – metrics over maps
- **Explainability** – every score must be auditable
- **Open data first** – no vendor lock-in
- **Industrial usability** – not academic demos

---

## 🛡️ Compliance & Ethics

- Uses publicly available EO data
- No personal data processing
- Designed for ESG, audit, and compliance use cases
- No surveillance or individual tracking

---

## 🚀 Roadmap (high level)

### Phase 1 – MVP
- EO ingestion (Sentinel-1 / Sentinel-2)
- vegetation indices
- risk scoring
- PDF reporting

### Phase 2 – Pilot
- operator feedback
- threshold calibration
- dashboard UX improvements

### Phase 3 – Scale
- predictive models
- API integrations
- multi-region support
- enterprise SLA

---

## 🤖 Codex / Agent Usage

This repository is designed to be developed with AI agents (Codex).

All agent behavior, scope rules, and development constraints are defined in:
**`AGENT_CODEX.md`**

---

## 📌 Status

Project status: **Early development / MVP**

---

## 📬 Contact
Repository owner: InfraWatch  



