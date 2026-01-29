const DEFAULT_CENTER = [52.6, 15.8];
const DEFAULT_ZOOM = 8;

const map = L.map("map").setView(DEFAULT_CENTER, DEFAULT_ZOOM);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 19,
}).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const bufferLayer = new L.FeatureGroup();
map.addLayer(bufferLayer);

const drawControl = new L.Control.Draw({
  edit: { featureGroup: drawnItems },
  draw: {
    polygon: false,
    rectangle: false,
    circle: false,
    marker: false,
    circlemarker: false,
  },
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, (event) => {
  drawnItems.addLayer(event.layer);
});

map.on(L.Draw.Event.DELETED, () => {
  bufferLayer.clearLayers();
});

const dateSelect = document.getElementById("dateSelect");
const scoreButton = document.getElementById("scoreButton");
const bufferInput = document.getElementById("bufferInput");
const thresholdInput = document.getElementById("thresholdInput");
const warningsEl = document.getElementById("warnings");
const healthStatus = document.getElementById("healthStatus");

function setHealth(status) {
  if (status) {
    healthStatus.textContent = "API OK";
    healthStatus.style.background = "#dcfce7";
    healthStatus.style.color = "#166534";
  } else {
    healthStatus.textContent = "API OFFLINE";
    healthStatus.style.background = "#fee2e2";
    healthStatus.style.color = "#991b1b";
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    setHealth(payload.ok === true);
  } catch (err) {
    setHealth(false);
  }
}

async function loadNdviDates() {
  try {
    const response = await fetch("/api/ndvi/dates");
    const payload = await response.json();
    dateSelect.innerHTML = "";
    payload.dates.forEach((date) => {
      const option = document.createElement("option");
      option.value = date;
      option.textContent = date;
      dateSelect.appendChild(option);
    });
    if (payload.latest_bounds) {
      const [[west, south], [east, north]] = payload.latest_bounds;
      const bounds = L.latLngBounds(
        L.latLng(south, west),
        L.latLng(north, east)
      );
      map.fitBounds(bounds.pad(0.1));
    }
  } catch (err) {
    warningsEl.textContent = "Failed to load NDVI dates.";
  }
}

function collectDrawnLines() {
  const geo = drawnItems.toGeoJSON();
  return geo;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    warningsEl.textContent = "";
    return;
  }
  warningsEl.innerHTML = warnings.map((w) => `<div>⚠️ ${w}</div>`).join("");
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }
  if (Number.isNaN(value)) {
    return "N/A";
  }
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  return value;
}

function updateTable(tableId, rows, columns) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = formatValue(row[col]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function updateTrendChart(trendRows) {
  const bySegment = {};
  trendRows.forEach((row) => {
    if (!bySegment[row.segment_id]) {
      bySegment[row.segment_id] = [];
    }
    bySegment[row.segment_id].push(row);
  });

  const traces = Object.entries(bySegment).map(([segmentId, rows]) => {
    const sorted = rows.slice().sort((a, b) => a.date.localeCompare(b.date));
    return {
      x: sorted.map((r) => r.date),
      y: sorted.map((r) => r.risk_score),
      type: "scatter",
      mode: "lines+markers",
      name: `Segment ${segmentId}`,
    };
  });

  const layout = {
    margin: { t: 10, l: 40, r: 10, b: 40 },
    yaxis: { title: "Risk Score" },
    xaxis: { title: "Date" },
  };

  Plotly.newPlot("trendChart", traces, layout, { responsive: true });
}

function updateBuffers() {
  bufferLayer.clearLayers();
  const bufferMeters = parseFloat(bufferInput.value || "0");
  if (!bufferMeters || bufferMeters <= 0) {
    return;
  }
  const geo = collectDrawnLines();
  geo.features.forEach((feature) => {
    try {
      const buffered = turf.buffer(feature, bufferMeters, { units: "meters" });
      const layer = L.geoJSON(buffered, { color: "#16a34a", weight: 1, fillOpacity: 0.15 });
      bufferLayer.addLayer(layer);
    } catch (err) {
      // ignore buffer errors on malformed geometry
    }
  });
}

scoreButton.addEventListener("click", async () => {
  const featureCollection = collectDrawnLines();
  if (!featureCollection.features || featureCollection.features.length === 0) {
    renderWarnings(["Draw at least one line to score."]);
    return;
  }

  const selectedDates = Array.from(dateSelect.selectedOptions).map((opt) => opt.value);
  const payload = {
    feature_collection: featureCollection,
    dates: selectedDates,
    buffer_m: parseFloat(bufferInput.value || "0"),
    ndvi_threshold: parseFloat(thresholdInput.value || "0.6"),
  };

  scoreButton.disabled = true;
  scoreButton.textContent = "Scoring...";
  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    renderWarnings(data.warnings || []);
    updateTable("segmentsTable", data.segments || [], [
      "segment_id",
      "mean_ndvi",
      "p90_ndvi",
      "pct_above_0_6",
      "risk_score",
      "risk_category",
      "data_status",
    ]);
    updateTable("trendTable", data.trend || [], [
      "date",
      "segment_id",
      "mean_ndvi",
      "risk_score",
      "risk_category",
      "data_status",
    ]);
    updateTrendChart(data.trend || []);
    updateBuffers();
  } catch (err) {
    renderWarnings(["Scoring request failed. Check server logs."]);
  } finally {
    scoreButton.disabled = false;
    scoreButton.textContent = "Score";
  }
});

map.on("draw:edited", updateBuffers);
map.on("draw:created", updateBuffers);

fetchHealth();
loadNdviDates();
