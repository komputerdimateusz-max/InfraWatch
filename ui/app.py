"""InfraWatch Streamlit UI.

Quick run:
  pip install -e .
  pip install streamlit folium streamlit-folium matplotlib
  streamlit run ui/app.py
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.features import GeoJsonTooltip
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from streamlit_folium import st_folium

from infrawatch.scoring import score_traction_segments


@dataclass
class NdviDetection:
    path: Path | None
    scene_date: str | None
    scene_folder: Path | None


DATE_PATTERN = re.compile(r"(20\d{6})")


def _parse_scene_date(path: Path) -> datetime | None:
    for part in path.parts:
        match = DATE_PATTERN.search(part)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except ValueError:
                continue
    return None


def detect_latest_ndvi(base_dir: Path) -> NdviDetection:
    candidates = list(base_dir.rglob("ndvi.tif"))
    if not candidates:
        return NdviDetection(path=None, scene_date=None, scene_folder=None)

    dated: list[tuple[datetime, Path]] = []
    undated: list[Path] = []
    for path in candidates:
        parsed = _parse_scene_date(path)
        if parsed:
            dated.append((parsed, path))
        else:
            undated.append(path)

    if dated:
        dated.sort(key=lambda item: item[0])
        scene_date, selected = dated[-1]
        return NdviDetection(
            path=selected,
            scene_date=scene_date.strftime("%Y-%m-%d"),
            scene_folder=selected.parent,
        )

    selected = max(undated, key=lambda item: item.stat().st_mtime)
    return NdviDetection(
        path=selected,
        scene_date=datetime.fromtimestamp(selected.stat().st_mtime).strftime("%Y-%m-%d"),
        scene_folder=selected.parent,
    )


def load_geojson(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        return payload
    if payload.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [payload]}
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": payload, "properties": {}}]}


def detect_geojson_crs(payload: dict[str, Any]) -> CRS:
    crs_payload = payload.get("crs")
    if isinstance(crs_payload, dict):
        props = crs_payload.get("properties") or {}
        name = props.get("name")
        if name:
            return CRS.from_user_input(name)
    return CRS.from_epsg(4326)


def to_crs_transformer(source: CRS, target: CRS):
    if source == target:
        return None
    from pyproj import Transformer

    return Transformer.from_crs(source, target, always_xy=True)


def transform_geometry(geom, transformer) -> Any:
    if transformer is None:
        return geom
    return shapely_transform(transformer.transform, geom)


def prepare_lines_for_scoring(
    lines_payload: dict[str, Any], source_crs: CRS, target_crs: CRS
) -> Path:
    if source_crs == target_crs:
        return Path(lines_payload["_path"])

    transformer = to_crs_transformer(source_crs, target_crs)
    transformed_features = []
    for feature in lines_payload.get("features", []):
        geom = shape(feature.get("geometry"))
        projected = transform_geometry(geom, transformer)
        new_feature = dict(feature)
        new_feature["geometry"] = mapping(projected)
        transformed_features.append(new_feature)

    transformed_payload = {
        "type": "FeatureCollection",
        "features": transformed_features,
    }
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
    temp.write(json.dumps(transformed_payload).encode("utf-8"))
    temp.close()
    return Path(temp.name)


def load_ndvi_overlay(ndvi_path: Path) -> tuple[str, list[list[float]], CRS]:
    with rasterio.open(ndvi_path) as dataset:
        ndvi = dataset.read(1).astype(np.float32)
        if dataset.nodata is not None:
            ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi)

        masked = np.ma.masked_invalid(ndvi)

        from matplotlib import pyplot as plt

        colormap = plt.get_cmap("viridis").copy()
        colormap.set_bad(alpha=0)
        buf = io.BytesIO()
        plt.imsave(buf, masked, cmap=colormap, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"

        bounds = dataset.bounds
        wgs_bounds = transform_bounds(dataset.crs, CRS.from_epsg(4326), *bounds)
        bounds_list = [[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]]
        return data_url, bounds_list, dataset.crs


def build_results_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df

    data_status = np.where(
        df["mean_ndvi"].isna() | df["p90_ndvi"].isna(),
        "NO_DATA",
        "OK",
    )
    df["data_status"] = data_status
    df["risk_category_display"] = np.where(
        df["data_status"] == "NO_DATA", "NO_DATA", df["risk_category"]
    )
    df["risk_score_display"] = np.where(df["data_status"] == "NO_DATA", np.nan, df["risk_score"])

    status_order = {"OK": 0, "NO_DATA": 1}
    category_order = {"HIGH": 0, "MED": 1, "LOW": 2, "NO_DATA": 3}
    df["_status_order"] = df["data_status"].map(status_order).fillna(2)
    df["_category_order"] = df["risk_category_display"].map(category_order).fillna(4)
    df = df.sort_values(
        by=["_status_order", "_category_order", "risk_score_display"],
        ascending=[True, True, False],
    )
    return df


def build_map_features(
    lines_payload: dict[str, Any],
    results_df: pd.DataFrame,
    source_crs: CRS,
    ndvi_crs: CRS | None,
    buffer_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = lines_payload.get("features", [])
    results_by_id = {}
    if not results_df.empty and "segment_id" in results_df.columns:
        results_by_id = results_df.set_index("segment_id").to_dict(orient="index")

    to_wgs84 = to_crs_transformer(source_crs, CRS.from_epsg(4326))
    to_ndvi = to_crs_transformer(source_crs, ndvi_crs) if ndvi_crs else None
    ndvi_to_wgs84 = to_crs_transformer(ndvi_crs, CRS.from_epsg(4326)) if ndvi_crs else None

    line_features = []
    buffer_features = []

    for idx, feature in enumerate(features, start=1):
        geom = shape(feature.get("geometry"))
        wgs_geom = transform_geometry(geom, to_wgs84)
        result = results_by_id.get(idx, {})
        properties = {
            "segment_id": idx,
            "mean_ndvi": result.get("mean_ndvi"),
            "p90_ndvi": result.get("p90_ndvi"),
            "pct_above_0_6": result.get("pct_above_0_6"),
            "risk_score": result.get("risk_score_display", result.get("risk_score")),
            "risk_category": result.get("risk_category_display", result.get("risk_category")),
            "data_status": result.get("data_status", "OK"),
        }
        line_features.append(
            {
                "type": "Feature",
                "geometry": mapping(wgs_geom),
                "properties": properties,
            }
        )

        if ndvi_crs and buffer_m > 0:
            ndvi_geom = transform_geometry(geom, to_ndvi)
            buffered = ndvi_geom.buffer(buffer_m)
            buffer_wgs = transform_geometry(buffered, ndvi_to_wgs84)
            buffer_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(buffer_wgs),
                    "properties": {"segment_id": idx},
                }
            )

    return line_features, buffer_features


def create_map(
    ndvi_overlay: tuple[str, list[list[float]], CRS] | None,
    line_features: list[dict[str, Any]],
    buffer_features: list[dict[str, Any]],
    show_buffers: bool,
    opacity: float,
) -> folium.Map:
    if ndvi_overlay:
        _, bounds, _ = ndvi_overlay
        center_lat = (bounds[0][0] + bounds[1][0]) / 2
        center_lon = (bounds[0][1] + bounds[1][1]) / 2
    elif line_features:
        sample = line_features[0]["geometry"]["coordinates"][0]
        center_lon, center_lat = sample
        bounds = None
    else:
        center_lat, center_lon = 0, 0
        bounds = None

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

    if ndvi_overlay:
        image, bounds, _ = ndvi_overlay
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=bounds,
            opacity=opacity,
            name="NDVI",
        ).add_to(fmap)

    def style_line(feature):
        category = feature["properties"].get("risk_category")
        color_map = {
            "LOW": "#2ca25f",
            "MED": "#fdae61",
            "HIGH": "#d73027",
            "NO_DATA": "#9e9e9e",
        }
        return {
            "color": color_map.get(category, "#2ca25f"),
            "weight": 4,
        }

    tooltip = GeoJsonTooltip(
        fields=[
            "segment_id",
            "mean_ndvi",
            "p90_ndvi",
            "pct_above_0_6",
            "risk_score",
            "risk_category",
            "data_status",
        ],
        aliases=[
            "Segment ID",
            "Mean NDVI",
            "P90 NDVI",
            "Pct > 0.6",
            "Risk score",
            "Risk category",
            "Data status",
        ],
        sticky=True,
        localize=True,
    )

    folium.GeoJson(line_features, name="Traction Segments", style_function=style_line, tooltip=tooltip).add_to(
        fmap
    )

    if show_buffers and buffer_features:
        folium.GeoJson(
            buffer_features,
            name="Buffers",
            style_function=lambda feature: {
                "color": "#3182bd",
                "weight": 1,
                "fill": True,
                "fillColor": "#3182bd",
                "fillOpacity": 0.2,
            },
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def main() -> None:
    st.set_page_config(page_title="InfraWatch MVP", layout="wide")
    st.title("InfraWatch — Traction Vegetation Risk")

    base_dir = Path("satellite_data/raw/s2")
    ndvi_detection = detect_latest_ndvi(base_dir)

    ndvi_default = str(ndvi_detection.path) if ndvi_detection.path else ""
    lines_default = str(Path("tests/data/demo_lines.geojson"))
    risk_default = "traction_risk.json" if Path("traction_risk.json").exists() else ""

    with st.sidebar:
        st.header("Inputs")
        ndvi_path_input = st.text_input("NDVI GeoTIFF path", value=ndvi_default)
        if ndvi_detection.path:
            st.caption(f"Scene date: {ndvi_detection.scene_date}")
            st.caption(f"Scene folder: {ndvi_detection.scene_folder}")
        else:
            st.warning(
                "No NDVI found under satellite_data/raw/s2/**/ndvi.tif. "
                "Run scripts/compute_ndvi.py to generate one."
            )
        lines_path_input = st.text_input("Traction lines GeoJSON path", value=lines_default)
        risk_path_input = st.text_input("Risk JSON path", value=risk_default)
        buffer_m = st.slider("Buffer distance (meters)", min_value=1, max_value=100, value=20)
        opacity = st.slider("NDVI overlay opacity", min_value=0.1, max_value=1.0, value=0.6)
        run_scoring = st.button("Run scoring")
        show_buffers = st.checkbox("Show buffer polygons", value=True)

    ndvi_path = Path(ndvi_path_input).expanduser()
    lines_path = Path(lines_path_input).expanduser()
    risk_path = Path(risk_path_input).expanduser() if risk_path_input else None

    lines_payload = {}
    ndvi_overlay = None
    ndvi_crs = None

    if ndvi_path_input and ndvi_path.exists():
        try:
            ndvi_overlay = load_ndvi_overlay(ndvi_path)
            ndvi_crs = ndvi_overlay[2]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load NDVI: {exc}")

    if lines_path.exists():
        lines_payload = load_geojson(lines_path)
        lines_payload["_path"] = str(lines_path)
    else:
        st.error("Traction lines GeoJSON not found.")

    results: list[dict[str, Any]] = []
    if lines_payload and (run_scoring or not (risk_path and risk_path.exists())):
        if ndvi_path.exists():
            source_crs = detect_geojson_crs(lines_payload)
            try:
                scoring_path = prepare_lines_for_scoring(lines_payload, source_crs, ndvi_crs or source_crs)
                results = score_traction_segments(ndvi_path, scoring_path, buffer_m=buffer_m)
                st.info("Scoring computed from NDVI and traction lines.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")
        else:
            st.warning("NDVI missing; cannot compute risk scores.")
    elif risk_path and risk_path.exists():
        try:
            results = json.loads(risk_path.read_text(encoding="utf-8"))
            st.info(f"Loaded risk results from {risk_path}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load risk JSON: {exc}")

    results_df = build_results_table(results)

    if lines_payload:
        source_crs = detect_geojson_crs(lines_payload)
        line_features, buffer_features = build_map_features(
            lines_payload, results_df, source_crs, ndvi_crs, buffer_m
        )
    else:
        line_features, buffer_features = [], []

    map_column, table_column = st.columns([3, 2])

    with map_column:
        fmap = create_map(
            ndvi_overlay=ndvi_overlay,
            line_features=line_features,
            buffer_features=buffer_features,
            show_buffers=show_buffers,
            opacity=opacity,
        )
        st_folium(fmap, width=800, height=600)

    with table_column:
        st.subheader("Risk Summary")
        if not results_df.empty:
            total = len(results_df)
            no_data = int((results_df["data_status"] == "NO_DATA").sum())
            high = int((results_df["risk_category_display"] == "HIGH").sum())
            med = int((results_df["risk_category_display"] == "MED").sum())
            low = int((results_df["risk_category_display"] == "LOW").sum())
            st.metric("Segments", total)
            st.metric("High risk", high)
            st.metric("Medium risk", med)
            st.metric("Low risk", low)
            st.metric("No data", no_data)

            display_columns = [
                "segment_id",
                "risk_category_display",
                "risk_score_display",
                "mean_ndvi",
                "p90_ndvi",
                "pct_above_0_6",
                "data_status",
            ]
            display_df = results_df[display_columns].rename(
                columns={
                    "risk_category_display": "risk_category",
                    "risk_score_display": "risk_score",
                }
            )
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No risk results to display yet.")


if __name__ == "__main__":
    main()
