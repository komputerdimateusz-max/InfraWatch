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
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

from infrawatch.scoring import score_traction_segments
from infrawatch.scoring.traction_risk import sample_ndvi_for_line


@dataclass
class NdviDetection:
    path: Path | None
    scene_date: str | None
    scene_folder: Path | None


DATE_PATTERN = re.compile(r"(20\d{6})")
LOG_PATH = Path("ui_runtime.log")
DEFAULT_UTM_EPSG = 32633
LONGITUDE_LIMIT_DEG = 180
LATITUDE_LIMIT_DEG = 90
DEMO_SEGMENT_HALF_LENGTH_M = 150.0


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("ui_runtime")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


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


def _first_geojson_coordinate(payload: dict[str, Any]) -> Sequence[float] | None:
    features = payload.get("features", [])
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if coordinates is None:
            continue
        for coord in iter_coordinates(coordinates):
            if coord and len(coord) >= 2:
                return coord
    return None


def _coordinates_look_projected(coord: Sequence[float] | None) -> bool:
    if not coord or len(coord) < 2:
        return False
    return abs(coord[0]) > LONGITUDE_LIMIT_DEG or abs(coord[1]) > LATITUDE_LIMIT_DEG


def normalize_crs(value):
    from pyproj import CRS

    if value is None:
        return None
    return CRS.from_user_input(value)


def detect_geojson_crs(payload: dict[str, Any], ndvi_crs=None):
    from pyproj import CRS

    crs_payload = payload.get("crs")
    if isinstance(crs_payload, dict):
        props = crs_payload.get("properties") or {}
        name = props.get("name")
        if name:
            return CRS.from_user_input(name)

    coord = _first_geojson_coordinate(payload)
    if _coordinates_look_projected(coord):
        return normalize_crs(ndvi_crs) or CRS.from_epsg(DEFAULT_UTM_EPSG)

    return CRS.from_epsg(4326)


def to_crs_transformer(source, target):
    from pyproj import Transformer

    source = normalize_crs(source)
    target = normalize_crs(target)

    if source is None or target is None:
        return None
    if source == target:
        return None

    return Transformer.from_crs(source, target, always_xy=True)


def transform_geometry(geom, transformer) -> Any:
    from shapely.ops import transform as shapely_transform

    if transformer is None:
        return geom
    return shapely_transform(transformer.transform, geom)


def prepare_lines_for_scoring(
    lines_payload: dict[str, Any], source_crs, target_crs
) -> Path:
    from shapely.geometry import mapping, shape

    if source_crs == target_crs:
        return Path(lines_payload["_path"])

    transformer = to_crs_transformer(source_crs, target_crs)
    transformed_features = []
    for feature in lines_payload.get("features", []):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
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


def load_ndvi_overlay(ndvi_path: Path) -> tuple[str, list[list[float]], Any] | None:
    logger = _get_logger()
    import rasterio
    from matplotlib import pyplot as plt
    from rasterio.enums import Resampling
    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS

    try:
        with rasterio.open(ndvi_path) as dataset:
            scale = min(1.0, 1024 / max(dataset.width, dataset.height))
            new_width = max(1, int(dataset.width * scale))
            new_height = max(1, int(dataset.height * scale))

            preview = dataset.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear,
                masked=True,
            ).astype(np.float32)
            if dataset.nodata is not None:
                preview = np.ma.masked_where(preview == dataset.nodata, preview)
            preview = np.ma.masked_invalid(preview)

            scale_x = dataset.width / new_width
            scale_y = dataset.height / new_height
            preview_transform = dataset.transform * Affine.scale(scale_x, scale_y)
            south, west, north, east = array_bounds(new_height, new_width, preview_transform)
            bounds = (west, south, east, north)
            ndvi_crs = dataset.crs

        colormap = plt.get_cmap("viridis").copy()
        colormap.set_bad(alpha=0)
        buf = io.BytesIO()
        plt.imsave(buf, preview, cmap=colormap, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"

        wgs_bounds = transform_bounds(ndvi_crs, CRS.from_epsg(4326), *bounds)
        bounds_list = [[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]]
        logger.info("ndvi_overlay_loaded path=%s", ndvi_path)
        return data_url, bounds_list, ndvi_crs
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        logger.exception("ndvi_overlay_failed path=%s", ndvi_path)
        return None


def read_ndvi_metadata(ndvi_path: Path) -> tuple[Any, Any, list[list[float]] | None]:
    """Return NDVI CRS, bounds in native CRS, and bounds in EPSG:4326."""
    logger = _get_logger()
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    try:
        with rasterio.open(ndvi_path) as dataset:
            bounds = dataset.bounds
            crs = dataset.crs
        if crs is None:
            return None, bounds, None
        wgs_bounds = transform_bounds(crs, CRS.from_epsg(4326), *bounds)
        bounds_list = [[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]]
        return crs, bounds, bounds_list
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read NDVI metadata: {exc}")
        logger.exception("ndvi_metadata_failed path=%s", ndvi_path)
    return None, None, None


def generate_demo_segment(ndvi_path: Path, output_path: Path) -> Path | None:
    """Generate a short demo LineString centered in the NDVI bounds."""
    import rasterio

    try:
        with rasterio.open(ndvi_path) as dataset:
            bounds = dataset.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        coords = [
            [center_x - DEMO_SEGMENT_HALF_LENGTH_M, center_y],
            [center_x + DEMO_SEGMENT_HALF_LENGTH_M, center_y],
        ]
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"segment_id": 1},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                }
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to generate demo segment: {exc}")
    return None


def _coordinates_to_lists(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_coordinates_to_lists(item) for item in value]
    return value


def normalize_geojson_coordinates(geometry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(geometry)
    if "coordinates" in geometry:
        normalized["coordinates"] = _coordinates_to_lists(geometry["coordinates"])
    return normalized


def iter_coordinates(coordinates: Any) -> Iterable[Sequence[float]]:
    if not isinstance(coordinates, (list, tuple)):
        return
    if coordinates and isinstance(coordinates[0], (int, float)):
        yield coordinates
    else:
        for item in coordinates:
            yield from iter_coordinates(item)


def bounds_from_features(features: list[dict[str, Any]]) -> list[list[float]] | None:
    lats: list[float] = []
    lons: list[float] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        for coord in iter_coordinates(geometry.get("coordinates")):
            if coord and len(coord) >= 2:
                lons.append(coord[0])
                lats.append(coord[1])
    if not lats or not lons:
        return None
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


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
    source_crs,
    ndvi_crs,
    buffer_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from pyproj import CRS
    from shapely.geometry import mapping, shape

    features = lines_payload.get("features", [])
    results_by_id = {}
    if not results_df.empty and "segment_id" in results_df.columns:
        results_by_id = results_df.set_index("segment_id").to_dict(orient="index")

    map_source_crs = ndvi_crs or source_crs
    to_wgs84 = to_crs_transformer(map_source_crs, CRS.from_epsg(4326))
    to_ndvi = to_crs_transformer(source_crs, ndvi_crs) if ndvi_crs else None
    ndvi_to_wgs84 = to_crs_transformer(ndvi_crs, CRS.from_epsg(4326)) if ndvi_crs else None

    line_features = []
    buffer_features = []

    for idx, feature in enumerate(features, start=1):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
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
                "geometry": normalize_geojson_coordinates(mapping(wgs_geom)),
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
                    "geometry": normalize_geojson_coordinates(mapping(buffer_wgs)),
                    "properties": {"segment_id": idx},
                }
            )

    return line_features, buffer_features


def first_reprojected_coordinate(features: list[dict[str, Any]]) -> Sequence[float] | None:
    for feature in features:
        geometry = feature.get("geometry") or {}
        for coord in iter_coordinates(geometry.get("coordinates")):
            if coord and len(coord) >= 2:
                return coord
    return None


def sanitize_properties_for_map(properties: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            sanitized[key] = "N/A"
        elif isinstance(value, float) and np.isnan(value):
            sanitized[key] = "N/A"
        elif pd.isna(value):
            sanitized[key] = "N/A"
        else:
            sanitized[key] = value
    return sanitized


def sanitize_features_for_map(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_features = []
    for feature in features:
        properties = feature.get("properties") or {}
        sanitized = dict(feature)
        sanitized["properties"] = sanitize_properties_for_map(properties)
        sanitized_features.append(sanitized)
    return sanitized_features


def create_map(
    ndvi_overlay: tuple[str, list[list[float]], CRS] | None,
    line_features: list[dict[str, Any]],
    buffer_features: list[dict[str, Any]],
    show_buffers: bool,
    opacity: float,
    fit_bounds: list[list[float]] | None,
) -> folium.Map:
    if fit_bounds:
        center_lat = (fit_bounds[0][0] + fit_bounds[1][0]) / 2
        center_lon = (fit_bounds[0][1] + fit_bounds[1][1]) / 2
    else:
        center_lat, center_lon = 0, 0

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

    if line_features:
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
    if fit_bounds:
        fmap.fit_bounds(fit_bounds)
    return fmap


def main() -> None:
    st.set_page_config(page_title="InfraWatch MVP", layout="wide")
    st.title("InfraWatch — Traction Vegetation Risk")
    st.info("UI heartbeat: Streamlit app is running.")

    logger = _get_logger()
    logger.info("ui_start")

    base_dir = Path("satellite_data/raw/s2")
    ndvi_detection = NdviDetection(path=None, scene_date=None, scene_folder=None)
    try:
        ndvi_detection = detect_latest_ndvi(base_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to detect latest NDVI: {exc}")

    ndvi_default = str(ndvi_detection.path) if ndvi_detection.path else ""
    lines_default = str(Path("tests/data/demo_lines.geojson"))
    risk_default = "traction_risk.json" if Path("traction_risk.json").exists() else ""
    demo_segment_path = Path("satellite_data/demo_segment.geojson")

    if "lines_path" not in st.session_state:
        st.session_state["lines_path"] = lines_default

    with st.sidebar:
        st.header("Inputs")
        ndvi_path_input = st.text_input("NDVI GeoTIFF path", value=ndvi_default)
        enable_ndvi_overlay = st.checkbox("Enable NDVI overlay (experimental)", value=False)
        if ndvi_detection.path:
            st.caption(f"Scene date: {ndvi_detection.scene_date}")
            st.caption(f"Scene folder: {ndvi_detection.scene_folder}")
        else:
            st.warning(
                "No NDVI found under satellite_data/raw/s2/**/ndvi.tif. "
                "Run scripts/compute_ndvi.py to generate one."
            )
        if st.button("Generate demo segment inside NDVI extent"):
            ndvi_path_candidate = Path(ndvi_path_input).expanduser()
            if ndvi_path_candidate.exists():
                generated = generate_demo_segment(ndvi_path_candidate, demo_segment_path)
                if generated:
                    st.session_state["lines_path"] = str(generated)
                    st.success(f"Generated demo segment at {generated}")
            else:
                st.warning("NDVI path not found; unable to generate demo segment.")
        lines_path_input = st.text_input("Traction lines GeoJSON path", key="lines_path")
        risk_path_input = st.text_input("Risk JSON path", value=risk_default)
        buffer_m = st.slider("Buffer distance (meters)", min_value=1, max_value=100, value=20)
        opacity = st.slider("NDVI overlay opacity", min_value=0.1, max_value=1.0, value=0.6)
        run_scoring = st.button("Run scoring")
        show_buffers = st.checkbox("Show buffer polygons", value=True)
        show_debug = st.checkbox("Show debug details", value=False)

    ndvi_path = Path(ndvi_path_input).expanduser()
    lines_path = Path(lines_path_input).expanduser()
    risk_path = Path(risk_path_input).expanduser() if risk_path_input else None

    lines_payload = {}
    ndvi_overlay = None
    ndvi_crs = None
    lines_crs = None
    ndvi_bounds = None
    ndvi_bounds_wgs84 = None

    if ndvi_path_input and ndvi_path.exists():
        ndvi_crs, ndvi_bounds, ndvi_bounds_wgs84 = read_ndvi_metadata(ndvi_path)

    if enable_ndvi_overlay:
        logger.info("ndvi_overlay_enabled path=%s", ndvi_path)
        if ndvi_path_input and ndvi_path.exists():
            ndvi_overlay = load_ndvi_overlay(ndvi_path)
            if ndvi_overlay:
                ndvi_crs = ndvi_overlay[2]
                if not ndvi_bounds_wgs84:
                    ndvi_bounds_wgs84 = ndvi_overlay[1]
        else:
            st.warning("NDVI path not found; overlay disabled.")

    if lines_path.exists():
        try:
            lines_payload = load_geojson(lines_path)
            lines_payload["_path"] = str(lines_path)
            logger.info("lines_loaded path=%s", lines_path)
            lines_crs = detect_geojson_crs(lines_payload, ndvi_crs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load traction lines: {exc}")
            lines_payload = {}
    else:
        st.error("Traction lines GeoJSON not found.")

    results: list[dict[str, Any]] = []
    if lines_payload and (run_scoring or not (risk_path and risk_path.exists())):
        if ndvi_path.exists():
            try:
                source_crs = lines_crs or detect_geojson_crs(lines_payload, ndvi_crs)
                scoring_path = prepare_lines_for_scoring(lines_payload, source_crs, ndvi_crs or source_crs)
                results = score_traction_segments(ndvi_path, scoring_path, buffer_m=buffer_m)
                st.info("Scoring computed from NDVI and traction lines.")
                logger.info("scoring_computed ndvi=%s lines=%s", ndvi_path, scoring_path)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")
        else:
            st.warning("NDVI missing; cannot compute risk scores.")
    elif risk_path and risk_path.exists():
        try:
            results = json.loads(risk_path.read_text(encoding="utf-8"))
            st.info(f"Loaded risk results from {risk_path}")
            logger.info("risk_results_loaded path=%s", risk_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load risk JSON: {exc}")

    try:
        results_df = build_results_table(results)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to prepare risk table: {exc}")
        results_df = pd.DataFrame()

    if not results_df.empty and (results_df["data_status"] == "NO_DATA").any():
        st.warning(
            "NDVI returned NO_DATA for one or more segments. This usually means the segment "
            "geometry does not overlap the NDVI raster or uses a mismatched CRS."
        )
        if show_debug and ndvi_bounds is not None:
            debug_payload: dict[str, Any] = {
                "ndvi_bounds_native": {
                    "left": ndvi_bounds.left,
                    "bottom": ndvi_bounds.bottom,
                    "right": ndvi_bounds.right,
                    "top": ndvi_bounds.top,
                },
                "ndvi_bounds_wgs84": ndvi_bounds_wgs84,
            }
            if lines_payload and ndvi_path.exists():
                source_crs = lines_crs or detect_geojson_crs(lines_payload, ndvi_crs)
                from shapely.geometry import shape

                first_feature = next(
                    (feature for feature in lines_payload.get("features", []) if feature.get("geometry")), None
                )
                if first_feature:
                    geom = shape(first_feature["geometry"])
                    sample_stats = sample_ndvi_for_line(
                        ndvi_path,
                        geom,
                        buffer_m,
                        line_crs=source_crs,
                    )
                    debug_payload["sample_line_stats"] = sample_stats
                    debug_payload["sample_buffer_m"] = buffer_m
                    debug_payload["sample_geom_type"] = geom.geom_type
            st.code(json.dumps(debug_payload, indent=2))

    if lines_payload:
        try:
            source_crs = lines_crs or detect_geojson_crs(lines_payload, ndvi_crs)
            line_features, buffer_features = build_map_features(
                lines_payload, results_df, source_crs, ndvi_crs, buffer_m
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to prepare map features: {exc}")
            line_features, buffer_features = [], []
    else:
        line_features, buffer_features = [], []

    if lines_payload:
        lines_crs_display = normalize_crs(lines_crs)
        ndvi_crs_display = normalize_crs(ndvi_crs)
        st.caption(
            f"Detected lines CRS: {lines_crs_display.to_string() if lines_crs_display else 'unknown'}"
        )
        st.caption(f"NDVI CRS: {ndvi_crs_display.to_string() if ndvi_crs_display else 'unavailable'}")
        first_coord = first_reprojected_coordinate(line_features)
        if first_coord:
            st.caption(f"First reprojected coordinate (lon, lat): {first_coord[0]:.6f}, {first_coord[1]:.6f}")
        else:
            st.caption("First reprojected coordinate (lon, lat): unavailable")

    map_column, table_column = st.columns([3, 2])

    with map_column:
        try:
            map_features = sanitize_features_for_map(line_features)
            fit_bounds = bounds_from_features(map_features)
            if fit_bounds is None and ndvi_bounds_wgs84:
                fit_bounds = ndvi_bounds_wgs84
            if fit_bounds is None:
                fit_bounds = bounds_from_features(line_features)
            buffer_features_for_map = sanitize_features_for_map(buffer_features)
            if not map_features:
                st.info("No segments available to display on map.")
            fmap = create_map(
                ndvi_overlay=ndvi_overlay,
                line_features=map_features,
                buffer_features=buffer_features_for_map,
                show_buffers=show_buffers,
                opacity=opacity,
                fit_bounds=fit_bounds,
            )
            st_folium(fmap, width=800, height=600)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to render map: {exc}")

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
