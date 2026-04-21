#!/usr/bin/env python3
"""
Diagnose point-axis mismatches between napari JSON files and tracker ingestion.

This tool is designed for the exact scenario where:
- points look correct in napari after load, but
- multiprocessing tracking behaves as if axes are swapped.

It evaluates candidate axis orders (including declared `point_axes`) against
the selected volume and reports:
- in-bounds fit in tracker zyx space,
- which axis order is best,
- whether manual napari save (without layer axis metadata) would scramble axes.

Example:
python /home/cfxuser/src/tubule-tracker/tubulemap/diagnose_point_axes_pipeline.py \
  --points-json /home/cfxuser/src/tubule-tracker/tubulemap/Demo/starting_points/nephron1.json \
  --volume /media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/demo_data/oldeci_bbox_cropped.zarr/0 \
  --json-out /tmp/point_axes_pipeline_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import zarr
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: zarr. Activate the tracking environment that includes zarr "
        "(for example: conda activate tubuletracker)."
    ) from exc

# Allow absolute-path execution from outside repo root.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SPATIAL_AXES = ("z", "y", "x")
AXIS_PERMUTATIONS_3D = [
    ["z", "y", "x"],
    ["z", "x", "y"],
    ["y", "z", "x"],
    ["y", "x", "z"],
    ["x", "z", "y"],
    ["x", "y", "z"],
]


def _default_axes_for_ndim(ndim: int) -> List[str]:
    """Return fallback axis names when metadata is missing."""
    if ndim == 5:
        return ["t", "c", "z", "y", "x"]
    if ndim == 4:
        return ["c", "z", "y", "x"]
    if ndim == 3:
        return ["z", "y", "x"]
    if ndim == 2:
        return ["y", "x"]
    return [f"dim_{i}" for i in range(ndim)]


def _normalize_axes_meta(axes_meta: Any, ndim: int) -> List[str]:
    """Normalize axis metadata into a lower-case axis list of length ndim."""
    if axes_meta is None:
        return _default_axes_for_ndim(ndim)
    names: List[str] = []
    for axis in axes_meta:
        if isinstance(axis, dict):
            name = str(axis.get("name", "")).strip().lower()
        else:
            name = str(axis).strip().lower()
        names.append(name)
    if len(names) != ndim:
        return _default_axes_for_ndim(ndim)
    return names


def _axis_index(axes: Sequence[str], axis_name: str) -> Optional[int]:
    """Find axis index by name."""
    target = str(axis_name).strip().lower()
    for idx, axis in enumerate(axes):
        if str(axis).strip().lower() == target:
            return idx
    return None


def _shape_zyx_from_shape(shape: Sequence[int], axes: Sequence[str]) -> Optional[Tuple[int, int, int]]:
    """Extract zyx shape tuple from a full shape and axis list."""
    z_idx = _axis_index(axes, "z")
    y_idx = _axis_index(axes, "y")
    x_idx = _axis_index(axes, "x")
    if z_idx is None or y_idx is None or x_idx is None:
        return None
    if max(z_idx, y_idx, x_idx) >= len(shape):
        return None
    return int(shape[z_idx]), int(shape[y_idx]), int(shape[x_idx])


def _is_numeric_key(name: str) -> bool:
    """Return whether a zarr group key is an integer-like level name."""
    try:
        int(name)
        return True
    except Exception:
        return False


def _sorted_level_entries(entries: Iterable[Tuple[str, zarr.Array]]) -> List[Tuple[str, zarr.Array]]:
    """Sort pyramid level arrays by numeric key if possible."""
    materialized = list(entries)
    numeric_entries = [(name, arr) for name, arr in materialized if _is_numeric_key(name)]
    if len(numeric_entries) == len(materialized):
        return sorted(materialized, key=lambda item: int(item[0]))
    return sorted(materialized, key=lambda item: item[0])


def _inspect_regular_array(path: str, array: zarr.Array) -> Dict[str, Any]:
    """Inspect plain zarr array source."""
    axes = _default_axes_for_ndim(array.ndim)
    levels = [
        {
            "index": 0,
            "path": "",
            "absolute_path": path,
            "shape": [int(v) for v in array.shape],
        }
    ]
    return {
        "path": path,
        "source_kind": "regular",
        "axes": axes,
        "is_multiscale": False,
        "levels": levels,
    }


def _inspect_regular_group(path: str, group: zarr.Group) -> Dict[str, Any]:
    """Inspect regular multiscale group with numeric/non-numeric level keys."""
    level_entries: List[Tuple[str, zarr.Array]] = []
    for key in list(group.keys()):
        try:
            entry = group[key]
        except Exception:
            continue
        if isinstance(entry, zarr.Array):
            level_entries.append((str(key), entry))

    if not level_entries:
        raise ValueError(f"No arrays found under zarr group: {path}")

    level_entries = _sorted_level_entries(level_entries)
    axes = _default_axes_for_ndim(level_entries[0][1].ndim)
    levels: List[Dict[str, Any]] = []
    for idx, (level_path, arr) in enumerate(level_entries):
        levels.append(
            {
                "index": idx,
                "path": level_path,
                "absolute_path": os.path.join(path, level_path) if level_path else path,
                "shape": [int(v) for v in arr.shape],
            }
        )

    return {
        "path": path,
        "source_kind": "regular",
        "axes": axes,
        "is_multiscale": len(levels) > 1,
        "levels": levels,
    }


def _inspect_ome_group(path: str, group: zarr.Group, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect OME-Zarr source using multiscales metadata dataset paths."""
    multiscales = attrs.get("multiscales", [])
    if not multiscales:
        raise ValueError(f"OME-Zarr source at {path} has empty 'multiscales' metadata.")
    primary = multiscales[0]
    datasets = primary.get("datasets", [])
    if not datasets:
        raise ValueError(f"OME-Zarr source at {path} has no multiscales datasets.")

    first_path = str(datasets[0].get("path", "")).strip()
    if not first_path:
        raise ValueError(f"OME-Zarr first dataset has empty path at {path}.")
    first_arr = group[first_path]
    if not isinstance(first_arr, zarr.Array):
        raise ValueError(f"OME-Zarr first dataset '{first_path}' is not an array.")

    axes = _normalize_axes_meta(primary.get("axes"), first_arr.ndim)
    levels: List[Dict[str, Any]] = []
    for idx, dataset in enumerate(datasets):
        dpath = str(dataset.get("path", "")).strip()
        if not dpath:
            continue
        arr = group[dpath]
        if not isinstance(arr, zarr.Array):
            continue
        levels.append(
            {
                "index": idx,
                "path": dpath,
                "absolute_path": os.path.join(path, dpath),
                "shape": [int(v) for v in arr.shape],
            }
        )

    return {
        "path": path,
        "source_kind": "ome",
        "axes": axes,
        "is_multiscale": len(levels) > 1,
        "levels": levels,
    }


def inspect_zarr_source(path: str) -> Dict[str, Any]:
    """Inspect zarr path and resolve axes + level metadata."""
    path = str(path).strip()
    if not path:
        raise ValueError("Zarr path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zarr path does not exist: {path}")

    obj = zarr.open(path, mode="r")
    if isinstance(obj, zarr.Array):
        return _inspect_regular_array(path, obj)
    if not isinstance(obj, zarr.Group):
        raise ValueError(f"Unsupported zarr object at {path}: {type(obj)}")

    attrs = dict(obj.attrs)
    if attrs.get("multiscales"):
        return _inspect_ome_group(path, obj, attrs)
    return _inspect_regular_group(path, obj)


def _extract_zyx(point: Sequence[Any], source_axes: Optional[Sequence[str]]) -> Tuple[float, float, float]:
    """Extract zyx coordinates from a row using optional explicit source axis labels."""
    values = list(point)
    if len(values) < 3:
        raise ValueError("Point must have at least 3 coordinates.")

    if source_axes is not None:
        axes = [str(axis).strip().lower() for axis in source_axes]
        axis_map = {axis_name: idx for idx, axis_name in enumerate(axes)}
        if {"z", "y", "x"}.issubset(axis_map):
            if len(values) == len(axes):
                return (
                    float(values[axis_map["z"]]),
                    float(values[axis_map["y"]]),
                    float(values[axis_map["x"]]),
                )
            if len(values) == 3:
                spatial_order = [name for name in axes if name in {"z", "y", "x"}]
                if len(spatial_order) == 3:
                    spatial_map = {axis_name: idx for idx, axis_name in enumerate(spatial_order)}
                    return (
                        float(values[spatial_map["z"]]),
                        float(values[spatial_map["y"]]),
                        float(values[spatial_map["x"]]),
                    )

    if len(values) >= 5:
        return float(values[2]), float(values[3]), float(values[4])
    return float(values[-3]), float(values[-2]), float(values[-1])


def normalize_points_to_zyx(points: Sequence[Sequence[Any]], source_axes: Optional[Sequence[str]] = None) -> List[List[float]]:
    """Normalize a points list to canonical [z,y,x] rows."""
    normalized: List[List[float]] = []
    for idx, point in enumerate(points):
        try:
            z, y, x = _extract_zyx(point, source_axes=source_axes)
        except Exception as exc:
            raise ValueError(
                f"Invalid point at index {idx}. Expected [z,y,x] or [t,c,z,y,x]. "
                f"source_axes={source_axes}"
            ) from exc
        normalized.append([z, y, x])
    return normalized


def _normalize_axes(axes: Optional[Sequence[Any]]) -> Optional[List[str]]:
    """Normalize axis labels to lower-case strings."""
    if not isinstance(axes, (list, tuple)):
        return None
    return [str(axis).strip().lower() for axis in axes]


def _all_rows_len(points: Sequence[Any], expected_len: int) -> bool:
    """Return True when every point row is a sequence of expected length."""
    if not points:
        return False
    for point in points:
        if not isinstance(point, (list, tuple)):
            return False
        if len(point) != expected_len:
            return False
    return True


def _candidate_source_paths(volume_path: str) -> List[str]:
    """Build source-inspection candidates from a provided volume path."""
    p = Path(str(volume_path).strip()).expanduser().resolve()
    candidates = [p, p.parent, p.parent.parent]
    seen = set()
    ordered: List[str] = []
    for item in candidates:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        if item.exists():
            ordered.append(key)
    return ordered


def _shape_zyx_for_source_meta(
    source_meta: Dict[str, Any],
    run_level: int,
    run_time_index: int,
    run_channel_index: int,
) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    """Resolve runtime tracker zyx shape for one source metadata candidate."""
    try:
        levels = source_meta.get("levels", [])
        if not levels:
            return None, "Source has no levels."
        if run_level < 0 or run_level >= len(levels):
            return None, f"run_level={run_level} out of range [0, {len(levels)-1}]"

        axes = [str(axis).strip().lower() for axis in source_meta.get("axes", [])]
        shape = [int(v) for v in levels[run_level].get("shape", [])]
        t_idx = _axis_index(axes, "t")
        c_idx = _axis_index(axes, "c")
        if t_idx is not None and t_idx < len(shape):
            if not (0 <= run_time_index < int(shape[t_idx])):
                return None, f"run_time_index={run_time_index} out of range [0, {int(shape[t_idx]) - 1}]"
        if c_idx is not None and c_idx < len(shape):
            if not (0 <= run_channel_index < int(shape[c_idx])):
                return None, f"run_channel_index={run_channel_index} out of range [0, {int(shape[c_idx]) - 1}]"

        shape_zyx = _shape_zyx_from_shape(shape, axes)
        if shape_zyx is None:
            return None, f"Could not resolve zyx shape from axes={axes}, shape={shape}"
        return shape_zyx, None
    except Exception as exc:
        return None, str(exc)


def _inspect_source_candidates(
    volume_path: str,
    run_level: int,
    run_time_index: int,
    run_channel_index: int,
) -> List[Dict[str, Any]]:
    """Inspect multiple nearby zarr paths and summarize candidate source metadata."""
    candidates = _candidate_source_paths(volume_path)
    reports: List[Dict[str, Any]] = []
    requested = str(Path(volume_path).expanduser().resolve())

    for path in candidates:
        report: Dict[str, Any] = {
            "path": path,
            "is_requested_path": path == requested,
            "ok": False,
        }
        try:
            source_meta = inspect_zarr_source(path)
            shape_zyx, runtime_error = _shape_zyx_for_source_meta(
                source_meta,
                run_level=run_level,
                run_time_index=run_time_index,
                run_channel_index=run_channel_index,
            )
            report.update(
                {
                    "ok": runtime_error is None,
                    "source_kind": source_meta.get("source_kind"),
                    "axes": [str(axis).strip().lower() for axis in source_meta.get("axes", [])],
                    "n_levels": len(source_meta.get("levels", [])),
                    "run_shape_zyx": list(shape_zyx) if shape_zyx is not None else None,
                    "runtime_error": runtime_error,
                    "source_meta": source_meta,
                }
            )
        except Exception as exc:
            report["runtime_error"] = str(exc)
        reports.append(report)
    return reports


def _iter_points_files(points_json: str, points_folder: str, max_files: int) -> List[Path]:
    """Return target JSON files to analyze."""
    if points_json:
        path = Path(points_json).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Points JSON not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file for --points-json: {path}")
        return [path]

    folder = Path(points_folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Points folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a directory for --points-folder: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if max_files > 0:
        files = files[:max_files]
    return files


def _load_points_payload(path: Path) -> Tuple[List[Any], Optional[List[str]], Dict[str, Any]]:
    """Load JSON payload and extract points + optional point_axes."""
    with path.open("r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        points = payload.get("points", [])
        point_axes = _normalize_axes(payload.get("point_axes"))
    else:
        points = payload
        point_axes = None
    if not isinstance(points, list):
        raise ValueError("JSON 'points' field is not a list.")
    return points, point_axes, payload if isinstance(payload, dict) else {}


def _point_dim_counts(points: Sequence[Any]) -> Dict[str, int]:
    """Count raw point dimensionalities."""
    counter: Counter = Counter()
    for point in points:
        if isinstance(point, (list, tuple)):
            counter[str(len(point))] += 1
        else:
            counter["invalid"] += 1
    return dict(counter)


def _in_bounds_ratio(points_zyx: np.ndarray, shape_zyx: Optional[Tuple[int, int, int]]) -> float:
    """Compute fraction of points inside zyx bounds."""
    if shape_zyx is None:
        return 0.0
    if points_zyx.size == 0:
        return 0.0

    zmax, ymax, xmax = [int(v) for v in shape_zyx]
    z = points_zyx[:, 0]
    y = points_zyx[:, 1]
    x = points_zyx[:, 2]
    mask = (z >= 0) & (z < zmax) & (y >= 0) & (y < ymax) & (x >= 0) & (x < xmax)
    return float(np.mean(mask))


def _bbox_zyx(points_zyx: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute zyx bounding box for normalized points."""
    if points_zyx.size == 0:
        return None
    return {
        "z_min": float(np.min(points_zyx[:, 0])),
        "z_max": float(np.max(points_zyx[:, 0])),
        "y_min": float(np.min(points_zyx[:, 1])),
        "y_max": float(np.max(points_zyx[:, 1])),
        "x_min": float(np.min(points_zyx[:, 2])),
        "x_max": float(np.max(points_zyx[:, 2])),
    }


def _step_stats(points_zyx: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute simple step-size statistics along the polyline."""
    if len(points_zyx) < 2:
        return None
    steps = np.linalg.norm(np.diff(points_zyx, axis=0), axis=1)
    return {
        "step_median": float(np.median(steps)),
        "step_p95": float(np.percentile(steps, 95)),
        "step_max": float(np.max(steps)),
    }


def _map_zyx_to_axes(points_zyx: np.ndarray, axes: Sequence[str]) -> np.ndarray:
    """Map canonical [z,y,x] rows to a requested output axis order."""
    axis_names = [str(axis).strip().lower() for axis in axes]
    if points_zyx.size == 0:
        return np.empty((0, len(axis_names)), dtype=float)
    out = np.zeros((len(points_zyx), len(axis_names)), dtype=float)
    z = points_zyx[:, 0]
    y = points_zyx[:, 1]
    x = points_zyx[:, 2]
    for idx, axis in enumerate(axis_names):
        if axis == "z":
            out[:, idx] = z
        elif axis == "y":
            out[:, idx] = y
        elif axis == "x":
            out[:, idx] = x
        else:
            out[:, idx] = 0.0
    return out


def _mae(points_a: np.ndarray, points_b: np.ndarray) -> Optional[float]:
    """Return mean absolute error between same-shape coordinate arrays."""
    if points_a.shape != points_b.shape:
        return None
    if points_a.size == 0:
        return 0.0
    return float(np.mean(np.abs(points_a - points_b)))


def _simulate_manual_save_risk(points_zyx: np.ndarray, output_axes: Sequence[str]) -> Dict[str, Any]:
    """
    Simulate napari load->display->save behavior.

    - `with_metadata`: points layer carries `tubulemap_point_axes=output_axes`.
    - `without_metadata`: manual layer without metadata (current fallback uses source_axes=None).
    """
    display_points = _map_zyx_to_axes(points_zyx, output_axes)
    saved_with_meta = np.asarray(
        normalize_points_to_zyx(display_points.tolist(), source_axes=list(output_axes)),
        dtype=float,
    )
    saved_without_meta = np.asarray(
        normalize_points_to_zyx(display_points.tolist(), source_axes=None),
        dtype=float,
    )
    return {
        "output_axes": [str(axis).strip().lower() for axis in output_axes],
        "with_metadata_mae_zyx": _mae(saved_with_meta, points_zyx),
        "without_metadata_mae_zyx": _mae(saved_without_meta, points_zyx),
    }


def _axis_assumptions_for_points(
    raw_points: Sequence[Any],
    declared_axes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build unique axis assumptions to test for a points payload."""
    assumptions: List[Dict[str, Any]] = []
    seen = set()

    def _push(name: str, axes: Optional[List[str]]) -> None:
        key = tuple(axes) if axes is not None else None
        if key in seen:
            return
        seen.add(key)
        assumptions.append({"name": name, "source_axes": axes})

    if declared_axes is not None:
        _push("declared_point_axes", declared_axes)

    _push("fallback_none", None)

    if _all_rows_len(raw_points, 3):
        for perm in AXIS_PERMUTATIONS_3D:
            _push(f"raw_is_{''.join(perm)}", list(perm))

    return assumptions


def _normalize_under_assumption(raw_points: Sequence[Any], source_axes: Optional[List[str]]) -> Tuple[np.ndarray, Optional[str]]:
    """Normalize raw points to zyx under a candidate source axis assumption."""
    try:
        points_zyx = normalize_points_to_zyx(raw_points, source_axes=source_axes)
        arr = np.asarray(points_zyx, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return np.empty((0, 3), dtype=float), "Normalization did not produce Nx3 points."
        return arr, None
    except Exception as exc:
        return np.empty((0, 3), dtype=float), str(exc)


def analyze_points_file_against_source(
    points_file: Path,
    source_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze one points JSON file against one source candidate."""
    raw_points, declared_axes, _payload = _load_points_payload(points_file)
    point_dims = _point_dim_counts(raw_points)
    assumptions = _axis_assumptions_for_points(raw_points, declared_axes)

    shape_zyx = tuple(source_report["run_shape_zyx"]) if source_report.get("run_shape_zyx") else None
    assumption_reports: List[Dict[str, Any]] = []
    for assumption in assumptions:
        source_axes = assumption["source_axes"]
        arr, error = _normalize_under_assumption(raw_points, source_axes)
        report = {
            "assumption": assumption["name"],
            "source_axes": source_axes,
            "ok": error is None,
            "error": error,
            "n_points": int(len(arr)),
            "in_bounds_ratio": _in_bounds_ratio(arr, shape_zyx) if error is None else 0.0,
            "bbox_zyx": _bbox_zyx(arr) if error is None else None,
            "step_stats": _step_stats(arr) if error is None else None,
        }
        if error is None:
            output_axes = source_report.get("axes", ["z", "y", "x"])
            gui_risk = _simulate_manual_save_risk(arr, output_axes)
            report["gui_roundtrip"] = gui_risk
        assumption_reports.append(report)

    assumption_reports.sort(key=lambda item: item.get("in_bounds_ratio", 0.0), reverse=True)
    best = assumption_reports[0] if assumption_reports else None

    declared_result = None
    for item in assumption_reports:
        if item["assumption"] == "declared_point_axes":
            declared_result = item
            break

    recommendations: List[str] = []
    if best is not None:
        if declared_result is None:
            recommendations.append(
                "Points JSON has no declared point_axes. Save explicit point_axes to avoid ambiguity."
            )
        else:
            delta = float(best["in_bounds_ratio"]) - float(declared_result["in_bounds_ratio"])
            if delta >= 0.3 and best["assumption"] != "declared_point_axes":
                recommendations.append(
                    "Declared point_axes looks inconsistent with this volume; best-fit assumption is "
                    f"{best['source_axes']} (in-bounds {best['in_bounds_ratio']:.1%} vs "
                    f"{declared_result['in_bounds_ratio']:.1%})."
                )

        if best.get("gui_roundtrip"):
            drift = best["gui_roundtrip"].get("without_metadata_mae_zyx")
            if drift is not None and drift > 1e-6:
                recommendations.append(
                    "Manual napari save without layer axis metadata can reorder coordinates for this "
                    f"axis configuration (expected drift MAE={drift:.3f} vox)."
                )
                recommendations.append(
                    "Best strategy: enforce strict point_axes in JSON and ensure points layer stores "
                    "explicit axis metadata before saving."
                )

    return {
        "points_file": str(points_file),
        "declared_point_axes": declared_axes,
        "point_dim_counts": point_dims,
        "n_raw_points": len(raw_points),
        "source_path": source_report.get("path"),
        "source_axes": source_report.get("axes"),
        "run_shape_zyx": source_report.get("run_shape_zyx"),
        "best_assumption": best,
        "declared_assumption_result": declared_result,
        "assumptions": assumption_reports,
        "recommendations": recommendations,
    }


def analyze(
    points_files: Sequence[Path],
    source_reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze all points files across all inspected source candidates."""
    usable_sources = [source for source in source_reports if source.get("ok")]
    if not usable_sources:
        raise RuntimeError("No usable source candidate was resolved from the provided volume path.")

    by_source: List[Dict[str, Any]] = []
    for source in usable_sources:
        file_reports = [analyze_points_file_against_source(path, source) for path in points_files]
        by_source.append(
            {
                "source": {
                    "path": source.get("path"),
                    "is_requested_path": source.get("is_requested_path"),
                    "source_kind": source.get("source_kind"),
                    "axes": source.get("axes"),
                    "n_levels": source.get("n_levels"),
                    "run_shape_zyx": source.get("run_shape_zyx"),
                },
                "files": file_reports,
            }
        )

    summary_flags: List[str] = []
    for source_block in by_source:
        src = source_block["source"]
        best_axes_counter: Counter = Counter()
        for file_report in source_block["files"]:
            best = file_report.get("best_assumption")
            if best and best.get("source_axes") is not None:
                best_axes_counter[str(best.get("source_axes"))] += 1
        if best_axes_counter:
            top_axes, votes = best_axes_counter.most_common(1)[0]
            summary_flags.append(
                f"Source {src['path']} (axes={src['axes']}) most often fits point order {top_axes} "
                f"({votes}/{len(source_block['files'])} files)."
            )

    return {
        "n_points_files": len(points_files),
        "source_candidates": [
            {
                "path": source.get("path"),
                "is_requested_path": source.get("is_requested_path"),
                "ok": source.get("ok"),
                "source_kind": source.get("source_kind"),
                "axes": source.get("axes"),
                "n_levels": source.get("n_levels"),
                "run_shape_zyx": source.get("run_shape_zyx"),
                "runtime_error": source.get("runtime_error"),
            }
            for source in source_reports
        ],
        "results_by_source": by_source,
        "summary_flags": summary_flags,
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print a concise human-readable report."""
    print("\n=== Source Candidates ===")
    for src in report["source_candidates"]:
        status = "OK" if src["ok"] else "ERROR"
        requested = " (requested path)" if src.get("is_requested_path") else ""
        print(
            f"- [{status}] {src['path']}{requested}\n"
            f"  kind={src.get('source_kind')} axes={src.get('axes')} "
            f"run_shape_zyx={src.get('run_shape_zyx')}"
        )
        if src.get("runtime_error"):
            print(f"  error={src['runtime_error']}")

    print("\n=== Per-Source File Results ===")
    for block in report["results_by_source"]:
        src = block["source"]
        print(
            f"\nSource: {src['path']} axes={src['axes']} run_shape_zyx={src['run_shape_zyx']}"
        )
        for item in block["files"]:
            print(f"  File: {item['points_file']}")
            print(
                f"    declared_point_axes={item.get('declared_point_axes')} "
                f"raw_dims={item.get('point_dim_counts')}"
            )
            best = item.get("best_assumption")
            if best:
                print(
                    "    best_assumption={name} source_axes={axes} in_bounds={ratio:.1%}".format(
                        name=best.get("assumption"),
                        axes=best.get("source_axes"),
                        ratio=float(best.get("in_bounds_ratio", 0.0)),
                    )
                )
                gui = best.get("gui_roundtrip") or {}
                if gui:
                    print(
                        "    gui_roundtrip_drift(with_metadata={wm:.6f}, without_metadata={wo:.6f})".format(
                            wm=float(gui.get("with_metadata_mae_zyx", 0.0)),
                            wo=float(gui.get("without_metadata_mae_zyx", 0.0)),
                        )
                    )
            for rec in item.get("recommendations", []):
                print(f"    recommendation: {rec}")

    print("\n=== Summary Flags ===")
    if report.get("summary_flags"):
        for flag in report["summary_flags"]:
            print(f"- {flag}")
    else:
        print("- No summary flags generated.")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose point axis order consistency between napari JSON and tracker volume ingest."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--points-json", default="", help="Path to one points JSON file.")
    group.add_argument("--points-folder", default="", help="Folder containing points JSON files.")
    parser.add_argument("--volume", required=True, help="Volume path used by tracking (array, group, or OME root).")
    parser.add_argument("--run-level", type=int, default=0, help="Run level index (default: 0).")
    parser.add_argument("--run-time-index", type=int, default=0, help="Run time index (default: 0).")
    parser.add_argument("--run-channel-index", type=int, default=0, help="Run channel index (default: 0).")
    parser.add_argument("--max-files", type=int, default=0, help="Max files when using --points-folder (0 = all).")
    parser.add_argument("--json-out", default="", help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> None:
    """Run CLI entrypoint."""
    args = _parse_args()
    points_files = _iter_points_files(args.points_json, args.points_folder, max_files=max(0, args.max_files))
    sources = _inspect_source_candidates(
        volume_path=args.volume,
        run_level=int(args.run_level),
        run_time_index=int(args.run_time_index),
        run_channel_index=int(args.run_channel_index),
    )
    report = analyze(points_files=points_files, source_reports=sources)
    print_report(report)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to: {out_path}")


if __name__ == "__main__":
    main()
