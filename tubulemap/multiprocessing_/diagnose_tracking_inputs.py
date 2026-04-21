#!/usr/bin/env python3
"""
Compare two tracking input pairs (keypoints folder + volume path) and flag likely
causes for immediate tracking drift/failure.

This script is standalone (no napari import) so it can be run directly on the
compute machine:

python /home/cfxuser/src/tubule-tracker/tubulemap/diagnose_tracking_inputs.py \
  --old-kp-folder /media/cfxuser/SSD2/Nephron_Tracking/GT/seedpoints \
  --old-volume /media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr/0 \
  --new-kp-folder /home/cfxuser/src/tubule-tracker/tubulemap/Demo/starting_points \
  --new-volume /media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/demo_data/oldeci_bbox_cropped.zarr/0 \
  --json-out /tmp/tracking_input_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr


SPATIAL_AXES = ("z", "y", "x")


def _default_axes_for_ndim(ndim: int) -> List[str]:
    if ndim == 5:
        return ["t", "c", "z", "y", "x"]
    if ndim == 4:
        return ["c", "z", "y", "x"]
    if ndim == 3:
        return ["z", "y", "x"]
    if ndim == 2:
        return ["y", "x"]
    return [f"dim_{i}" for i in range(ndim)]


def _normalize_axes(axes_meta: Any, ndim: int) -> List[str]:
    if axes_meta is None:
        return _default_axes_for_ndim(ndim)
    axes: List[str] = []
    for axis in axes_meta:
        if isinstance(axis, dict):
            name = str(axis.get("name", "")).strip().lower()
        else:
            name = str(axis).strip().lower()
        axes.append(name)
    if len(axes) != ndim:
        return _default_axes_for_ndim(ndim)
    return axes


def _axis_index(axes: Sequence[str], name: str) -> Optional[int]:
    name = str(name).lower()
    for idx, axis in enumerate(axes):
        if str(axis).lower() == name:
            return idx
    return None


def _shape_zyx(shape: Sequence[int], axes: Sequence[str]) -> Optional[Tuple[int, int, int]]:
    idx_z = _axis_index(axes, "z")
    idx_y = _axis_index(axes, "y")
    idx_x = _axis_index(axes, "x")
    if idx_z is None or idx_y is None or idx_x is None:
        return None
    return (int(shape[idx_z]), int(shape[idx_y]), int(shape[idx_x]))


def _sorted_level_entries(entries: Iterable[Tuple[str, zarr.Array]]) -> List[Tuple[str, zarr.Array]]:
    materialized = list(entries)
    numeric = []
    for name, arr in materialized:
        try:
            int(name)
            numeric.append((name, arr))
        except Exception:
            pass
    if len(numeric) == len(materialized):
        return sorted(materialized, key=lambda item: int(item[0]))
    return sorted(materialized, key=lambda item: item[0])


def inspect_volume(path: str) -> Dict[str, Any]:
    path = str(path).strip()
    if not path:
        raise ValueError("Volume path is empty.")
    if not Path(path).exists():
        raise FileNotFoundError(f"Volume path not found: {path}")

    obj = zarr.open(path, mode="r")
    levels: List[Dict[str, Any]] = []

    if isinstance(obj, zarr.Array):
        axes = _default_axes_for_ndim(obj.ndim)
        level = {
            "index": 0,
            "path": path,
            "shape": [int(v) for v in obj.shape],
            "shape_zyx": _shape_zyx(obj.shape, axes),
            "dtype": str(obj.dtype),
        }
        levels.append(level)
        return {
            "path": path,
            "kind": "array",
            "axes": axes,
            "is_multiscale": False,
            "levels": levels,
            "shape_zyx": level["shape_zyx"],
        }

    if not isinstance(obj, zarr.Group):
        raise ValueError(f"Unsupported zarr object at {path}: {type(obj)}")

    attrs = dict(obj.attrs)
    multiscales = attrs.get("multiscales", [])
    if multiscales:
        primary = multiscales[0]
        datasets = primary.get("datasets", [])
        if not datasets:
            raise ValueError(f"OME-Zarr metadata found but no datasets at {path}")

        first_path = str(datasets[0].get("path", "")).strip()
        if not first_path:
            raise ValueError(f"OME-Zarr first dataset has empty path at {path}")
        arr0 = obj[first_path]
        if not isinstance(arr0, zarr.Array):
            raise ValueError(f"OME-Zarr first dataset is not an array: {first_path}")
        axes = _normalize_axes(primary.get("axes"), arr0.ndim)

        for idx, dataset in enumerate(datasets):
            dpath = str(dataset.get("path", "")).strip()
            if not dpath:
                continue
            arr = obj[dpath]
            if not isinstance(arr, zarr.Array):
                continue
            levels.append(
                {
                    "index": idx,
                    "path": dpath,
                    "shape": [int(v) for v in arr.shape],
                    "shape_zyx": _shape_zyx(arr.shape, axes),
                    "dtype": str(arr.dtype),
                }
            )
        return {
            "path": path,
            "kind": "ome",
            "axes": axes,
            "is_multiscale": len(levels) > 1,
            "levels": levels,
            "shape_zyx": levels[0]["shape_zyx"] if levels else None,
        }

    level_entries: List[Tuple[str, zarr.Array]] = []
    for key in list(obj.keys()):
        try:
            entry = obj[key]
        except Exception:
            continue
        if isinstance(entry, zarr.Array):
            level_entries.append((str(key), entry))

    if not level_entries:
        raise ValueError(f"No arrays found under zarr group: {path}")

    level_entries = _sorted_level_entries(level_entries)
    axes = _default_axes_for_ndim(level_entries[0][1].ndim)
    for idx, (level_path, arr) in enumerate(level_entries):
        levels.append(
            {
                "index": idx,
                "path": level_path,
                "shape": [int(v) for v in arr.shape],
                "shape_zyx": _shape_zyx(arr.shape, axes),
                "dtype": str(arr.dtype),
            }
        )

    return {
        "path": path,
        "kind": "regular_group",
        "axes": axes,
        "is_multiscale": len(levels) > 1,
        "levels": levels,
        "shape_zyx": levels[0]["shape_zyx"] if levels else None,
    }


def _extract_zyx(point: Sequence[Any], source_axes: Optional[Sequence[str]]) -> Tuple[float, float, float]:
    values = list(point)
    if len(values) < 3:
        raise ValueError("point has fewer than 3 coordinates")

    if source_axes is not None:
        axis_names = [str(a).strip().lower() for a in source_axes]
        axis_map = {name: idx for idx, name in enumerate(axis_names)}
        if {"z", "y", "x"}.issubset(axis_map):
            if len(values) == len(axis_names):
                return (
                    float(values[axis_map["z"]]),
                    float(values[axis_map["y"]]),
                    float(values[axis_map["x"]]),
                )
            if len(values) == 3:
                spatial = [a for a in axis_names if a in {"z", "y", "x"}]
                if len(spatial) == 3:
                    spatial_map = {name: idx for idx, name in enumerate(spatial)}
                    return (
                        float(values[spatial_map["z"]]),
                        float(values[spatial_map["y"]]),
                        float(values[spatial_map["x"]]),
                    )

    if len(values) >= 5:
        return float(values[2]), float(values[3]), float(values[4])
    return float(values[-3]), float(values[-2]), float(values[-1])


def _in_bounds_zyx(point_zyx: Sequence[float], shape_zyx: Optional[Tuple[int, int, int]]) -> bool:
    if shape_zyx is None:
        return False
    z, y, x = point_zyx
    zmax, ymax, xmax = shape_zyx
    return (0 <= z < zmax) and (0 <= y < ymax) and (0 <= x < xmax)


def _score_permutations_for_3d_points(points: np.ndarray, shape_zyx: Optional[Tuple[int, int, int]]) -> Dict[str, float]:
    # Interpret raw 3D point columns in different orders and score in-bounds ratio.
    perms = {
        "zyx": (0, 1, 2),
        "zxy": (0, 2, 1),
        "yzx": (1, 0, 2),
        "yxz": (1, 2, 0),
        "xzy": (2, 0, 1),
        "xyz": (2, 1, 0),
    }
    scores: Dict[str, float] = {}
    if points.size == 0 or shape_zyx is None:
        return scores
    for name, idxs in perms.items():
        interpreted = points[:, idxs]
        ok = sum(_in_bounds_zyx(row, shape_zyx) for row in interpreted)
        scores[name] = ok / float(len(interpreted))
    return scores


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def analyze_points_file(
    file_path: Path,
    shape_zyx: Optional[Tuple[int, int, int]],
    other_shape_zyx: Optional[Tuple[int, int, int]],
) -> Dict[str, Any]:
    with file_path.open("r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_points = payload.get("points", [])
        point_axes = payload.get("point_axes")
    else:
        raw_points = payload
        point_axes = None

    if point_axes is not None:
        point_axes = [str(a).strip().lower() for a in point_axes]

    dim_counter: Counter = Counter()
    normalized: List[List[float]] = []
    parse_errors = 0
    raw_3d_rows: List[List[float]] = []

    for point in raw_points:
        if not isinstance(point, (list, tuple)):
            parse_errors += 1
            continue
        dim_counter[len(point)] += 1
        if len(point) == 3:
            try:
                raw_3d_rows.append([float(point[0]), float(point[1]), float(point[2])])
            except Exception:
                pass
        try:
            z, y, x = _extract_zyx(point, point_axes)
            normalized.append([z, y, x])
        except Exception:
            parse_errors += 1

    arr = np.asarray(normalized, dtype=float) if normalized else np.empty((0, 3), dtype=float)
    in_bounds = [_in_bounds_zyx(row, shape_zyx) for row in normalized]
    in_bounds_ratio = (sum(in_bounds) / len(in_bounds)) if in_bounds else 0.0

    in_other = [_in_bounds_zyx(row, other_shape_zyx) for row in normalized]
    in_other_ratio = (sum(in_other) / len(in_other)) if in_other else 0.0

    bbox = None
    step_stats = None
    if len(arr) > 0:
        bbox = {
            "z_min": float(np.min(arr[:, 0])),
            "z_max": float(np.max(arr[:, 0])),
            "y_min": float(np.min(arr[:, 1])),
            "y_max": float(np.max(arr[:, 1])),
            "x_min": float(np.min(arr[:, 2])),
            "x_max": float(np.max(arr[:, 2])),
        }
    if len(arr) > 1:
        steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        step_stats = {
            "step_median": float(np.median(steps)),
            "step_p95": float(np.percentile(steps, 95)),
            "step_max": float(np.max(steps)),
        }

    perm_scores = {}
    best_perm = None
    if point_axes is None and raw_3d_rows:
        perm_scores = _score_permutations_for_3d_points(np.asarray(raw_3d_rows, dtype=float), shape_zyx)
        if perm_scores:
            best_perm = max(perm_scores, key=perm_scores.get)

    warnings: List[str] = []
    if len(normalized) == 0:
        warnings.append("No valid points parsed from this file.")
    if len(normalized) > 0 and in_bounds_ratio < 0.5:
        warnings.append(f"Only {in_bounds_ratio:.1%} points are inside this volume bounds.")
    if len(normalized) > 0 and (in_other_ratio - in_bounds_ratio) > 0.5:
        warnings.append(
            "Points fit the OTHER volume much better than this one; likely coordinate frame mismatch."
        )
    if best_perm is not None and best_perm != "zyx":
        best_score = perm_scores.get(best_perm, 0.0)
        zyx_score = perm_scores.get("zyx", 0.0)
        if best_score - zyx_score > 0.3:
            warnings.append(
                f"3D points look closer to '{best_perm}' ordering than 'zyx' "
                f"({best_score:.1%} vs {zyx_score:.1%} in-bounds)."
            )

    return {
        "file": str(file_path),
        "point_axes": point_axes,
        "n_points_raw": len(raw_points),
        "n_points_parsed": len(normalized),
        "parse_errors": int(parse_errors),
        "point_dim_counts": dict(dim_counter),
        "in_bounds_ratio": float(in_bounds_ratio),
        "in_other_ratio": float(in_other_ratio),
        "bbox_zyx": bbox,
        "step_stats": step_stats,
        "best_perm_3d": best_perm,
        "perm_scores_3d": perm_scores,
        "warnings": warnings,
    }


def analyze_folder(
    kp_folder: str,
    shape_zyx: Optional[Tuple[int, int, int]],
    other_shape_zyx: Optional[Tuple[int, int, int]],
    max_files: int,
) -> Dict[str, Any]:
    folder = Path(kp_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Keypoint folder not found: {kp_folder}")
    if not folder.is_dir():
        raise ValueError(f"Keypoint folder is not a directory: {kp_folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if max_files > 0:
        files = files[:max_files]

    file_reports: List[Dict[str, Any]] = []
    for path in files:
        try:
            report = analyze_points_file(path, shape_zyx, other_shape_zyx)
        except Exception as exc:
            report = {
                "file": str(path),
                "error": str(exc),
                "warnings": [f"Failed to parse JSON: {exc}"],
            }
        file_reports.append(report)

    in_bounds_values = [r["in_bounds_ratio"] for r in file_reports if "in_bounds_ratio" in r]
    problematic = []
    for r in file_reports:
        if "in_bounds_ratio" in r and r["in_bounds_ratio"] < 0.5:
            problematic.append({"file": r["file"], "in_bounds_ratio": r["in_bounds_ratio"]})
        elif r.get("warnings"):
            problematic.append({"file": r["file"], "warning": r["warnings"][0]})

    best_perm_counter: Counter = Counter(
        r["best_perm_3d"] for r in file_reports if r.get("best_perm_3d") is not None
    )

    return {
        "kp_folder": str(folder),
        "n_json_files": len(files),
        "median_in_bounds_ratio": _safe_median(in_bounds_values),
        "min_in_bounds_ratio": (min(in_bounds_values) if in_bounds_values else None),
        "best_perm_counter": dict(best_perm_counter),
        "problematic_examples": problematic[:10],
        "files": file_reports,
    }


def compare_cases(
    old_kp_folder: str,
    old_volume: str,
    new_kp_folder: str,
    new_volume: str,
    max_files: int,
) -> Dict[str, Any]:
    old_vol = inspect_volume(old_volume)
    new_vol = inspect_volume(new_volume)

    old_shape = tuple(old_vol["shape_zyx"]) if old_vol.get("shape_zyx") else None
    new_shape = tuple(new_vol["shape_zyx"]) if new_vol.get("shape_zyx") else None

    old_folder = analyze_folder(old_kp_folder, old_shape, new_shape, max_files=max_files)
    new_folder = analyze_folder(new_kp_folder, new_shape, old_shape, max_files=max_files)

    high_level_flags: List[str] = []
    old_med = old_folder.get("median_in_bounds_ratio")
    new_med = new_folder.get("median_in_bounds_ratio")
    if old_med is not None and new_med is not None and (old_med - new_med) > 0.25:
        high_level_flags.append(
            f"New keypoints fit their volume much worse than old set "
            f"(median in-bounds {new_med:.1%} vs {old_med:.1%})."
        )

    new_perm_counter = new_folder.get("best_perm_counter", {})
    non_zyx_votes = sum(v for k, v in new_perm_counter.items() if k != "zyx")
    zyx_votes = new_perm_counter.get("zyx", 0)
    if non_zyx_votes > zyx_votes:
        high_level_flags.append(
            "New keypoints frequently look better under non-'zyx' axis order; possible axis swap."
        )

    old_shape_list = list(old_shape) if old_shape is not None else None
    new_shape_list = list(new_shape) if new_shape is not None else None

    return {
        "old": {"volume": old_vol, "keypoints": old_folder},
        "new": {"volume": new_vol, "keypoints": new_folder},
        "high_level_flags": high_level_flags,
        "shape_zyx": {"old": old_shape_list, "new": new_shape_list},
    }


def print_report(report: Dict[str, Any]) -> None:
    def _fmt_ratio(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    old = report["old"]
    new = report["new"]
    print("\n=== Volume Comparison ===")
    print(f"OLD volume: {old['volume']['path']}")
    print(f"  kind={old['volume']['kind']} axes={old['volume']['axes']} shape_zyx={old['volume']['shape_zyx']}")
    print(f"NEW volume: {new['volume']['path']}")
    print(f"  kind={new['volume']['kind']} axes={new['volume']['axes']} shape_zyx={new['volume']['shape_zyx']}")

    print("\n=== Keypoint Folder Summary ===")
    print(f"OLD kp folder: {old['keypoints']['kp_folder']}")
    print(
        "  files={files} median_in_bounds={med} min_in_bounds={minv} perm_votes={perms}".format(
            files=old["keypoints"]["n_json_files"],
            med=_fmt_ratio(old["keypoints"]["median_in_bounds_ratio"]),
            minv=_fmt_ratio(old["keypoints"]["min_in_bounds_ratio"]),
            perms=old["keypoints"]["best_perm_counter"],
        )
    )
    print(f"NEW kp folder: {new['keypoints']['kp_folder']}")
    print(
        "  files={files} median_in_bounds={med} min_in_bounds={minv} perm_votes={perms}".format(
            files=new["keypoints"]["n_json_files"],
            med=_fmt_ratio(new["keypoints"]["median_in_bounds_ratio"]),
            minv=_fmt_ratio(new["keypoints"]["min_in_bounds_ratio"]),
            perms=new["keypoints"]["best_perm_counter"],
        )
    )

    print("\n=== Potential Issues ===")
    flags = report.get("high_level_flags", [])
    if not flags:
        print("No high-level mismatch flag was triggered.")
    else:
        for flag in flags:
            print(f"- {flag}")

    examples = new["keypoints"]["problematic_examples"]
    if examples:
        print("\nMost problematic NEW files:")
        for item in examples[:10]:
            if "in_bounds_ratio" in item:
                print(f"- {item['file']}: in-bounds={item['in_bounds_ratio']:.1%}")
            else:
                print(f"- {item['file']}: {item.get('warning', 'issue')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare working vs failing tracking inputs and diagnose likely mismatches."
    )
    parser.add_argument("--old-kp-folder", required=True, help="Working keypoints folder.")
    parser.add_argument("--old-volume", required=True, help="Working volume path.")
    parser.add_argument("--new-kp-folder", required=True, help="Failing/new keypoints folder.")
    parser.add_argument("--new-volume", required=True, help="Failing/new volume path.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of JSON files per folder (0 means all).",
    )
    parser.add_argument("--json-out", default="", help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compare_cases(
        old_kp_folder=args.old_kp_folder,
        old_volume=args.old_volume,
        new_kp_folder=args.new_kp_folder,
        new_volume=args.new_volume,
        max_files=max(0, int(args.max_files)),
    )
    print_report(report)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to: {out_path}")


if __name__ == "__main__":
    main()

