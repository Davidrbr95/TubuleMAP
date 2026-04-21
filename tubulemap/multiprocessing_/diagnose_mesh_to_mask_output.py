#!/usr/bin/env python3
"""
Diagnose mesh->mask outputs for emptiness, shape mismatches, and axis/bounds issues.

Example:
python /path/to/tubulemap/diagnose_mesh_to_mask_output.py \
  --volume /path/to/volume.zarr/0 \
  --mask /path/to/output_mesh_mask.zarr \
  --obj-folder /path/to/mesh_folder \
  --json-out /tmp/mesh_mask_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

PERM_LABELS = {
    (0, 1, 2): "xyz",
    (0, 2, 1): "xzy",
    (1, 0, 2): "yxz",
    (1, 2, 0): "yzx",
    (2, 0, 1): "zxy",
    (2, 1, 0): "zyx",
}


def _safe_int_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def collect_obj_paths(obj_specs: Sequence[str], max_labels: Optional[int] = None) -> List[str]:
    """Collect OBJ paths from directories and glob expressions (non-recursive for dirs)."""
    import glob
    import os

    obj_paths: List[str] = []
    for spec in obj_specs:
        spec_str = str(spec)
        if os.path.isdir(spec_str):
            obj_paths.extend(glob.glob(os.path.join(spec_str, "*.obj")))
        else:
            obj_paths.extend(glob.glob(spec_str))

    unique = sorted(set(obj_paths))
    if max_labels is not None:
        unique = unique[: max(0, int(max_labels))]
    return unique


def collect_points_json_paths(
    points_json: Optional[Sequence[str]] = None,
    points_folder: Optional[str] = None,
    points_glob: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    """Collect points JSON paths from explicit paths, folder, and glob expressions."""
    import glob
    import os

    paths: List[str] = []

    if points_json:
        for p in points_json:
            if p:
                paths.append(str(Path(p).expanduser().resolve()))

    if points_folder:
        folder = str(Path(points_folder).expanduser().resolve())
        if os.path.isdir(folder):
            paths.extend(glob.glob(os.path.join(folder, "*.json")))

    if points_glob:
        for spec in points_glob:
            paths.extend(glob.glob(str(spec)))

    unique = sorted(set(paths))
    if max_files is not None:
        unique = unique[: max(0, int(max_files))]
    return unique


def load_mesh_vertices(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices/faces without transforms (already expected in XYZ index space)."""
    import trimesh

    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    return vertices, faces


def resolve_reference_shape_xyz(reference_path: str, array_name: Optional[str] = None) -> Tuple[int, int, int]:
    """Resolve target XYZ shape from a zarr array or group path."""
    ref = zarr.open(reference_path, mode="r")
    if isinstance(ref, zarr.Array):
        arr_ref = ref
    else:
        if array_name and array_name in ref:
            arr_ref = ref[array_name]
        else:
            arr_ref = next((v for v in ref.values() if isinstance(v, zarr.Array)), None)
            if arr_ref is None:
                raise RuntimeError(f"No zarr array found in reference path: {reference_path}")

    shape = arr_ref.shape[-3:] if arr_ref.ndim >= 3 else arr_ref.shape
    if len(shape) != 3:
        raise RuntimeError(f"Reference shape must be 3D. Got shape: {arr_ref.shape}")
    return tuple(int(v) for v in shape)


def inspect_zarr_source(path: str) -> Dict[str, object]:
    """Lightweight zarr source inspection (standalone fallback)."""
    p = str(Path(path).expanduser().resolve())
    obj = zarr.open(p, mode="r")
    meta: Dict[str, object] = {"path": p}
    if isinstance(obj, zarr.Array):
        meta.update(
            {
                "kind": "array",
                "shape": [int(v) for v in obj.shape],
                "chunks": [int(v) for v in obj.chunks],
                "dtype": str(obj.dtype),
            }
        )
    else:
        arrays = [(k, v) for k, v in obj.items() if isinstance(v, zarr.Array)]
        meta.update(
            {
                "kind": "group",
                "array_keys": [k for k, _ in arrays],
            }
        )
        if arrays:
            key, arr = arrays[0]
            meta["first_array"] = key
            meta["first_array_shape"] = [int(v) for v in arr.shape]
            meta["first_array_chunks"] = [int(v) for v in arr.chunks]
            meta["first_array_dtype"] = str(arr.dtype)
    return meta


def _slice_for_chunk(chunk_idx: Tuple[int, ...], chunks: Tuple[int, ...], shape: Tuple[int, ...]):
    slices = []
    starts = []
    for axis, idx in enumerate(chunk_idx):
        start = idx * chunks[axis]
        stop = min(start + chunks[axis], shape[axis])
        starts.append(start)
        slices.append(slice(start, stop))
    return tuple(slices), tuple(starts)


def _iter_chunk_indices(shape: Tuple[int, ...], chunks: Tuple[int, ...]):
    grid = [int(math.ceil(shape[i] / chunks[i])) for i in range(len(shape))]
    if len(grid) != 3:
        raise ValueError("Expected 3D array for chunk iteration.")
    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                yield (i, j, k), tuple(grid)


def _scan_mask_array(
    arr: zarr.Array,
    scan_mode: str,
    max_chunks: int,
    max_nonzero_samples: int = 12,
    max_unique_labels: int = 512,
) -> Dict[str, object]:
    shape = _safe_int_tuple(arr.shape)
    chunks = _safe_int_tuple(arr.chunks)

    total_chunks = int(math.ceil(shape[0] / chunks[0]) * math.ceil(shape[1] / chunks[1]) * math.ceil(shape[2] / chunks[2]))

    if scan_mode == "full":
        effective_mode = "full"
        chunks_to_scan = total_chunks
    elif scan_mode == "sample":
        effective_mode = "sample"
        chunks_to_scan = min(total_chunks, max_chunks)
    else:  # auto
        if total_chunks <= max_chunks:
            effective_mode = "full"
            chunks_to_scan = total_chunks
        else:
            effective_mode = "sample"
            chunks_to_scan = max_chunks

    scanned_chunks = 0
    nonzero_chunks = 0
    nonzero_voxels_scanned = 0
    global_min = None
    global_max = None
    nonzero_examples: List[Dict[str, object]] = []
    unique_labels = set()

    for chunk_idx, _ in _iter_chunk_indices(shape, chunks):
        if scanned_chunks >= chunks_to_scan:
            break
        slices, starts = _slice_for_chunk(chunk_idx, chunks, shape)
        data = np.asarray(arr[slices])
        scanned_chunks += 1

        nz = int(np.count_nonzero(data))
        if nz == 0:
            continue

        nonzero_chunks += 1
        nonzero_voxels_scanned += nz

        coords = np.argwhere(data > 0)
        local_min = coords.min(axis=0)
        local_max = coords.max(axis=0)
        chunk_global_min = [int(starts[a] + local_min[a]) for a in range(3)]
        chunk_global_max = [int(starts[a] + local_max[a]) for a in range(3)]

        if global_min is None:
            global_min = chunk_global_min
            global_max = chunk_global_max
        else:
            global_min = [min(global_min[a], chunk_global_min[a]) for a in range(3)]
            global_max = [max(global_max[a], chunk_global_max[a]) for a in range(3)]

        if len(nonzero_examples) < max_nonzero_samples:
            nonzero_examples.append(
                {
                    "chunk_index": list(chunk_idx),
                    "nonzero_voxels": nz,
                    "global_min_xyz": chunk_global_min,
                    "global_max_xyz": chunk_global_max,
                }
            )

        if len(unique_labels) < max_unique_labels:
            labels = np.unique(data[data > 0])
            for label in labels.tolist():
                unique_labels.add(int(label))
                if len(unique_labels) >= max_unique_labels:
                    break

    return {
        "shape_xyz": list(shape),
        "chunks_xyz": list(chunks),
        "total_chunks": total_chunks,
        "scanned_chunks": scanned_chunks,
        "scan_mode_requested": scan_mode,
        "scan_mode_effective": effective_mode,
        "nonzero_chunks": nonzero_chunks,
        "nonzero_voxels_scanned": int(nonzero_voxels_scanned),
        "nonzero_bbox_xyz": None
        if global_min is None
        else {
            "x_min": int(global_min[0]),
            "x_max": int(global_max[0]),
            "y_min": int(global_min[1]),
            "y_max": int(global_max[1]),
            "z_min": int(global_min[2]),
            "z_max": int(global_max[2]),
        },
        "nonzero_examples": nonzero_examples,
        "unique_labels_sample": sorted(unique_labels),
        "is_empty_in_scanned_region": nonzero_chunks == 0,
    }


def _score_perm_xyz(vertices_xyz: np.ndarray, perm: Tuple[int, int, int], shape_xyz: Tuple[int, int, int]) -> float:
    if vertices_xyz.ndim != 2 or vertices_xyz.shape[1] < 3 or len(vertices_xyz) == 0:
        return 0.0
    v = vertices_xyz[:, list(perm)]
    inside = (
        (v[:, 0] >= 0.0)
        & (v[:, 0] < float(shape_xyz[0]))
        & (v[:, 1] >= 0.0)
        & (v[:, 1] < float(shape_xyz[1]))
        & (v[:, 2] >= 0.0)
        & (v[:, 2] < float(shape_xyz[2]))
    )
    return float(np.mean(inside))


def _in_bounds_ratio_xyz(points_xyz: np.ndarray, shape_xyz: Tuple[int, int, int]) -> float:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3 or len(points_xyz) == 0:
        return 0.0
    inside = (
        (points_xyz[:, 0] >= 0.0)
        & (points_xyz[:, 0] < float(shape_xyz[0]))
        & (points_xyz[:, 1] >= 0.0)
        & (points_xyz[:, 1] < float(shape_xyz[1]))
        & (points_xyz[:, 2] >= 0.0)
        & (points_xyz[:, 2] < float(shape_xyz[2]))
    )
    return float(np.mean(inside))


def _bbox_from_points_xyz(points_xyz: np.ndarray) -> Optional[Dict[str, float]]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3 or len(points_xyz) == 0:
        return None
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return {
        "x_min": float(mins[0]),
        "x_max": float(maxs[0]),
        "y_min": float(mins[1]),
        "y_max": float(maxs[1]),
        "z_min": float(mins[2]),
        "z_max": float(maxs[2]),
    }


def _bbox_overlap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[bool]:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    return not (
        a["x_max"] < b["x_min"]
        or b["x_max"] < a["x_min"]
        or a["y_max"] < b["y_min"]
        or b["y_max"] < a["y_min"]
        or a["z_max"] < b["z_min"]
        or b["z_max"] < a["z_min"]
    )


def _load_points_json(points_json_path: str) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, object]]:
    payload = json.loads(Path(points_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Points JSON must be an object: {points_json_path}")

    if "points" not in payload:
        raise RuntimeError(f"Points JSON is missing required key 'points': {points_json_path}")

    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list):
        raise RuntimeError(f"Points JSON 'points' must be a list: {points_json_path}")

    if len(raw_points) == 0:
        points_arr = np.zeros((0, 3), dtype=float)
    else:
        points_arr = np.asarray(raw_points, dtype=float)
        if points_arr.ndim != 2 or points_arr.shape[1] < 3:
            raise RuntimeError(
                f"Points array must be NxM with M>=3, got {points_arr.shape} in {points_json_path}"
            )

    point_axes = payload.get("point_axes", None)
    if isinstance(point_axes, list):
        axes = [str(a).strip().lower() for a in point_axes]
    else:
        axes = None
    return points_arr, axes, payload


def _points_to_xyz_with_axes(points_arr: np.ndarray, axes: Sequence[str]) -> Optional[np.ndarray]:
    axes_lower = [str(a).strip().lower() for a in axes]
    if not {"x", "y", "z"}.issubset(set(axes_lower)):
        return None
    x_idx = axes_lower.index("x")
    y_idx = axes_lower.index("y")
    z_idx = axes_lower.index("z")
    if max(x_idx, y_idx, z_idx) >= points_arr.shape[1]:
        return None
    return points_arr[:, [x_idx, y_idx, z_idx]]


def _sample_point_on_mask_ratio(mask_arr: zarr.Array, points_xyz: np.ndarray, shape_xyz: Tuple[int, int, int]) -> Dict[str, object]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3 or len(points_xyz) == 0:
        return {
            "in_bounds_count": 0,
            "on_mask_count": 0,
            "on_mask_ratio": 0.0,
            "sampled_points": 0,
        }

    rounded = np.rint(points_xyz).astype(int)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < shape_xyz[0])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < shape_xyz[1])
        & (rounded[:, 2] >= 0)
        & (rounded[:, 2] < shape_xyz[2])
    )
    idx = np.where(in_bounds)[0]
    if len(idx) == 0:
        return {
            "in_bounds_count": 0,
            "on_mask_count": 0,
            "on_mask_ratio": 0.0,
            "sampled_points": 0,
        }

    max_points = 5000
    if len(idx) > max_points:
        take = np.linspace(0, len(idx) - 1, max_points).astype(int)
        idx = idx[take]

    on_mask_count = 0
    for i in idx.tolist():
        x, y, z = rounded[i]
        if int(mask_arr[x, y, z]) > 0:
            on_mask_count += 1

    return {
        "in_bounds_count": int(len(np.where(in_bounds)[0])),
        "on_mask_count": int(on_mask_count),
        "on_mask_ratio": float(on_mask_count / max(len(idx), 1)),
        "sampled_points": int(len(idx)),
    }


def _diagnose_points_overlap(
    points_json_path: str,
    shape_xyz: Tuple[int, int, int],
    obj_bbox: Optional[Dict[str, float]],
    mask_bbox: Optional[Dict[str, float]],
    mask_arr: zarr.Array,
) -> Dict[str, object]:
    points_arr, declared_axes, _ = _load_points_json(points_json_path)

    assumptions = []

    # Declared point_axes interpretation (if available)
    if declared_axes is not None:
        declared_xyz = _points_to_xyz_with_axes(points_arr, declared_axes)
        if declared_xyz is not None:
            assumptions.append(("declared_point_axes", declared_xyz))

    # Raw permutations from first 3 columns
    base = points_arr[:, :3] if points_arr.shape[1] >= 3 else points_arr
    for perm in permutations((0, 1, 2)):
        assumptions.append((f"raw_first3_{PERM_LABELS[perm]}", base[:, list(perm)]))

    # Also consider last 3 dims when points include extra dimensions.
    if points_arr.shape[1] > 3:
        last3 = points_arr[:, -3:]
        for perm in permutations((0, 1, 2)):
            assumptions.append((f"raw_last3_{PERM_LABELS[perm]}", last3[:, list(perm)]))

    evaluated = []
    for name, pts_xyz in assumptions:
        in_bounds = _in_bounds_ratio_xyz(pts_xyz, shape_xyz)
        bbox = _bbox_from_points_xyz(pts_xyz)
        mask_hit = _sample_point_on_mask_ratio(mask_arr, pts_xyz, shape_xyz)
        evaluated.append(
            {
                "assumption": name,
                "in_bounds_ratio": float(in_bounds),
                "bbox_xyz": bbox,
                "bbox_overlap_obj": _bbox_overlap(bbox, obj_bbox),
                "bbox_overlap_mask": _bbox_overlap(bbox, mask_bbox),
                "point_on_mask": mask_hit,
            }
        )

    evaluated.sort(
        key=lambda row: (
            float(row.get("in_bounds_ratio", 0.0)),
            float(row.get("point_on_mask", {}).get("on_mask_ratio", 0.0)),
        ),
        reverse=True,
    )
    best = evaluated[0] if evaluated else None

    return {
        "points_json": str(Path(points_json_path).expanduser().resolve()),
        "declared_point_axes": declared_axes,
        "raw_points_shape": list(points_arr.shape),
        "best_assumption": best,
        "assumptions": evaluated,
    }


def _clip_bbox_xyz(
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
    shape_xyz: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int]:
    X, Y, Z = shape_xyz
    return (
        max(0, min(x0, X)),
        max(0, min(x1, X)),
        max(0, min(y0, Y)),
        max(0, min(y1, Y)),
        max(0, min(z0, Z)),
        max(0, min(z1, Z)),
    )


def _diagnose_obj_bounds(
    obj_paths: Sequence[str],
    shape_xyz: Tuple[int, int, int],
    max_objs: int,
    max_vertices_for_perm_scoring: int = 20000,
) -> Dict[str, object]:
    sampled = list(obj_paths)[: max(0, max_objs)]
    per_obj = []
    in_bounds_count = 0
    outside_count = 0
    best_perm_counter: Dict[str, int] = {}

    union_raw_min = None
    union_raw_max = None
    union_clipped_min = None
    union_clipped_max = None

    for path in sampled:
        vertices, faces = load_mesh_vertices(path)
        if len(vertices) == 0 or len(faces) == 0:
            per_obj.append(
                {
                    "path": str(path),
                    "is_empty_mesh": True,
                    "in_bounds": False,
                }
            )
            outside_count += 1
            continue

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)

        x0 = int(math.floor(float(mins[0])))
        x1 = int(math.ceil(float(maxs[0])))
        y0 = int(math.floor(float(mins[1])))
        y1 = int(math.ceil(float(maxs[1])))
        z0 = int(math.floor(float(mins[2])))
        z1 = int(math.ceil(float(maxs[2])))

        cx0, cx1, cy0, cy1, cz0, cz1 = _clip_bbox_xyz(x0, x1, y0, y1, z0, z1, shape_xyz)
        in_bounds = (cx1 > cx0) and (cy1 > cy0) and (cz1 > cz0)

        if in_bounds:
            in_bounds_count += 1
        else:
            outside_count += 1

        if union_raw_min is None:
            union_raw_min = [x0, y0, z0]
            union_raw_max = [x1, y1, z1]
            union_clipped_min = [cx0, cy0, cz0]
            union_clipped_max = [cx1, cy1, cz1]
        else:
            union_raw_min = [min(union_raw_min[0], x0), min(union_raw_min[1], y0), min(union_raw_min[2], z0)]
            union_raw_max = [max(union_raw_max[0], x1), max(union_raw_max[1], y1), max(union_raw_max[2], z1)]
            union_clipped_min = [
                min(union_clipped_min[0], cx0),
                min(union_clipped_min[1], cy0),
                min(union_clipped_min[2], cz0),
            ]
            union_clipped_max = [
                max(union_clipped_max[0], cx1),
                max(union_clipped_max[1], cy1),
                max(union_clipped_max[2], cz1),
            ]

        if len(vertices) > max_vertices_for_perm_scoring:
            idx = np.linspace(0, len(vertices) - 1, max_vertices_for_perm_scoring).astype(int)
            score_vertices = vertices[idx]
        else:
            score_vertices = vertices

        perm_scores = {}
        best_perm = None
        best_score = -1.0
        for perm in permutations((0, 1, 2)):
            score = _score_perm_xyz(score_vertices, perm, shape_xyz)
            label = PERM_LABELS[perm]
            perm_scores[label] = score
            if score > best_score:
                best_score = score
                best_perm = label

        best_perm_counter[best_perm] = best_perm_counter.get(best_perm, 0) + 1

        per_obj.append(
            {
                "path": str(path),
                "is_empty_mesh": False,
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "raw_bbox_xyz": {
                    "x_min": x0,
                    "x_max": x1,
                    "y_min": y0,
                    "y_max": y1,
                    "z_min": z0,
                    "z_max": z1,
                },
                "clipped_bbox_xyz": {
                    "x_min": cx0,
                    "x_max": cx1,
                    "y_min": cy0,
                    "y_max": cy1,
                    "z_min": cz0,
                    "z_max": cz1,
                },
                "in_bounds": bool(in_bounds),
                "perm_scores": {k: float(v) for k, v in perm_scores.items()},
                "best_perm": best_perm,
                "best_perm_score": float(best_score),
                "identity_perm_score": float(perm_scores["xyz"]),
            }
        )

    identity_scores = [obj["identity_perm_score"] for obj in per_obj if not obj.get("is_empty_mesh", False)]
    best_scores = [obj["best_perm_score"] for obj in per_obj if not obj.get("is_empty_mesh", False)]

    def _median(vals):
        return float(np.median(vals)) if vals else 0.0

    return {
        "sampled_obj_count": len(sampled),
        "in_bounds_count": int(in_bounds_count),
        "outside_count": int(outside_count),
        "best_perm_counter": best_perm_counter,
        "median_identity_perm_score": _median(identity_scores),
        "median_best_perm_score": _median(best_scores),
        "union_raw_bbox_xyz": None
        if union_raw_min is None
        else {
            "x_min": int(union_raw_min[0]),
            "x_max": int(union_raw_max[0]),
            "y_min": int(union_raw_min[1]),
            "y_max": int(union_raw_max[1]),
            "z_min": int(union_raw_min[2]),
            "z_max": int(union_raw_max[2]),
        },
        "union_clipped_bbox_xyz": None
        if union_clipped_min is None
        else {
            "x_min": int(union_clipped_min[0]),
            "x_max": int(union_clipped_max[0]),
            "y_min": int(union_clipped_min[1]),
            "y_max": int(union_clipped_max[1]),
            "z_min": int(union_clipped_min[2]),
            "z_max": int(union_clipped_max[2]),
        },
        "objects": per_obj,
    }


def _open_mask_array(mask_path: str, mask_array: str = "0") -> Tuple[zarr.Array, str]:
    """Open mask zarr array from either array path or group path."""
    p = str(Path(mask_path).expanduser().resolve())
    ref = zarr.open(p, mode="r")

    if isinstance(ref, zarr.Array):
        return ref, p

    if mask_array in ref:
        arr = ref[mask_array]
        if isinstance(arr, zarr.Array):
            return arr, f"{p}/{mask_array}"

    first_array = next((v for v in ref.values() if isinstance(v, zarr.Array)), None)
    if first_array is None:
        raise RuntimeError(f"No zarr array found in mask path: {p}")

    # best effort resolved subpath
    resolved = p
    for key, value in ref.items():
        if value is first_array:
            resolved = f"{p}/{key}"
            break
    return first_array, resolved


def build_report(args: argparse.Namespace) -> Dict[str, object]:
    volume_path = str(Path(args.volume).expanduser().resolve())
    mask_path = str(Path(args.mask).expanduser().resolve())

    report: Dict[str, object] = {
        "inputs": {
            "volume": volume_path,
            "mask": mask_path,
            "obj_folder": args.obj_folder,
            "obj_glob": args.obj_glob,
            "points_json": args.points_json,
            "points_folder": args.points_folder,
            "points_glob": args.points_glob,
        },
        "flags": [],
    }

    # Volume metadata and expected shape (same resolver as mesh_to_mask)
    try:
        volume_meta = inspect_zarr_source(volume_path)
    except Exception as exc:
        volume_meta = {"error": str(exc)}
    report["volume_meta"] = volume_meta

    expected_shape_xyz = resolve_reference_shape_xyz(volume_path, array_name=args.volume_array)
    report["expected_shape_xyz_from_volume"] = list(expected_shape_xyz)

    # Mask metadata and scan
    mask_arr, resolved_mask_array_path = _open_mask_array(mask_path, mask_array=args.mask_array)
    report["mask_array_path"] = resolved_mask_array_path
    report["mask_shape_xyz"] = list(_safe_int_tuple(mask_arr.shape))
    report["mask_chunks_xyz"] = list(_safe_int_tuple(mask_arr.chunks))
    report["mask_dtype"] = str(mask_arr.dtype)

    shape_match = _safe_int_tuple(mask_arr.shape) == expected_shape_xyz
    report["shape_match_volume_vs_mask"] = bool(shape_match)
    if not shape_match:
        report["flags"].append(
            "Mask shape does not match volume reference shape used for mesh_to_mask writing."
        )

    mask_scan = _scan_mask_array(mask_arr, scan_mode=args.scan_mode, max_chunks=args.max_chunks)
    report["mask_scan"] = mask_scan

    if mask_scan.get("is_empty_in_scanned_region", True):
        if mask_scan.get("scan_mode_effective") == "full":
            report["flags"].append("Mask appears empty (full scan found zero nonzero voxels).")
        else:
            report["flags"].append(
                "Mask appears empty in scanned sample; rerun with --scan-mode full to confirm globally."
            )

    # OBJ diagnostics (optional)
    obj_specs: List[str] = []
    if args.obj_folder:
        obj_specs.append(str(Path(args.obj_folder).expanduser().resolve()))
    if args.obj_glob:
        obj_specs.extend(args.obj_glob)

    if obj_specs:
        obj_paths = collect_obj_paths(obj_specs, max_labels=args.max_objs)
        report["obj_paths_sampled_count"] = len(obj_paths)
        if len(obj_paths) == 0:
            report["flags"].append("No OBJ files found with provided obj-folder/obj-glob.")
            report["obj_diagnostics"] = {
                "sampled_obj_count": 0,
                "in_bounds_count": 0,
                "outside_count": 0,
                "best_perm_counter": {},
                "median_identity_perm_score": 0.0,
                "median_best_perm_score": 0.0,
                "union_raw_bbox_xyz": None,
                "union_clipped_bbox_xyz": None,
                "objects": [],
            }
        else:
            obj_diag = _diagnose_obj_bounds(obj_paths, expected_shape_xyz, max_objs=args.max_objs)
            report["obj_diagnostics"] = obj_diag

            if obj_diag["in_bounds_count"] == 0:
                report["flags"].append(
                    "All sampled OBJ meshes are out of bounds for the provided volume shape."
                )

            identity_median = float(obj_diag.get("median_identity_perm_score", 0.0))
            best_median = float(obj_diag.get("median_best_perm_score", 0.0))
            best_counter = obj_diag.get("best_perm_counter", {})
            dominant_perm = None
            dominant_count = 0
            for perm_label, count in best_counter.items():
                if int(count) > dominant_count:
                    dominant_perm = perm_label
                    dominant_count = int(count)

            if dominant_perm and dominant_perm != "xyz" and best_median > max(0.5, identity_median + 0.25):
                report["flags"].append(
                    f"OBJ coordinate order may be mismatched: dominant best-fit permutation is '{dominant_perm}'"
                    f" (median best {best_median:.3f} vs identity {identity_median:.3f})."
                )

            mask_bbox = mask_scan.get("nonzero_bbox_xyz")
            obj_bbox = obj_diag.get("union_clipped_bbox_xyz")
            if isinstance(mask_bbox, dict) and isinstance(obj_bbox, dict):
                overlap = not (
                    mask_bbox["x_max"] < obj_bbox["x_min"]
                    or obj_bbox["x_max"] < mask_bbox["x_min"]
                    or mask_bbox["y_max"] < obj_bbox["y_min"]
                    or obj_bbox["y_max"] < mask_bbox["y_min"]
                    or mask_bbox["z_max"] < obj_bbox["z_min"]
                    or obj_bbox["z_max"] < mask_bbox["z_min"]
                )
                report["mask_obj_bbox_overlap"] = bool(overlap)
                if not overlap:
                    report["flags"].append(
                        "Mask nonzero bounding box does not overlap clipped OBJ bounding box (location mismatch)."
                    )
    else:
        report["obj_diagnostics"] = None

    # Points-vs-OBJ-vs-mask overlap diagnostics (optional)
    point_paths = collect_points_json_paths(
        points_json=args.points_json,
        points_folder=args.points_folder,
        points_glob=args.points_glob,
        max_files=args.max_points_files,
    )
    report["points_paths_sampled_count"] = len(point_paths)

    if point_paths:
        obj_bbox = None
        if isinstance(report.get("obj_diagnostics"), dict):
            obj_bbox = report["obj_diagnostics"].get("union_clipped_bbox_xyz")
        mask_bbox = report.get("mask_scan", {}).get("nonzero_bbox_xyz")

        point_diagnostics: List[Dict[str, object]] = []
        skipped_points_files: List[Dict[str, str]] = []
        declared_axis_mismatch = 0
        low_in_bounds = 0
        no_obj_overlap = 0
        no_mask_overlap = 0
        low_on_mask = 0
        good_overlap = 0

        for point_path in point_paths:
            points_path = Path(point_path).expanduser().resolve()
            if not points_path.exists():
                report["flags"].append(f"Points JSON does not exist: {points_path}")
                continue

            try:
                points_diag = _diagnose_points_overlap(
                    points_json_path=str(points_path),
                    shape_xyz=expected_shape_xyz,
                    obj_bbox=obj_bbox,
                    mask_bbox=mask_bbox,
                    mask_arr=mask_arr,
                )
            except Exception as exc:
                skipped_points_files.append({"path": str(points_path), "reason": str(exc)})
                continue
            point_diagnostics.append(points_diag)

            best = points_diag.get("best_assumption")
            declared_axes = points_diag.get("declared_point_axes")
            if not isinstance(best, dict):
                continue

            if declared_axes and best.get("assumption") != "declared_point_axes":
                declared_axis_mismatch += 1
            if float(best.get("in_bounds_ratio", 0.0)) < 0.5:
                low_in_bounds += 1
            if best.get("bbox_overlap_obj") is False:
                no_obj_overlap += 1
            if best.get("bbox_overlap_mask") is False:
                no_mask_overlap += 1

            point_on_mask = best.get("point_on_mask", {})
            in_bounds_count = int(point_on_mask.get("in_bounds_count", 0))
            on_mask_ratio = float(point_on_mask.get("on_mask_ratio", 0.0))
            if in_bounds_count > 0 and on_mask_ratio < 0.2:
                low_on_mask += 1

            if (
                float(best.get("in_bounds_ratio", 0.0)) >= 0.5
                and best.get("bbox_overlap_mask") is not False
                and (best.get("bbox_overlap_obj") is not False if obj_bbox is not None else True)
                and (on_mask_ratio >= 0.2 if in_bounds_count > 0 else False)
            ):
                good_overlap += 1

        report["points_diagnostics"] = point_diagnostics
        report["points_skipped"] = skipped_points_files
        report["points_summary"] = {
            "files_evaluated": int(len(point_diagnostics)),
            "files_skipped": int(len(skipped_points_files)),
            "declared_axis_mismatch_count": int(declared_axis_mismatch),
            "low_in_bounds_count": int(low_in_bounds),
            "no_obj_overlap_count": int(no_obj_overlap),
            "no_mask_overlap_count": int(no_mask_overlap),
            "low_point_on_mask_count": int(low_on_mask),
            "good_overlap_count": int(good_overlap),
        }

        if len(point_diagnostics) == 0:
            report["flags"].append("No points JSON files could be evaluated.")
        if len(skipped_points_files) > 0:
            report["flags"].append(
                f"Skipped {len(skipped_points_files)} JSON file(s) that are not valid points JSON."
            )
        if declared_axis_mismatch > 0:
            report["flags"].append(
                f"{declared_axis_mismatch}/{len(point_diagnostics)} points files have declared axis metadata that "
                "looks inconsistent with volume/mask alignment."
            )
        if low_in_bounds > 0:
            report["flags"].append(
                f"{low_in_bounds}/{len(point_diagnostics)} points files have low in-bounds ratio (<50%)."
            )
        if no_obj_overlap > 0:
            report["flags"].append(
                f"{no_obj_overlap}/{len(point_diagnostics)} points files do not overlap OBJ bounds."
            )
        if no_mask_overlap > 0:
            report["flags"].append(
                f"{no_mask_overlap}/{len(point_diagnostics)} points files do not overlap nonzero mask bounds."
            )
        if low_on_mask > 0:
            report["flags"].append(
                f"{low_on_mask}/{len(point_diagnostics)} points files have low point-on-mask overlap."
            )
    else:
        report["points_diagnostics"] = None
        report["points_skipped"] = None
        report["points_summary"] = None

    return report


def print_human_summary(report: Dict[str, object]) -> None:
    print("=== Mesh->Mask Diagnosis ===")
    print(f"volume: {report['inputs']['volume']}")
    print(f"mask:   {report['inputs']['mask']}")
    print(f"mask array: {report.get('mask_array_path')}")
    print()

    print(f"expected volume shape xyz: {report.get('expected_shape_xyz_from_volume')}")
    print(f"mask shape xyz:           {report.get('mask_shape_xyz')}")
    print(f"shape match:              {report.get('shape_match_volume_vs_mask')}")

    scan = report.get("mask_scan", {})
    print()
    print("mask scan:")
    print(
        f"  mode: {scan.get('scan_mode_effective')} "
        f"({scan.get('scanned_chunks')}/{scan.get('total_chunks')} chunks)"
    )
    print(f"  nonzero chunks: {scan.get('nonzero_chunks')}")
    print(f"  nonzero voxels (scanned): {scan.get('nonzero_voxels_scanned')}")
    print(f"  nonzero bbox xyz: {scan.get('nonzero_bbox_xyz')}")

    obj_diag = report.get("obj_diagnostics")
    if isinstance(obj_diag, dict):
        print()
        print("obj diagnostics:")
        print(f"  sampled objs: {obj_diag.get('sampled_obj_count')}")
        print(f"  in bounds:    {obj_diag.get('in_bounds_count')}")
        print(f"  out of bounds:{obj_diag.get('outside_count')}")
        print(f"  best perm count: {obj_diag.get('best_perm_counter')}")
        print(
            "  median perm scores "
            f"identity(xyz)={obj_diag.get('median_identity_perm_score'):.3f}, "
            f"best={obj_diag.get('median_best_perm_score'):.3f}"
        )

    points_diag = report.get("points_diagnostics")
    if isinstance(points_diag, list):
        print()
        print("points diagnostics:")
        print(f"  files evaluated: {len(points_diag)}")
        max_rows = 10
        for row in points_diag[:max_rows]:
            best = row.get("best_assumption")
            if not isinstance(best, dict):
                print(f"  - {Path(row.get('points_json', '<unknown>')).name}: no best assumption.")
                continue
            pom = best.get("point_on_mask", {})
            print(
                "  - "
                f"{Path(row.get('points_json', '<unknown>')).name}: "
                f"best={best.get('assumption')}, "
                f"in-bounds={float(best.get('in_bounds_ratio', 0.0)):.3f}, "
                f"obj_overlap={best.get('bbox_overlap_obj')}, "
                f"mask_overlap={best.get('bbox_overlap_mask')}, "
                f"on-mask={float(pom.get('on_mask_ratio', 0.0)):.3f}"
            )
        if len(points_diag) > max_rows:
            print(f"  ... {len(points_diag) - max_rows} more files not shown")

        points_summary = report.get("points_summary")
        if isinstance(points_summary, dict):
            print(
                "  summary: "
                f"good_overlap={points_summary.get('good_overlap_count')}/"
                f"{points_summary.get('files_evaluated')}, "
                f"declared_axis_mismatch={points_summary.get('declared_axis_mismatch_count')}, "
                f"low_in_bounds={points_summary.get('low_in_bounds_count')}, "
                f"no_obj_overlap={points_summary.get('no_obj_overlap_count')}, "
                f"no_mask_overlap={points_summary.get('no_mask_overlap_count')}, "
                f"low_on_mask={points_summary.get('low_point_on_mask_count')}"
            )

    flags = report.get("flags", [])
    print()
    if flags:
        print("flags:")
        for item in flags:
            print(f"  - {item}")
    else:
        print("flags: none")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose mesh_to_mask outputs for emptiness, bounds mismatch, and axis/permutation issues."
    )
    parser.add_argument("--volume", required=True, help="Volume zarr path used (or intended) as mesh_to_mask --m3 input.")
    parser.add_argument("--volume-array", default=None, help="Optional array name when --volume points to a zarr group.")
    parser.add_argument("--mask", required=True, help="Generated mask zarr path (array path or group path).")
    parser.add_argument("--mask-array", default="0", help="Array name inside mask group (default: 0).")
    parser.add_argument("--obj-folder", default=None, help="Folder containing OBJ files (non-recursive) to compare against mask location.")
    parser.add_argument("--obj-glob", nargs="*", default=None, help="Optional OBJ glob(s) in addition to --obj-folder.")
    parser.add_argument(
        "--points-json",
        action="append",
        default=None,
        help="Optional points JSON path. May be provided multiple times.",
    )
    parser.add_argument(
        "--points-folder",
        default=None,
        help="Optional folder with points JSON files (non-recursive).",
    )
    parser.add_argument(
        "--points-glob",
        nargs="*",
        default=None,
        help="Optional glob(s) for points JSON files.",
    )
    parser.add_argument(
        "--max-points-files",
        type=int,
        default=100,
        help="Max points JSON files to evaluate from --points-folder/--points-glob/--points-json.",
    )
    parser.add_argument("--max-objs", type=int, default=50, help="Max OBJ files to sample for diagnostics.")
    parser.add_argument(
        "--scan-mode",
        choices=["auto", "full", "sample"],
        default="auto",
        help="How to scan mask chunks: auto/full/sample.",
    )
    parser.add_argument("--max-chunks", type=int, default=5000, help="Max chunks to scan in auto/sample mode.")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path.")

    args = parser.parse_args()

    try:
        report = build_report(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print_human_summary(report)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
