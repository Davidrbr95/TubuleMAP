from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

from tubulemap.cellpose_tracker.core_post_processing import save_images_to_hdf5
from tubulemap.cellpose_tracker.geometry import (
    get_frame,
    get_point_curve_ras,
    load_image,
    set_slice_view,
)
from tubulemap.cellpose_tracker.initialization import (
    initialize_tracking_state,
    load_data,
    setup_logging_and_folders,
)
from tubulemap.cellpose_tracker.mesehwithlid import reconstruct_mesh_from_pair
from tubulemap.cellpose_tracker.parameters import ALL_PARAMETERS, TracingParameters
from tubulemap.cellpose_tracker.segmentation import run_cellpose


RUN_DIR_PATTERN = re.compile(r"^Run_(\d+)$")
MASK_KEY_PATTERN = re.compile(r"^mask_(\d+)$")


class MeshGenerationError(RuntimeError):
    """Raised when mesh generation workflow cannot proceed."""


def _progress_event(progress: int, message: str, stage: Optional[str] = None, **extra) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "type": "progress",
        "progress": int(max(0, min(100, progress))),
        "message": str(message),
    }
    if stage:
        payload["stage"] = stage
    if extra:
        payload.update(extra)
    return payload


def _run_index(path: Path) -> Optional[int]:
    match = RUN_DIR_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _list_run_dirs(track_dir: Path) -> List[Tuple[int, Path]]:
    runs: List[Tuple[int, Path]] = []
    for entry in track_dir.iterdir():
        if not entry.is_dir():
            continue
        idx = _run_index(entry)
        if idx is None:
            continue
        runs.append((idx, entry))
    runs.sort(key=lambda item: item[0])
    return runs


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    raise MeshGenerationError(f"JSON payload must be an object: {path}")


def _point_count(points_json: Path) -> int:
    payload = _load_json(points_json)
    points = payload.get("points", [])
    if not isinstance(points, list):
        raise MeshGenerationError(f"Invalid 'points' in {points_json}")
    return len(points)


def _load_points_array(points_json: Path) -> np.ndarray:
    payload = _load_json(points_json)
    points = payload.get("points", [])
    if not isinstance(points, list):
        raise MeshGenerationError(f"Invalid 'points' in {points_json}")
    if not points:
        return np.zeros((0, 3), dtype=float)
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise MeshGenerationError(f"Unexpected points array shape in {points_json}: {arr.shape}")
    return arr[:, :3]


def _find_final_points_json(run_dirs: Sequence[Tuple[int, Path]]) -> Tuple[int, Path, Path]:
    for run_idx, run_dir in reversed(run_dirs):
        corrected = run_dir / "corrected_points.json"
        if corrected.exists():
            return run_idx, run_dir, corrected
        result_trace = run_dir / "result_trace.json"
        if result_trace.exists():
            return run_idx, run_dir, result_trace
    raise MeshGenerationError("No corrected_points.json or result_trace.json found in any Run_* folder.")


def _find_associated_hdf5(run_dirs: Sequence[Tuple[int, Path]], max_run_idx: int) -> Optional[Path]:
    for run_idx, run_dir in reversed(run_dirs):
        if run_idx > max_run_idx:
            continue
        candidate = run_dir / "ortho_planes.hdf5"
        if candidate.exists():
            return candidate
    return None


def _indices_changed_by_manual_correction(points_json: Path, run_dir: Path) -> List[int]:
    """
    Return indices that should be re-generated because corrected points changed
    relative to result_trace points in the same run.
    """
    if points_json.name != "corrected_points.json":
        return []

    result_trace_path = run_dir / "result_trace.json"
    if not result_trace_path.exists():
        return []

    corrected = _load_points_array(points_json)
    original = _load_points_array(result_trace_path)
    if corrected.size == 0:
        return []

    changed: Set[int] = set()
    min_len = min(len(corrected), len(original))
    if min_len > 0:
        delta = np.linalg.norm(corrected[:min_len] - original[:min_len], axis=1)
        changed.update(int(i) for i in np.where(delta > 1e-6)[0].tolist())

    if len(corrected) > len(original):
        changed.update(range(len(original), len(corrected)))

    # If corrected points trimmed the trace, regenerate all remaining corrected indices.
    if len(corrected) < len(original):
        changed.update(range(len(corrected)))

    return sorted(changed)


def _extract_mask_indices(hdf5_path: Path) -> Set[int]:
    if not hdf5_path.exists():
        return set()

    indices: Set[int] = set()
    with h5py.File(hdf5_path, "r") as handle:
        for key in handle.keys():
            match = MASK_KEY_PATTERN.match(str(key))
            if match:
                indices.add(int(match.group(1)))
    return indices


def _merge_existing_planes_from_runs(
    run_dirs: Sequence[Tuple[int, Path]],
    max_run_idx: int,
    target_hdf5_path: Path,
) -> Dict[str, object]:
    """
    Merge existing mask/raw planes from all run HDF5 files into target_hdf5_path.

    Priority:
    - Keep datasets already in target_hdf5_path.
    - Fill only missing datasets from other runs, preferring newer runs first.
    """
    target_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_hdf5_path.exists():
        with h5py.File(target_hdf5_path, "a"):
            pass

    source_candidates: List[Tuple[int, Path]] = []
    for run_idx, run_dir in sorted(run_dirs, key=lambda item: item[0], reverse=True):
        if run_idx > max_run_idx:
            continue
        candidate = run_dir / "ortho_planes.hdf5"
        if candidate.exists():
            source_candidates.append((run_idx, candidate))

    copied_masks = 0
    copied_raw = 0
    skipped_existing = 0
    failed_sources: List[str] = []
    per_source: Dict[str, Dict[str, int]] = {}

    target_resolved = target_hdf5_path.resolve()
    with h5py.File(target_hdf5_path, "a") as target_h5:
        for run_idx, source_path in source_candidates:
            source_resolved = source_path.resolve()
            if source_resolved == target_resolved:
                continue

            source_key = f"Run_{run_idx}"
            per_source[source_key] = {"mask": 0, "raw": 0}

            try:
                with h5py.File(source_path, "r") as source_h5:
                    for dataset_name in source_h5.keys():
                        if not (
                            dataset_name.startswith("mask_")
                            or dataset_name.startswith("raw_")
                        ):
                            continue
                        if dataset_name in target_h5:
                            skipped_existing += 1
                            continue
                        source_h5.copy(dataset_name, target_h5, name=dataset_name)
                        if dataset_name.startswith("mask_"):
                            copied_masks += 1
                            per_source[source_key]["mask"] += 1
                        else:
                            copied_raw += 1
                            per_source[source_key]["raw"] += 1
            except OSError:
                failed_sources.append(str(source_path))
                per_source.pop(source_key, None)

    return {
        "source_hdf5_count": len(source_candidates),
        "copied_masks": int(copied_masks),
        "copied_raw": int(copied_raw),
        "skipped_existing_datasets": int(skipped_existing),
        "failed_sources": failed_sources,
        "per_source_copied": per_source,
    }


def _run_parameters_for_run(run_dir: Path) -> Dict:
    run_params_path = run_dir / "run_parameters.json"
    if not run_params_path.exists():
        return {}
    try:
        return _load_json(run_params_path)
    except Exception:
        return {}


def _iter_run_parameters_candidates(
    run_dirs: Sequence[Tuple[int, Path]],
    preferred_idx: int,
) -> Iterable[Tuple[int, Dict]]:
    by_idx = {idx: run_dir for idx, run_dir in run_dirs}
    ordered_indices = sorted(by_idx.keys())

    # Prefer the selected/final run, then walk backward, then forward.
    backward = [idx for idx in ordered_indices if idx <= preferred_idx]
    forward = [idx for idx in ordered_indices if idx > preferred_idx]
    search_order = list(reversed(backward)) + forward

    seen: Set[int] = set()
    for run_idx in search_order:
        if run_idx in seen:
            continue
        seen.add(run_idx)
        run_params = _run_parameters_for_run(by_idx[run_idx])
        if run_params:
            yield run_idx, run_params


def _resolve_existing_path(value: Optional[str], track_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    path_str = str(value).strip()
    if path_str in {"", ".", "./"}:
        return None

    direct = Path(path_str)
    if direct.exists():
        return direct

    relative_to_track = (track_dir / path_str).resolve()
    if relative_to_track.exists():
        return relative_to_track

    return None


def _resolve_data_set_path(
    track_dir: Path,
    run_dirs: Sequence[Tuple[int, Path]],
    preferred_idx: int,
    volume_override: Optional[str],
) -> Tuple[Path, Dict]:
    override_path = _resolve_existing_path(volume_override, track_dir)
    if override_path is not None:
        return override_path, {}

    for _, run_params in _iter_run_parameters_candidates(run_dirs, preferred_idx):
        candidate = _resolve_existing_path(run_params.get("data_set_path"), track_dir)
        if candidate is not None:
            return candidate, run_params

        source_meta = run_params.get("source_metadata")
        if isinstance(source_meta, dict):
            candidate = _resolve_existing_path(source_meta.get("path"), track_dir)
            if candidate is not None:
                return candidate, run_params

    raise MeshGenerationError(
        "Could not resolve image data path from run_parameters.json. "
        "Load the source image in the GUI first or provide a valid volume override."
    )


def _estimate_vector_xyz(curvenode_xyz: Sequence[Sequence[float]], idx: int) -> np.ndarray:
    points = np.asarray(curvenode_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise MeshGenerationError("Unexpected curve node shape while estimating vectors.")

    n_points = points.shape[0]
    if n_points < 2:
        return np.array([[1.0], [0.0], [0.0]], dtype=float)

    if idx <= 0:
        delta = points[1] - points[0]
    elif idx >= n_points - 1:
        delta = points[-1] - points[-2]
    else:
        delta = points[idx + 1] - points[idx - 1]

    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        if idx > 0:
            delta = points[idx] - points[idx - 1]
            norm = float(np.linalg.norm(delta))
        if norm < 1e-8 and idx < n_points - 1:
            delta = points[idx + 1] - points[idx]
            norm = float(np.linalg.norm(delta))
        if norm < 1e-8:
            delta = np.array([1.0, 0.0, 0.0], dtype=float)
            norm = 1.0

    vector = (delta / norm).reshape(3, 1)
    return vector


def _build_fill_trace_params(
    points_json_path: Path,
    data_set_path: Path,
    run_params: Dict,
    temp_save_dir: Path,
) -> Dict:
    params = {key: info.get("default") for key, info in ALL_PARAMETERS.items()}

    # Keep only known keys from run parameters.
    for key, value in run_params.items():
        if key in ALL_PARAMETERS:
            params[key] = value

    # Hard requirements for mesh-plane filling.
    params.update(
        {
            "kp_source": True,
            "data_source": True,
            "kp_path": str(points_json_path),
            "data_set_path": str(data_set_path),
            "name": "mesh_fill",
            "save_dir": str(temp_save_dir),
            "multiprocessing": True,
            "write_ply": False,
        }
    )

    # Keep stable defaults requested in widget and by user history.
    params["diameter"] = float(params.get("diameter", 81) or 81)
    params["stepsize"] = int(params.get("stepsize", 15) or 15)
    params["jitter"] = int(params.get("jitter", 30) or 30)
    params["dim"] = int(params.get("dim", 200) or 200)

    model_suite = str(params.get("model_suite", "")).strip()
    if model_suite:
        model_path = Path(model_suite)
        if not model_path.exists():
            candidate = (Path.cwd() / model_suite).resolve()
            if candidate.exists():
                params["model_suite"] = str(candidate)
            else:
                params["model_suite"] = ALL_PARAMETERS["model_suite"]["default"]

    return params


def _fill_missing_planes_events(
    points_json_path: Path,
    target_hdf5_path: Path,
    data_set_path: Path,
    run_params: Dict,
    force_regenerate_indices: Optional[Sequence[int]] = None,
) -> Iterator[Dict[str, object]]:
    total_points = _point_count(points_json_path)
    existing = _extract_mask_indices(target_hdf5_path)
    missing = set(idx for idx in range(total_points) if idx not in existing)
    if force_regenerate_indices:
        for idx in force_regenerate_indices:
            idx_int = int(idx)
            if 0 <= idx_int < total_points:
                missing.add(idx_int)
    missing = sorted(missing)

    if not missing:
        yield _progress_event(
            78,
            "No missing orthogonal planes detected.",
            stage="fill_planes",
            total_points=total_points,
        )
        return [], [], total_points

    generated: List[int] = []
    failed: List[int] = []
    total_missing = len(missing)

    yield _progress_event(
        30,
        f"Filling {total_missing} missing/corrected orthogonal plane(s).",
        stage="fill_planes",
        total_missing=total_missing,
        total_points=total_points,
    )

    with tempfile.TemporaryDirectory(prefix="tubule_mesh_fill_") as temp_dir:
        temp_save_dir = Path(temp_dir)
        trace_params = _build_fill_trace_params(
            points_json_path=points_json_path,
            data_set_path=data_set_path,
            run_params=run_params,
            temp_save_dir=temp_save_dir,
        )

        trace = TracingParameters(**trace_params)
        setup_logging_and_folders(trace)
        trace.napari_viewer = None
        load_data(trace)
        initialize_tracking_state(trace)

        target_hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            for i, idx in enumerate(missing, start=1):
                progress = 30 + int((45 * i) / max(total_missing, 1))
                yield _progress_event(
                    progress,
                    f"Generating plane {i}/{total_missing} (index {idx}).",
                    stage="fill_planes",
                    current_index=int(idx),
                    completed=i - 1,
                    total_missing=total_missing,
                )
                try:
                    trace.reset_iteration()
                    trace.pointIndex = int(idx)
                    vector = _estimate_vector_xyz(trace.curvenode, idx)
                    trace.vectors = [vector]
                    trace.current_chunk = load_image(trace, idx=idx)
                    point_xyz = get_point_curve_ras(idx, trace.curvenode)
                    set_slice_view(trace, vector=vector, points=point_xyz)
                    get_frame(trace)
                    run_cellpose(trace)
                    if trace.current_mask is None or trace.current_raw is None:
                        failed.append(idx)
                        continue
                    save_images_to_hdf5(trace, str(target_hdf5_path))
                    generated.append(idx)
                except Exception:
                    failed.append(idx)
        finally:
            trace.close_writers()

    yield _progress_event(
        78,
        f"Plane fill complete: generated {len(generated)}, failed {len(failed)}.",
        stage="fill_planes",
        generated_count=len(generated),
        failed_count=len(failed),
    )
    return generated, failed, total_points


def _drain_generator_and_get_return(generator: Iterator[Dict[str, object]]):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def generate_mesh_from_track_folder_events(
    track_folder: str,
    volume_override: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """
    Build a mesh for the latest run in a track folder.

    Workflow:
    1. Select final points JSON (prefer corrected_points.json, then result_trace.json).
    2. Resolve associated ortho_planes.hdf5 and copy to latest run when needed.
    3. Fill missing mask/raw datasets for point indices without HDF5 entries.
    4. Reconstruct and export an OBJ mesh using the mesh-with-lid pipeline.
    """
    yield _progress_event(2, "Validating track folder.", stage="setup")
    track_dir = Path(track_folder).expanduser().resolve()
    if not track_dir.exists() or not track_dir.is_dir():
        raise MeshGenerationError(f"Track folder does not exist: {track_dir}")

    run_dirs = _list_run_dirs(track_dir)
    if not run_dirs:
        raise MeshGenerationError(f"No Run_* folders found in: {track_dir}")
    yield _progress_event(8, f"Found {len(run_dirs)} run folder(s).", stage="setup")

    final_run_idx, final_run_dir, points_json_path = _find_final_points_json(run_dirs)
    yield _progress_event(
        14,
        f"Using {points_json_path.name} from {final_run_dir.name}.",
        stage="resolve_inputs",
    )

    source_hdf5 = _find_associated_hdf5(run_dirs, max_run_idx=final_run_idx)
    target_hdf5 = final_run_dir / "ortho_planes.hdf5"
    if source_hdf5 is not None and source_hdf5.resolve() != target_hdf5.resolve():
        if not target_hdf5.exists():
            target_hdf5.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_hdf5, target_hdf5)
    elif source_hdf5 is None:
        target_hdf5.parent.mkdir(parents=True, exist_ok=True)
    yield _progress_event(20, "Resolved orthogonal planes HDF5.", stage="resolve_inputs")

    existing_before_merge = _extract_mask_indices(target_hdf5)
    yield _progress_event(22, "Merging existing planes from all Run_* folders.", stage="merge_hdf5")
    merge_stats = _merge_existing_planes_from_runs(
        run_dirs=run_dirs,
        max_run_idx=final_run_idx,
        target_hdf5_path=target_hdf5,
    )
    existing_before = _extract_mask_indices(target_hdf5)
    points_before = _point_count(points_json_path)
    missing_before = sorted(idx for idx in range(points_before) if idx not in existing_before)
    corrected_changed_indices = _indices_changed_by_manual_correction(points_json_path, final_run_dir)

    yield _progress_event(24, "Resolving source volume for plane filling.", stage="resolve_volume")
    data_set_path, params_source = _resolve_data_set_path(
        track_dir=track_dir,
        run_dirs=run_dirs,
        preferred_idx=final_run_idx,
        volume_override=volume_override,
    )
    yield _progress_event(28, f"Using volume: {data_set_path}", stage="resolve_volume")

    generated, failed, total_points = yield from _fill_missing_planes_events(
        points_json_path=points_json_path,
        target_hdf5_path=target_hdf5,
        data_set_path=data_set_path,
        run_params=params_source,
        force_regenerate_indices=corrected_changed_indices,
    )

    existing_after = _extract_mask_indices(target_hdf5)
    missing_after = sorted(idx for idx in range(total_points) if idx not in existing_after)

    mesh_dir = track_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_output = mesh_dir / f"{final_run_dir.name}_{points_json_path.stem}_mesh.obj"

    yield _progress_event(82, "Reconstructing mesh surface.", stage="reconstruct_mesh")
    mesh = reconstruct_mesh_from_pair(points_json_path, target_hdf5)
    mesh.export(mesh_output)
    yield _progress_event(94, "Mesh exported. Writing report.", stage="save_report")

    report = {
        "track_folder": str(track_dir),
        "final_run": str(final_run_dir),
        "points_json": str(points_json_path),
        "hdf5_path": str(target_hdf5),
        "resolved_data_set_path": str(data_set_path),
        "points_count": int(total_points),
        "merge_stats": merge_stats,
        "mask_indices_before_merge": sorted(int(i) for i in existing_before_merge),
        "mask_indices_before": sorted(int(i) for i in existing_before),
        "missing_indices_before": missing_before,
        "corrected_changed_indices": corrected_changed_indices,
        "generated_indices": generated,
        "failed_indices": failed,
        "mask_indices_after": sorted(int(i) for i in existing_after),
        "missing_indices_after": missing_after,
        "mesh_output": str(mesh_output),
    }

    report_path = mesh_dir / "mesh_generation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    report["report_path"] = str(report_path)
    yield _progress_event(100, "Mesh generation complete.", stage="done")
    yield {"type": "result", "report": report}
    return report


def generate_mesh_from_track_folder(
    track_folder: str,
    volume_override: Optional[str] = None,
) -> Dict[str, object]:
    return _drain_generator_and_get_return(
        generate_mesh_from_track_folder_events(
            track_folder=track_folder,
            volume_override=volume_override,
        )
    )


__all__ = [
    "MeshGenerationError",
    "generate_mesh_from_track_folder",
    "generate_mesh_from_track_folder_events",
]
