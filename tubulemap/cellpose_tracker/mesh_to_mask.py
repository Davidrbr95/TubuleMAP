#!/usr/bin/env python3
"""Mesh-centric OBJ -> labels.zarr writer (Z-slab voxelization, 1:1 XYZ mapping)."""

import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import trimesh
import zarr
from numcodecs import Blosc
from scipy.ndimage import binary_fill_holes


class MeshToMaskError(RuntimeError):
    """Raised when mesh-to-mask generation cannot proceed."""


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


def _drain_generator_and_get_return(generator: Iterator[Dict[str, object]]) -> Dict[str, object]:
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def load_mesh_vertices(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices/faces without transforms (already expected in XYZ index space)."""
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    return vertices, faces


_GLOBAL_OUT_PATH = None
_GLOBAL_SHAPE_XYZ = None
_GLOBAL_LOCK = None
_GLOBAL_SLAB_SIZE = None


def _worker_init(out_path: str, shape_xyz: Tuple[int, int, int], lock, slab_size: int) -> None:
    global _GLOBAL_OUT_PATH, _GLOBAL_SHAPE_XYZ
    global _GLOBAL_LOCK, _GLOBAL_SLAB_SIZE

    _GLOBAL_OUT_PATH = out_path
    _GLOBAL_SHAPE_XYZ = tuple(shape_xyz)
    _GLOBAL_LOCK = lock
    _GLOBAL_SLAB_SIZE = int(slab_size)


def process_one_mesh(mi_and_path: Tuple[int, str]) -> Dict[str, object]:
    """Voxelize a single mesh and merge it into labels.zarr."""
    start_mesh = time.time()
    mesh_index, path = mi_and_path
    label_id = np.uint16(mesh_index + 1)

    vertices_idx, faces = load_mesh_vertices(path)
    if vertices_idx.size == 0 or faces.size == 0:
        return {"mesh_index": mesh_index, "path": path, "voxels": 0, "n_slabs": 0}

    mins = vertices_idx.min(axis=0)
    maxs = vertices_idx.max(axis=0)

    x0_full = int(math.floor(mins[0]))
    y0_full = int(math.floor(mins[1]))
    z0_full = int(math.floor(mins[2]))
    x1_full = int(math.ceil(maxs[0]))
    y1_full = int(math.ceil(maxs[1]))
    z1_full = int(math.ceil(maxs[2]))

    X, Y, Z = _GLOBAL_SHAPE_XYZ
    x0_full = max(x0_full, 0)
    y0_full = max(y0_full, 0)
    z0_full = max(z0_full, 0)
    x1_full = min(x1_full, X)
    y1_full = min(y1_full, Y)
    z1_full = min(z1_full, Z)

    if z1_full <= z0_full or y1_full <= y0_full or x1_full <= x0_full:
        return {"mesh_index": mesh_index, "path": path, "voxels": 0, "n_slabs": 0}

    z_coords = vertices_idx[:, 2]
    face_z_min = z_coords[faces].min(axis=1)
    face_z_max = z_coords[faces].max(axis=1)

    slab_size = _GLOBAL_SLAB_SIZE
    total_voxels = 0
    n_slabs = 0

    z_start_global = z0_full
    z_end_global = z1_full
    margin = 1.0

    while z_start_global < z_end_global:
        z_slab0 = z_start_global
        z_slab1 = min(z_start_global + slab_size, z_end_global)

        face_mask = (face_z_max >= (z_slab0 - margin)) & (face_z_min <= (z_slab1 + margin))
        if not np.any(face_mask):
            z_start_global = z_slab1
            continue

        faces_slab_old = faces[face_mask]
        used_vert_idx = np.unique(faces_slab_old.reshape(-1))
        if used_vert_idx.size == 0:
            z_start_global = z_slab1
            continue

        old_to_new = -np.ones(len(vertices_idx), dtype=np.int64)
        old_to_new[used_vert_idx] = np.arange(used_vert_idx.size, dtype=np.int64)
        faces_slab = old_to_new[faces_slab_old]

        vertices_slab_idx = vertices_idx[used_vert_idx].copy()

        mins_sub = vertices_slab_idx.min(axis=0)
        maxs_sub = vertices_slab_idx.max(axis=0)

        x0_int = int(math.floor(mins_sub[0]))
        y0_int = int(math.floor(mins_sub[1]))
        z0_int = int(math.floor(mins_sub[2]))
        x1_int = int(math.ceil(maxs_sub[0]))
        y1_int = int(math.ceil(maxs_sub[1]))
        z1_int = int(math.ceil(maxs_sub[2]))

        x0_int = max(x0_int, 0)
        y0_int = max(y0_int, 0)
        z0_int = max(z0_int, 0)
        x1_int = min(x1_int, X)
        y1_int = min(y1_int, Y)
        z1_int = min(z1_int, Z)

        if z1_int <= z0_int or y1_int <= y0_int or x1_int <= x0_int:
            z_start_global = z_slab1
            continue

        vertices_local = vertices_slab_idx.copy()
        vertices_local[:, 0] -= x0_int
        vertices_local[:, 1] -= y0_int
        vertices_local[:, 2] -= z0_int

        mesh_local = trimesh.Trimesh(vertices=vertices_local, faces=faces_slab, process=False)
        if vertices_local.size == 0 or faces_slab.size == 0:
            z_start_global = z_slab1
            continue

        try:
            vox = mesh_local.voxelized(pitch=1.0)
        except Exception:
            z_start_global = z_slab1
            continue

        vox_matrix = vox.matrix.astype(bool)
        try:
            if mesh_local.is_watertight:
                vox_matrix = vox.fill().matrix.astype(bool)
        except Exception:
            pass

        binary_xyz = vox_matrix.copy()
        nz = binary_xyz.shape[2]
        for z in range(nz):
            if binary_xyz[..., z].any():
                binary_xyz[..., z] = binary_fill_holes(binary_xyz[..., z])

        binary_xyz = binary_xyz.astype(np.uint8)
        nx_local, ny_local, nz_local = binary_xyz.shape

        g_x0 = x0_int
        g_x1 = x0_int + nx_local
        g_y0 = y0_int
        g_y1 = y0_int + ny_local
        g_z0 = z0_int
        g_z1 = z0_int + nz_local

        g_z0 = max(g_z0, 0, z_slab0)
        g_z1 = min(g_z1, Z, z_slab1)
        g_y0 = max(g_y0, 0)
        g_y1 = min(g_y1, Y)
        g_x0 = max(g_x0, 0)
        g_x1 = min(g_x1, X)

        if g_z1 <= g_z0 or g_y1 <= g_y0 or g_x1 <= g_x0:
            z_start_global = z_slab1
            continue

        l_x0 = g_x0 - x0_int
        l_x1 = g_x1 - x0_int
        l_y0 = g_y0 - y0_int
        l_y1 = g_y1 - y0_int
        l_z0 = g_z0 - z0_int
        l_z1 = g_z1 - z0_int

        sub_binary = binary_xyz[l_x0:l_x1, l_y0:l_y1, l_z0:l_z1]
        if sub_binary.size == 0:
            z_start_global = z_slab1
            continue

        mask = sub_binary > 0
        n_vox_slab = int(mask.sum())
        if n_vox_slab == 0:
            z_start_global = z_slab1
            continue

        with _GLOBAL_LOCK:
            root = zarr.open_group(_GLOBAL_OUT_PATH, mode="r+")
            arr = root["0"]

            sub = arr[g_x0:g_x1, g_y0:g_y1, g_z0:g_z1].astype(np.uint16, copy=True)
            sub_masked = sub[mask]
            sub[mask] = np.maximum(sub_masked, label_id)
            arr[g_x0:g_x1, g_y0:g_y1, g_z0:g_z1] = sub

        total_voxels += n_vox_slab
        n_slabs += 1
        z_start_global = z_slab1

    elapsed_mesh = time.time() - start_mesh
    return {
        "mesh_index": mesh_index,
        "path": path,
        "voxels": int(total_voxels),
        "elapsed_sec": round(elapsed_mesh, 3),
        "x0": x0_full,
        "x1": x1_full,
        "y0": y0_full,
        "y1": y1_full,
        "z0": z0_full,
        "z1": z1_full,
        "n_slabs": n_slabs,
    }


def build_multiscale_pyramid(out_path: str, base_dataset: str = "0", num_levels: int = 4, downscale: int = 2) -> None:
    """Build a max-pooled multiscale label pyramid inside the same zarr store."""
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store)
    if base_dataset not in root:
        raise MeshToMaskError(f"Base dataset '{base_dataset}' not found in {out_path}")

    base = root[base_dataset]
    if base.ndim != 3:
        raise MeshToMaskError(f"Expected 3D labels, got shape {base.shape}")

    datasets = [{"path": base_dataset, "transform": {"type": "scale", "scale": [1.0, 1.0, 1.0]}}]

    current_name = base_dataset
    current_scale = 1.0
    base_chunks = base.chunks

    for level in range(1, num_levels):
        next_name = str(level)
        current_scale *= downscale

        src = root[current_name]
        darr = da.from_zarr(src)
        reduced = da.coarsen(np.max, darr, {0: downscale, 1: downscale, 2: downscale}, trim_excess=True)
        reduced = reduced.rechunk(base_chunks)
        reduced = reduced.astype(np.uint16)
        reduced.to_zarr(
            store,
            component=next_name,
            overwrite=True,
            compute=True,
            compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )

        datasets.append({
            "path": next_name,
            "transform": {"type": "scale", "scale": [float(current_scale)] * 3},
        })
        current_name = next_name

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "x", "type": "space", "unit": "pixel"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "z", "type": "space", "unit": "pixel"},
        ],
        "datasets": datasets,
    }]


def collect_obj_paths(obj_specs: Sequence[str], max_labels: Optional[int] = None) -> List[str]:
    """Collect OBJ paths from directories and glob expressions (non-recursive for dirs)."""
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
                raise MeshToMaskError(f"No zarr array found in reference path: {reference_path}")

    shape = arr_ref.shape[-3:] if arr_ref.ndim >= 3 else arr_ref.shape
    if len(shape) != 3:
        raise MeshToMaskError(f"Reference shape must be 3D. Got shape: {arr_ref.shape}")
    return tuple(int(v) for v in shape)


def ensure_labels_store(out_path: str, shape_xyz: Tuple[int, int, int], source_path: str) -> bool:
    """Create output labels zarr when missing. Returns True if newly created."""
    if os.path.exists(out_path):
        return False

    root = zarr.open_group(out_path, mode="w")
    root.create_dataset(
        "0",
        shape=shape_xyz,
        chunks=(64, 512, 512),
        dtype=np.uint16,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
        fill_value=0,
    )
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
        "datasets": [{"path": "0"}],
    }]
    root["0"].attrs["source"] = str(source_path)
    return True


def load_done_mesh_indices(resume_log: Optional[str]) -> set:
    """Read completed mesh indices from resume log."""
    if not resume_log or not os.path.exists(resume_log):
        return set()

    done = set()
    with open(resume_log, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            mesh_index = rec.get("mesh_index")
            if mesh_index is not None:
                done.add(int(mesh_index))
    return done


def maybe_build_pyramid(out_path: str, pyramid_levels: int, overwrite_pyramid: bool) -> Dict[str, object]:
    """Build/reuse multiscale pyramid and return summary metadata."""
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store)

    existing_levels = sorted(int(k) for k in root.keys() if str(k).isdigit())
    has_pyramid = any(level > 0 for level in existing_levels)

    report = {
        "requested": True,
        "built": False,
        "skipped": False,
        "overwrite": bool(overwrite_pyramid),
        "existing_levels": existing_levels,
        "num_levels": int(pyramid_levels),
    }

    if overwrite_pyramid or not has_pyramid:
        build_multiscale_pyramid(
            out_path=out_path,
            base_dataset="0",
            num_levels=int(pyramid_levels),
            downscale=2,
        )
        report["built"] = True
    else:
        report["skipped"] = True

    return report


def generate_mask_from_meshes_events(
    m3: str,
    objs: Sequence[str],
    out: str,
    *,
    array: Optional[str] = None,
    max_labels: int = 1500,
    workers: int = 16,
    resume_log: Optional[str] = None,
    slab_size: int = 64,
    pyramid_levels: int = 4,
    build_pyramid: bool = True,
    overwrite_pyramid: bool = False,
) -> Iterator[Dict[str, object]]:
    """Generate labels.zarr from OBJ meshes and yield structured progress events."""
    started_at = time.time()

    if int(workers) <= 0:
        raise MeshToMaskError("workers must be greater than 0")
    if int(slab_size) <= 0:
        raise MeshToMaskError("slab_size must be greater than 0")
    if build_pyramid and int(pyramid_levels) <= 0:
        raise MeshToMaskError("pyramid_levels must be greater than 0 when pyramid build is enabled")

    reference_path = Path(str(m3)).expanduser().resolve()
    if not reference_path.exists():
        raise MeshToMaskError(f"Reference volume path does not exist: {reference_path}")

    output_path = Path(str(out)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obj_paths = collect_obj_paths(objs, max_labels=max_labels)
    if not obj_paths:
        raise MeshToMaskError("No OBJ files found for the requested folder(s)/pattern(s).")

    yield _progress_event(2, "Validating mesh-to-mask inputs.", stage="setup")

    shape_xyz = resolve_reference_shape_xyz(str(reference_path), array_name=array)
    yield _progress_event(8, f"Resolved reference shape XYZ={shape_xyz}.", stage="setup")

    created_output = ensure_labels_store(str(output_path), shape_xyz, str(reference_path))
    if created_output:
        yield _progress_event(12, f"Created output labels store: {output_path}", stage="setup")
    else:
        yield _progress_event(12, f"Using existing labels store: {output_path}", stage="setup")

    done_meshes = load_done_mesh_indices(resume_log)
    work_items = [(mesh_index, path) for mesh_index, path in enumerate(obj_paths) if mesh_index not in done_meshes]

    processed_results: List[Dict[str, object]] = []
    failed_meshes: List[Dict[str, object]] = []

    if not work_items:
        yield _progress_event(80, "Nothing to do: all meshes already listed in resume log.", stage="voxelize")
    else:
        yield _progress_event(
            18,
            f"Voxelizing {len(work_items)} mesh(es) from {len(obj_paths)} OBJ file(s).",
            stage="voxelize",
            total=len(work_items),
        )

        manager = Manager()
        lock = manager.Lock()

        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_worker_init,
            initargs=(str(output_path), shape_xyz, lock, int(slab_size)),
        ) as executor, (
            open(resume_log, "a", encoding="utf-8") if resume_log else open(os.devnull, "w", encoding="utf-8")
        ) as log_handle:
            futures = {executor.submit(process_one_mesh, item): item for item in work_items}
            total = len(futures)
            completed = 0

            for future in as_completed(futures):
                mesh_index, mesh_path = futures[future]
                completed += 1
                progress = 18 + int((62 * completed) / max(total, 1))

                try:
                    result = future.result()
                    if result is None:
                        result = {"mesh_index": mesh_index, "path": mesh_path, "voxels": 0, "n_slabs": 0}
                    processed_results.append(result)
                    if resume_log:
                        log_handle.write(json.dumps(result) + "\n")
                        log_handle.flush()
                    yield _progress_event(
                        progress,
                        f"Processed mesh {completed}/{total}: {Path(mesh_path).name}",
                        stage="voxelize",
                        completed=completed,
                        total=total,
                        mesh_index=mesh_index,
                        mesh_path=str(mesh_path),
                    )
                except Exception as exc:
                    failed = {"mesh_index": mesh_index, "path": str(mesh_path), "error": str(exc)}
                    failed_meshes.append(failed)
                    yield _progress_event(
                        progress,
                        f"Failed mesh {completed}/{total}: {Path(mesh_path).name}",
                        stage="voxelize",
                        completed=completed,
                        total=total,
                        mesh_index=mesh_index,
                        mesh_path=str(mesh_path),
                        error=str(exc),
                    )

    pyramid_report = {
        "requested": bool(build_pyramid),
        "built": False,
        "skipped": False,
        "existing_levels": [],
        "num_levels": int(pyramid_levels),
        "overwrite": bool(overwrite_pyramid),
    }

    if build_pyramid:
        yield _progress_event(84, "Preparing label pyramid.", stage="pyramid")
        pyramid_report = maybe_build_pyramid(
            out_path=str(output_path),
            pyramid_levels=int(pyramid_levels),
            overwrite_pyramid=bool(overwrite_pyramid),
        )
        if pyramid_report.get("built"):
            yield _progress_event(96, "Built multiscale pyramid.", stage="pyramid")
        else:
            yield _progress_event(96, "Existing multiscale pyramid reused.", stage="pyramid")
    else:
        yield _progress_event(96, "Skipped multiscale pyramid build.", stage="pyramid")

    elapsed = round(time.time() - started_at, 3)
    report: Dict[str, object] = {
        "reference_path": str(reference_path),
        "output_path": str(output_path),
        "shape_xyz": [int(v) for v in shape_xyz],
        "obj_count": int(len(obj_paths)),
        "work_items_count": int(len(work_items)),
        "processed_meshes": int(len(processed_results)),
        "failed_meshes": int(len(failed_meshes)),
        "results": processed_results,
        "failed_items": failed_meshes,
        "workers": int(workers),
        "slab_size": int(slab_size),
        "max_labels": int(max_labels),
        "resume_log": str(resume_log) if resume_log else None,
        "created_output": bool(created_output),
        "pyramid": pyramid_report,
        "elapsed_sec": elapsed,
    }

    yield _progress_event(100, "Mask generation complete.", stage="done")
    yield {"type": "result", "report": report}
    return report


def generate_mask_from_meshes(**kwargs) -> Dict[str, object]:
    """Generate labels.zarr from OBJ meshes and return a summary report."""
    return _drain_generator_and_get_return(generate_mask_from_meshes_events(**kwargs))


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Mesh-centric OBJ->labels.zarr with Z-slab voxelization (1:1 XYZ mapping)",
    )
    parser.add_argument("--m3", required=True, help="Reference zarr (for shape)")
    parser.add_argument("--array", default=None, help="Array name if group; omit if root is array")
    parser.add_argument("--objs", required=True, nargs="+", help="OBJ globs or directories")
    parser.add_argument("--out", required=True, help="Output labels.zarr path")
    parser.add_argument("--max-labels", type=int, default=1500, help="Maximum number of meshes/labels")
    parser.add_argument("--workers", type=int, default=16, help="Worker process count")
    parser.add_argument("--resume-log", default=None, help="JSONL path for resume bookkeeping")
    parser.add_argument("--slab-size", type=int, default=64, help="Z slab thickness in voxels")
    parser.add_argument("--pyramid-levels", type=int, default=4, help="Number of multiscale levels")
    parser.add_argument("--no-pyramid", action="store_true", help="Skip multiscale pyramid generation")
    parser.add_argument("--overwrite-pyramid", action="store_true", help="Rebuild pyramid levels even if present")

    args = parser.parse_args()

    try:
        report: Optional[Dict[str, object]] = None
        for event in generate_mask_from_meshes_events(
            m3=args.m3,
            objs=args.objs,
            out=args.out,
            array=args.array,
            max_labels=args.max_labels,
            workers=args.workers,
            resume_log=args.resume_log,
            slab_size=args.slab_size,
            pyramid_levels=args.pyramid_levels,
            build_pyramid=not args.no_pyramid,
            overwrite_pyramid=args.overwrite_pyramid,
        ):
            payload_type = event.get("type")
            if payload_type == "progress":
                print(f"[{int(event.get('progress', 0)):>3}%] {event.get('message', '')}")
            elif payload_type == "result":
                report = event.get("report")

        if report is not None:
            print(f"\nDone. Output: {report.get('output_path')}")
            print(f"Processed meshes: {report.get('processed_meshes')}  Failed meshes: {report.get('failed_meshes')}")
        return 0
    except MeshToMaskError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
