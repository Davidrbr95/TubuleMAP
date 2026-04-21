from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

from tubulemap.cellpose_tracker.mesh_to_mask import (
    MeshToMaskError,
    generate_mask_from_meshes,
    generate_mask_from_meshes_events,
)


class MeshToMaskPipelineError(RuntimeError):
    """Raised when GUI mesh-to-mask workflow inputs are invalid."""


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


def _find_zarr_root(path: Path) -> Optional[Path]:
    current = path
    while True:
        if current.suffix.lower() == ".zarr":
            return current
        if current.parent == current:
            return None
        current = current.parent


def derive_default_mask_output_path(volume_source_path: str, obj_folder_name: str) -> Path:
    """Derive `<volume-folder>/<obj_folder_name>_mesh_mask.zarr` from source path."""
    source_path = Path(str(volume_source_path)).expanduser().resolve()
    zarr_root = _find_zarr_root(source_path)

    if zarr_root is not None:
        output_dir = zarr_root.parent
    elif source_path.is_file():
        output_dir = source_path.parent
    else:
        # Non-zarr directory source: treat that directory as the volume folder.
        output_dir = source_path

    safe_name = obj_folder_name.strip() or "mesh"
    return output_dir / f"{safe_name}_mesh_mask.zarr"


def generate_mask_from_obj_folder_events(
    obj_folder: str,
    volume_source_path: str,
    output_path: Optional[str] = None,
    *,
    workers: int = 16,
    max_labels: int = 1500,
    slab_size: int = 64,
    pyramid_levels: int = 4,
    overwrite_pyramid: bool = False,
) -> Iterator[Dict[str, object]]:
    """
    Validate GUI mesh-to-mask inputs, derive output path, and stream progress events.
    """
    obj_dir = Path(str(obj_folder)).expanduser().resolve()
    if not obj_dir.exists() or not obj_dir.is_dir():
        raise MeshToMaskPipelineError(f"OBJ folder does not exist: {obj_dir}")

    volume_path = Path(str(volume_source_path)).expanduser().resolve()
    if not volume_path.exists():
        raise MeshToMaskPipelineError(f"Volume source path does not exist: {volume_path}")

    obj_files = sorted(obj_dir.glob("*.obj"))
    if not obj_files:
        raise MeshToMaskPipelineError(f"No OBJ files found in: {obj_dir}")

    if output_path is None:
        resolved_output_path = derive_default_mask_output_path(str(volume_path), obj_dir.name)
    else:
        resolved_output_path = Path(str(output_path)).expanduser().resolve()

    yield _progress_event(1, f"Validated OBJ folder with {len(obj_files)} OBJ file(s).", stage="setup")
    yield _progress_event(3, f"Using volume bounds source: {volume_path}", stage="setup")
    yield _progress_event(4, f"Mask output path: {resolved_output_path}", stage="setup")

    try:
        report = yield from generate_mask_from_meshes_events(
            m3=str(volume_path),
            objs=[str(obj_dir)],
            out=str(resolved_output_path),
            max_labels=max_labels,
            workers=workers,
            slab_size=slab_size,
            pyramid_levels=pyramid_levels,
            build_pyramid=True,
            overwrite_pyramid=overwrite_pyramid,
        )
    except MeshToMaskError as exc:
        raise MeshToMaskPipelineError(str(exc)) from exc

    return report


def generate_mask_from_obj_folder(
    obj_folder: str,
    volume_source_path: str,
    output_path: Optional[str] = None,
    *,
    workers: int = 16,
    max_labels: int = 1500,
    slab_size: int = 64,
    pyramid_levels: int = 4,
    overwrite_pyramid: bool = False,
) -> Dict[str, object]:
    """Non-event convenience wrapper for mesh-to-mask generation from GUI inputs."""
    try:
        return generate_mask_from_meshes(
            m3=str(Path(str(volume_source_path)).expanduser().resolve()),
            objs=[str(Path(str(obj_folder)).expanduser().resolve())],
            out=str(
                Path(str(output_path)).expanduser().resolve()
                if output_path is not None
                else derive_default_mask_output_path(volume_source_path, Path(str(obj_folder)).name)
            ),
            max_labels=max_labels,
            workers=workers,
            slab_size=slab_size,
            pyramid_levels=pyramid_levels,
            build_pyramid=True,
            overwrite_pyramid=overwrite_pyramid,
        )
    except MeshToMaskError as exc:
        raise MeshToMaskPipelineError(str(exc)) from exc


__all__ = [
    "MeshToMaskPipelineError",
    "derive_default_mask_output_path",
    "generate_mask_from_obj_folder",
    "generate_mask_from_obj_folder_events",
]
