import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr


SPATIAL_AXES = ("z", "y", "x")
DEFAULT_SCALE_ZYX = [1.0, 1.0, 1.0]
DEFAULT_TRANSLATION_ZYX = [0.0, 0.0, 0.0]


def default_axes_for_ndim(ndim: int) -> List[str]:
    """Build the default axes for ndim."""
    if ndim == 5:
        return ["t", "c", "z", "y", "x"]
    if ndim == 4:
        return ["c", "z", "y", "x"]
    if ndim == 3:
        return ["z", "y", "x"]
    if ndim == 2:
        return ["y", "x"]
    return [f"dim_{i}" for i in range(ndim)]


def normalize_axes(axes_meta: Any, ndim: int) -> List[str]:
    """Normalize axes."""
    if axes_meta is None:
        return default_axes_for_ndim(ndim)

    names: List[str] = []
    for axis in axes_meta:
        if isinstance(axis, dict):
            axis_name = str(axis.get("name", "")).strip().lower()
        else:
            axis_name = str(axis).strip().lower()
        names.append(axis_name)

    if len(names) != ndim:
        return default_axes_for_ndim(ndim)
    return names


def axis_index(axes: Sequence[str], axis_name: str) -> Optional[int]:
    """Compute axis index."""
    axis_name = axis_name.lower()
    for idx, candidate in enumerate(axes):
        if str(candidate).lower() == axis_name:
            return idx
    return None


def shape_ratio_scale_zyx(base_shape: Sequence[int], level_shape: Sequence[int], axes: Sequence[str]) -> List[float]:
    """Compute shape ratio scale zyx."""
    result: List[float] = []
    for spatial_axis in SPATIAL_AXES:
        idx = axis_index(axes, spatial_axis)
        if idx is None:
            result.append(1.0)
            continue
        base_dim = float(base_shape[idx])
        level_dim = float(level_shape[idx])
        if level_dim <= 0:
            result.append(1.0)
            continue
        ratio = base_dim / level_dim
        if not math.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
        result.append(float(ratio))
    return result


def extract_scale_translation(dataset_entry: Dict[str, Any], ndim: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Compute extract scale translation."""
    scale = np.ones(ndim, dtype=float)
    translation = np.zeros(ndim, dtype=float)
    has_scale = False

    for transform in dataset_entry.get("coordinateTransformations", []):
        t_type = transform.get("type")
        if t_type == "scale":
            values = np.asarray(transform.get("scale", scale), dtype=float)
            if values.size == ndim:
                scale = values
                has_scale = True
        elif t_type == "translation":
            values = np.asarray(transform.get("translation", translation), dtype=float)
            if values.size == ndim:
                translation = values
    return scale, translation, has_scale


def _safe_int(value: Any, fallback: int = 0) -> int:
    """Compute safe int."""
    try:
        return int(value)
    except Exception:
        return fallback


def _is_numeric_key(name: str) -> bool:
    """Return whether key is numeric."""
    try:
        int(name)
        return True
    except Exception:
        return False


def _sorted_level_entries(entries: Iterable[Tuple[str, zarr.Array]]) -> List[Tuple[str, zarr.Array]]:
    """Compute sorted level entries."""
    materialized = list(entries)
    numeric_entries = [(name, arr) for name, arr in materialized if _is_numeric_key(name)]
    if len(numeric_entries) == len(materialized):
        return sorted(materialized, key=lambda item: int(item[0]))
    return sorted(materialized, key=lambda item: item[0])


def _level_dict(
    *,
    index: int,
    level_path: str,
    absolute_path: str,
    shape: Sequence[int],
    scale_zyx: Sequence[float],
    translation_zyx: Sequence[float],
) -> Dict[str, Any]:
    """Compute level dict."""
    return {
        "index": int(index),
        "path": str(level_path),
        "absolute_path": str(absolute_path),
        "shape": [int(v) for v in shape],
        "scale_zyx": [float(v) for v in scale_zyx],
        "translation_zyx": [float(v) for v in translation_zyx],
    }


def _inspect_regular_array(path: str, array: zarr.Array) -> Dict[str, Any]:
    """Inspect regular array."""
    axes = default_axes_for_ndim(array.ndim)
    levels = [
        _level_dict(
            index=0,
            level_path="",
            absolute_path=path,
            shape=array.shape,
            scale_zyx=DEFAULT_SCALE_ZYX,
            translation_zyx=DEFAULT_TRANSLATION_ZYX,
        )
    ]
    return {
        "path": path,
        "source_kind": "regular",
        "axes": axes,
        "is_multiscale": False,
        "levels": levels,
    }


def _inspect_regular_group(path: str, group: zarr.Group) -> Dict[str, Any]:
    """Inspect regular group."""
    level_entries: List[Tuple[str, zarr.Array]] = []
    for key in list(group.keys()):
        try:
            entry = group[key]
        except Exception:
            continue
        if isinstance(entry, zarr.Array):
            level_entries.append((str(key), entry))

    if not level_entries:
        raise ValueError(f"No array datasets were found under zarr group: {path}")

    level_entries = _sorted_level_entries(level_entries)
    base_shape = level_entries[0][1].shape
    base_ndim = level_entries[0][1].ndim
    axes = default_axes_for_ndim(base_ndim)

    levels: List[Dict[str, Any]] = []
    for idx, (level_path, arr) in enumerate(level_entries):
        if arr.ndim != base_ndim:
            raise ValueError(
                f"Inconsistent level ndim in {path}: level '{level_path}' has ndim={arr.ndim}, "
                f"expected ndim={base_ndim}."
            )
        scale_zyx = shape_ratio_scale_zyx(base_shape, arr.shape, axes)
        abs_path = os.path.join(path, level_path) if level_path else path
        levels.append(
            _level_dict(
                index=idx,
                level_path=level_path,
                absolute_path=abs_path,
                shape=arr.shape,
                scale_zyx=scale_zyx,
                translation_zyx=DEFAULT_TRANSLATION_ZYX,
            )
        )

    return {
        "path": path,
        "source_kind": "regular",
        "axes": axes,
        "is_multiscale": len(levels) > 1,
        "levels": levels,
    }


def _inspect_ome_group(path: str, group: zarr.Group, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect ome group."""
    multiscales = attrs.get("multiscales", [])
    if not multiscales:
        raise ValueError(f"OME-Zarr source at {path} has empty 'multiscales' metadata.")

    primary = multiscales[0]
    datasets = primary.get("datasets", [])
    if not datasets:
        raise ValueError(f"OME-Zarr source at {path} has no datasets in multiscales metadata.")

    # Build temporary records first so we can decide whether to trust transform scales.
    temp_records: List[Dict[str, Any]] = []
    for idx, dataset in enumerate(datasets):
        level_path = str(dataset.get("path", "")).strip()
        if not level_path:
            raise ValueError(f"OME-Zarr dataset at index {idx} is missing a dataset path.")
        try:
            arr = group[level_path]
        except Exception as exc:
            raise ValueError(f"Unable to read OME-Zarr dataset '{level_path}' under {path}.") from exc
        if not isinstance(arr, zarr.Array):
            raise ValueError(f"OME-Zarr dataset '{level_path}' is not an array.")
        scale_vec, translation_vec, has_scale = extract_scale_translation(dataset, arr.ndim)
        temp_records.append(
            {
                "index": idx,
                "path": level_path,
                "array": arr,
                "shape": arr.shape,
                "scale_vec": scale_vec,
                "translation_vec": translation_vec,
                "has_scale": has_scale,
            }
        )

    base_record = temp_records[0]
    axes = normalize_axes(primary.get("axes"), base_record["array"].ndim)
    base_shape = base_record["shape"]
    base_scale_vec = base_record["scale_vec"]
    all_have_scale = all(record["has_scale"] for record in temp_records)

    levels: List[Dict[str, Any]] = []
    for record in temp_records:
        if all_have_scale:
            ratio = []
            for spatial_axis in SPATIAL_AXES:
                axis_idx = axis_index(axes, spatial_axis)
                if axis_idx is None:
                    ratio.append(1.0)
                    continue
                denom = float(base_scale_vec[axis_idx])
                num = float(record["scale_vec"][axis_idx])
                if denom == 0 or not math.isfinite(num) or not math.isfinite(denom):
                    ratio.append(1.0)
                else:
                    value = num / denom
                    ratio.append(float(value) if value > 0 else 1.0)
            scale_zyx = ratio
        else:
            scale_zyx = shape_ratio_scale_zyx(base_shape, record["shape"], axes)

        translation_zyx = []
        for spatial_axis in SPATIAL_AXES:
            axis_idx = axis_index(axes, spatial_axis)
            if axis_idx is None:
                translation_zyx.append(0.0)
            else:
                translation_zyx.append(float(record["translation_vec"][axis_idx]))

        abs_path = os.path.join(path, record["path"])
        levels.append(
            _level_dict(
                index=record["index"],
                level_path=record["path"],
                absolute_path=abs_path,
                shape=record["shape"],
                scale_zyx=scale_zyx,
                translation_zyx=translation_zyx,
            )
        )

    return {
        "path": path,
        "source_kind": "ome",
        "axes": axes,
        "is_multiscale": len(levels) > 1,
        "levels": levels,
    }


def inspect_zarr_source(path: str) -> Dict[str, Any]:
    """Inspect zarr source."""
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


def get_axis_size_for_level(source_meta: Dict[str, Any], level_idx: int, axis_name: str) -> Optional[int]:
    """Get axis size for level."""
    levels = source_meta.get("levels", [])
    if not levels:
        return None
    if level_idx < 0 or level_idx >= len(levels):
        return None
    idx = axis_index(source_meta.get("axes", []), axis_name)
    if idx is None:
        return None
    shape = levels[level_idx].get("shape", [])
    if idx >= len(shape):
        return None
    return int(shape[idx])


def has_translation_mismatch(source_meta: Dict[str, Any], level_idx: Optional[int] = None, tol: float = 1e-6) -> bool:
    """Return whether translation mismatch exists."""
    levels = source_meta.get("levels", [])
    if len(levels) <= 1:
        return False

    base = np.asarray(levels[0].get("translation_zyx", DEFAULT_TRANSLATION_ZYX), dtype=float)
    candidate_indices = [int(level_idx)] if level_idx is not None else list(range(1, len(levels)))

    for idx in candidate_indices:
        if idx < 0 or idx >= len(levels):
            continue
        current = np.asarray(levels[idx].get("translation_zyx", DEFAULT_TRANSLATION_ZYX), dtype=float)
        if np.any(np.abs(current - base) > tol):
            return True
    return False


def open_level_array(source_meta: Dict[str, Any], level_idx: int) -> zarr.Array:
    """Open level array."""
    levels = source_meta.get("levels", [])
    if level_idx < 0 or level_idx >= len(levels):
        raise ValueError(f"Invalid run level {level_idx}. Available range is 0..{len(levels)-1}.")
    level_path = levels[level_idx]["absolute_path"]
    arr = zarr.open(level_path, mode="r")
    if not isinstance(arr, zarr.Array):
        raise ValueError(f"Resolved level path is not a zarr array: {level_path}")
    return arr


@dataclass
class TCZYXArrayView:
    source: zarr.Array
    axes: Sequence[str]
    time_index: int = 0
    channel_index: int = 0

    def __post_init__(self) -> None:
        """Compute post init."""
        axis_map = {str(axis).lower(): i for i, axis in enumerate(self.axes)}
        missing_spatial = [axis for axis in SPATIAL_AXES if axis not in axis_map]
        if missing_spatial:
            raise ValueError(f"Volume axes are missing spatial dimensions {missing_spatial}: {self.axes}")

        self._axis_map = axis_map
        self._z_idx = axis_map["z"]
        self._y_idx = axis_map["y"]
        self._x_idx = axis_map["x"]
        self._t_idx = axis_map.get("t")
        self._c_idx = axis_map.get("c")

        if self._t_idx is not None:
            t_size = int(self.source.shape[self._t_idx])
            if not (0 <= int(self.time_index) < t_size):
                raise ValueError(f"run_time_index {self.time_index} is out of range [0, {t_size-1}]")
        if self._c_idx is not None:
            c_size = int(self.source.shape[self._c_idx])
            if not (0 <= int(self.channel_index) < c_size):
                raise ValueError(f"run_channel_index {self.channel_index} is out of range [0, {c_size-1}]")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Compute shape."""
        return (
            int(self.source.shape[self._z_idx]),
            int(self.source.shape[self._y_idx]),
            int(self.source.shape[self._x_idx]),
        )

    def __getitem__(self, item: Any) -> Any:
        """Compute getitem."""
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) != 3:
            raise IndexError("TCZYXArrayView expects 3D indexing as [z, y, x].")

        def _is_int_index(value: Any) -> bool:
            return isinstance(value, (int, np.integer))

        full_index: List[Any] = [slice(None)] * self.source.ndim
        if self._t_idx is not None:
            full_index[self._t_idx] = int(self.time_index)
        if self._c_idx is not None:
            full_index[self._c_idx] = int(self.channel_index)

        full_index[self._z_idx] = item[0]
        full_index[self._y_idx] = item[1]
        full_index[self._x_idx] = item[2]
        result = self.source[tuple(full_index)]

        # Ensure returned array dimensions follow canonical [z, y, x] order
        # (filtered by non-integer indexed axes), regardless of source axis order.
        if not hasattr(result, "ndim") or int(getattr(result, "ndim", 0)) <= 1:
            return result

        surviving_source_axes = [
            axis for axis in range(self.source.ndim) if not _is_int_index(full_index[axis])
        ]
        desired_source_axes = [
            axis_idx
            for axis_idx, axis_item in (
                (self._z_idx, item[0]),
                (self._y_idx, item[1]),
                (self._x_idx, item[2]),
            )
            if not _is_int_index(axis_item)
        ]

        # Preserve any non-spatial surviving axes at the end in original source order.
        desired_source_axes.extend(
            [axis for axis in surviving_source_axes if axis not in desired_source_axes]
        )

        if len(desired_source_axes) != len(surviving_source_axes):
            return result

        perm = [surviving_source_axes.index(axis) for axis in desired_source_axes]
        if perm == list(range(len(perm))):
            return result
        return np.transpose(result, axes=perm)


def create_run_volume_view(
    source_array: zarr.Array,
    axes: Sequence[str],
    run_time_index: int,
    run_channel_index: int,
) -> Any:
    """Create run volume view."""
    axis_names = [str(axis).lower() for axis in axes]
    if source_array.ndim == 3 and axis_names == ["z", "y", "x"]:
        return source_array
    return TCZYXArrayView(
        source=source_array,
        axes=axis_names,
        time_index=int(run_time_index),
        channel_index=int(run_channel_index),
    )


def _scale_xyz_point(point_xyz: Sequence[float], scale_zyx: Sequence[float], to_run: bool) -> List[float]:
    """Compute scale xyz point."""
    x = float(point_xyz[0])
    y = float(point_xyz[1])
    z = float(point_xyz[2])
    sz, sy, sx = [float(v) for v in scale_zyx]

    if to_run:
        sx = sx if sx != 0 else 1.0
        sy = sy if sy != 0 else 1.0
        sz = sz if sz != 0 else 1.0
        return [x / sx, y / sy, z / sz]
    return [x * sx, y * sy, z * sz]


def scale_curve_nodes_xyz(points_xyz: Sequence[Sequence[float]], scale_zyx: Sequence[float], to_run: bool) -> List[List[float]]:
    """Compute scale curve nodes xyz."""
    return [_scale_xyz_point(point, scale_zyx, to_run=to_run) for point in points_xyz]


def _extract_zyx(point: Sequence[float]) -> Tuple[float, float, float]:
    """Compute extract zyx."""
    values = list(point)
    if len(values) < 3:
        raise ValueError("Point must have at least 3 coordinates.")
    if len(values) >= 5:
        return float(values[2]), float(values[3]), float(values[4])
    return float(values[-3]), float(values[-2]), float(values[-1])


def _replace_zyx(point: Sequence[float], z: float, y: float, x: float) -> List[float]:
    """Compute replace zyx."""
    values = [float(v) for v in point]
    if len(values) >= 5:
        values[2], values[3], values[4] = z, y, x
        return values
    if len(values) == 3:
        return [z, y, x]
    values[-3], values[-2], values[-1] = z, y, x
    return values


def scale_points_zyx(points: Sequence[Sequence[float]], scale_zyx: Sequence[float], to_run: bool) -> List[List[float]]:
    """Compute scale points zyx."""
    sz, sy, sx = [float(v) for v in scale_zyx]
    if sz == 0:
        sz = 1.0
    if sy == 0:
        sy = 1.0
    if sx == 0:
        sx = 1.0

    converted: List[List[float]] = []
    for point in points:
        z, y, x = _extract_zyx(point)
        if to_run:
            z_new, y_new, x_new = z / sz, y / sy, x / sx
        else:
            z_new, y_new, x_new = z * sz, y * sy, x * sx
        converted.append(_replace_zyx(point, z_new, y_new, x_new))
    return converted


XY_SCALED_PARAMS = {
    "diameter",
    "jitter",
    "adapt_diam_lower",
    "adapt_diam_upper",
    "dim",
}

XYZ_SCALED_PARAMS = {
    "stepsize",
    "break_distance",
    "resample_step_size",
    "bktk_search_radius",
}

INTEGER_PARAMS = {
    "jitter",
    "adapt_diam_lower",
    "adapt_diam_upper",
    "dim",
    "stepsize",
    "break_distance",
    "resample_step_size",
    "bktk_search_radius",
}


def _scaled_value(value: Any, factor: float, force_int: bool) -> Any:
    """Compute scaled value."""
    try:
        scaled = float(value) / float(factor)
    except Exception:
        return value

    if force_int:
        return max(1, int(round(scaled)))
    return float(scaled)


def scale_parameter_dict_for_level(parameters: Dict[str, Any], scale_zyx: Sequence[float]) -> Dict[str, Any]:
    """Compute scale parameter dict for level."""
    sz, sy, sx = [float(v) for v in scale_zyx]
    s_xy = (sy + sx) / 2.0 if (sy + sx) > 0 else 1.0
    s_xyz = (sz + sy + sx) / 3.0 if (sz + sy + sx) > 0 else 1.0

    scaled = dict(parameters)
    for key in XY_SCALED_PARAMS:
        if key not in scaled:
            continue
        scaled[key] = _scaled_value(scaled[key], s_xy, key in INTEGER_PARAMS)

    for key in XYZ_SCALED_PARAMS:
        if key not in scaled:
            continue
        scaled[key] = _scaled_value(scaled[key], s_xyz, key in INTEGER_PARAMS)

    return scaled


def apply_parameter_scaling_to_trace(trace: Any) -> None:
    """Apply parameter scaling to trace."""
    if not bool(getattr(trace, "auto_scale_for_level", True)):
        return
    if bool(getattr(trace, "_run_level_scaling_applied", False)):
        return

    scale_zyx = getattr(trace, "run_level_scale_zyx", DEFAULT_SCALE_ZYX)
    try:
        all_ones = all(abs(float(v) - 1.0) < 1e-9 for v in scale_zyx)
    except Exception:
        all_ones = True
    if all_ones:
        setattr(trace, "_run_level_scaling_applied", True)
        return

    param_subset = {key: getattr(trace, key) for key in XY_SCALED_PARAMS.union(XYZ_SCALED_PARAMS) if hasattr(trace, key)}
    scaled = scale_parameter_dict_for_level(param_subset, scale_zyx)
    for key, value in scaled.items():
        setattr(trace, key, value)
    setattr(trace, "_run_level_scaling_applied", True)
