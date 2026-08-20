from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget


DEFAULT_DOWNSAMPLE_FACTOR = 4.0
_DOWNSAMPLE_STATE_BY_VIEWER_ID = {}

POINT_AXES_KEY = "tubulemap_point_axes"
ORIGINAL_POINT_AXES_KEY = "tubulemap_original_point_axes"
POINT_SIDECAR_KEY = "tubulemap_point_sidecar"


def _ensure_downsample_state(viewer):
    """Ensure downsample state."""
    viewer_id = id(viewer)
    state = _DOWNSAMPLE_STATE_BY_VIEWER_ID.get(viewer_id)
    if not isinstance(state, dict):
        state = {"enabled": False, "factor": DEFAULT_DOWNSAMPLE_FACTOR}
        _DOWNSAMPLE_STATE_BY_VIEWER_ID[viewer_id] = state

    enabled = bool(state.get("enabled", False))
    factor = float(state.get("factor", DEFAULT_DOWNSAMPLE_FACTOR))
    if factor <= 0:
        factor = DEFAULT_DOWNSAMPLE_FACTOR

    state["enabled"] = enabled
    state["factor"] = factor
    return state


def is_downsample_enabled(viewer):
    """Return whether downsample is enabled."""
    return _ensure_downsample_state(viewer)["enabled"]


def get_downsample_factor(viewer):
    """Get downsample factor."""
    return _ensure_downsample_state(viewer)["factor"]


def set_downsample_enabled(viewer, enabled):
    """Set downsample enabled."""
    _ensure_downsample_state(viewer)["enabled"] = bool(enabled)


def set_downsample_factor(viewer, factor):
    """Set downsample factor."""
    factor = float(factor)
    if factor <= 0:
        factor = DEFAULT_DOWNSAMPLE_FACTOR
    _ensure_downsample_state(viewer)["factor"] = factor


def to_downsample_points(points, factor):
    """Convert canonical Z,Y,X points to downsampled coordinates."""
    factor = float(factor)
    if factor <= 0:
        raise ValueError("Downsample factor must be greater than 0.")

    converted = []
    for point in points:
        values = list(point)
        if len(values) < 3:
            raise ValueError("Point must contain Z, Y and X coordinates.")
        z, y, x = float(values[-3]), float(values[-2]), float(values[-1])
        converted.append([z / factor, y / factor, x / factor])
    return converted


def to_original_points(points, factor):
    """Restore canonical downsampled Z,Y,X points to original coordinates."""
    factor = float(factor)
    if factor <= 0:
        raise ValueError("Downsample factor must be greater than 0.")

    converted = []
    for point in points:
        values = list(point)
        if len(values) < 3:
            raise ValueError("Point must contain Z, Y and X coordinates.")
        z, y, x = float(values[-3]), float(values[-2]), float(values[-1])
        converted.append([z * factor, y * factor, x * factor])
    return converted


def split_points_for_3d_display(points, point_axes=None):
    """Return canonical Z,Y,X points plus data needed to restore each original row."""
    rows = [list(point) for point in points]
    if not rows:
        axes = list(point_axes) if point_axes else ["z", "y", "x"]
        return [], axes, []

    point_dim = len(rows[0])
    if any(len(row) != point_dim for row in rows):
        raise ValueError("All point rows must have the same dimensionality.")

    if point_axes is None:
        if point_dim == 3:
            axes = ["z", "y", "x"]
        elif point_dim == 5:
            axes = ["t", "c", "z", "y", "x"]
        else:
            raise ValueError("Points must use 3D ZYX or declare point_axes.")
    else:
        axes = [str(axis).strip().lower() for axis in point_axes]

    if len(axes) != point_dim:
        raise ValueError("point_axes length must match the point row dimensionality.")
    if not all(axis in axes for axis in ("z", "y", "x")):
        raise ValueError("point_axes must contain z, y and x.")

    spatial_indices = {axis: axes.index(axis) for axis in ("z", "y", "x")}
    spatial_index_set = set(spatial_indices.values())
    display_points = []
    sidecar = []
    for row in rows:
        display_points.append(
            [
                float(row[spatial_indices["z"]]),
                float(row[spatial_indices["y"]]),
                float(row[spatial_indices["x"]]),
            ]
        )
        sidecar.append(
            {
                str(index): float(value)
                for index, value in enumerate(row)
                if index not in spatial_index_set
            }
        )
    return display_points, axes, sidecar


def restore_points_from_3d_display(points, original_axes, sidecar=None):
    """Rebuild original-dimensional point rows from canonical Z,Y,X points."""
    axes = [str(axis).strip().lower() for axis in original_axes]
    if not all(axis in axes for axis in ("z", "y", "x")):
        raise ValueError("original_axes must contain z, y and x.")

    sidecar = list(sidecar or [])
    restored = []
    for row_index, point in enumerate(points):
        values = list(point)
        if len(values) < 3:
            raise ValueError("Point must contain Z, Y and X coordinates.")
        z, y, x = [float(value) for value in values[-3:]]
        spatial = {"z": z, "y": y, "x": x}
        extra = sidecar[row_index] if row_index < len(sidecar) else {}
        row = []
        for index, axis in enumerate(axes):
            if axis in spatial:
                row.append(spatial[axis])
            else:
                row.append(float(extra.get(str(index), 0.0)))
        restored.append(row)
    return restored


class DownsampleControlWidget(QWidget):
    """Global downsample settings shared by point editing widgets."""

    def __init__(self, viewer):
        """Initialize the instance state."""
        super().__init__()
        self.viewer = viewer
        self.name = "Downsample Settings"

        set_downsample_enabled(self.viewer, is_downsample_enabled(self.viewer))
        set_downsample_factor(self.viewer, get_downsample_factor(self.viewer))

        self.enable_checkbox = QCheckBox("Use downsample data")
        self.enable_checkbox.setChecked(is_downsample_enabled(self.viewer))

        self.factor_label = QLabel("Factor")
        self.factor_spinbox = QDoubleSpinBox()
        self.factor_spinbox.setDecimals(3)
        self.factor_spinbox.setRange(0.001, 9999.0)
        self.factor_spinbox.setSingleStep(0.5)
        self.factor_spinbox.setValue(get_downsample_factor(self.viewer))

        row = QHBoxLayout()
        row.addWidget(self.enable_checkbox)
        row.addWidget(self.factor_label)
        row.addWidget(self.factor_spinbox)
        self.setLayout(row)

        self.enable_checkbox.toggled.connect(self._on_enable_toggled)
        self.factor_spinbox.valueChanged.connect(self._on_factor_changed)
        self._sync_factor_enabled(self.enable_checkbox.isChecked())

    def _on_enable_toggled(self, checked):
        """Update shared downsample enable state and sync control availability."""
        set_downsample_enabled(self.viewer, checked)
        self._sync_factor_enabled(checked)

    def _on_factor_changed(self, value):
        """Update the shared downsample factor value."""
        set_downsample_factor(self.viewer, value)

    def _sync_factor_enabled(self, enabled):
        """Compute sync factor enabled."""
        self.factor_label.setEnabled(enabled)
        self.factor_spinbox.setEnabled(enabled)
