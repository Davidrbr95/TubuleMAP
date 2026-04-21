from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget


DEFAULT_DOWNSAMPLE_FACTOR = 4.0
_DOWNSAMPLE_STATE_BY_VIEWER_ID = {}


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
    """Convert downsample points."""
    converted = []
    for point in points:
        values = list(point)
        if len(values) < 3:
            continue
        if len(values) >= 5:
            # Already carries explicit T,C,Z,Y,X coordinates.
            converted.append(
                [float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])]
            )
            continue
        z, y, x = float(values[-3]), float(values[-2]), float(values[-1])
        converted.append([0.0, 0.0, z, y / factor, x / factor])
    return converted


def to_original_points(points, factor):
    """Convert original points."""
    converted = []
    for point in points:
        values = list(point)
        if len(values) < 3:
            continue
        if len(values) >= 5:
            z, y, x = float(values[2]), float(values[3]), float(values[4])
            converted.append([z, y * factor, x * factor])
            continue
        z, y, x = float(values[-3]), float(values[-2]), float(values[-1])
        converted.append([z, y, x])
    return converted


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
