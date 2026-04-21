import os
import json
import re
import shutil
from itertools import permutations
from pathlib import Path
import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QMessageBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QProgressBar,
)
from tubulemap.cellpose_tracker import tracking
from tubulemap.cellpose_tracker.parameters import ALL_PARAMETERS
from napari.qt.threading import thread_worker
import napari
from tubulemap.utils.zarr_resolution import (
    get_axis_size_for_level,
    has_translation_mismatch,
    inspect_zarr_source,
)

@thread_worker
def start_run_trace(trace_parameters, should_cancel):
    """Start run trace."""
    trace_parameters['should_cancel'] = should_cancel
    yield from tracking.run_core_function(**trace_parameters)


@thread_worker
def start_generate_mesh(track_folder, volume_override):
    """Generate mesh for the selected track folder."""
    from tubulemap.cellpose_tracker.mesh_pipeline import generate_mesh_from_track_folder_events

    report = yield from generate_mesh_from_track_folder_events(
        track_folder=track_folder,
        volume_override=volume_override,
    )
    return report


@thread_worker
def start_generate_mask(obj_folder, volume_source_path, output_path):
    """Generate a dense label mask zarr from OBJ meshes."""
    from tubulemap.cellpose_tracker.mesh_to_mask_pipeline import generate_mask_from_obj_folder_events

    report = yield from generate_mask_from_obj_folder_events(
        obj_folder=obj_folder,
        volume_source_path=volume_source_path,
        output_path=output_path,
    )
    return report


class TubuleTrackerWidget(QWidget):
    _NO_POINTS_LAYER = "No points layers available"
    _NO_IMAGE_LAYER = "No image layers available"
    _RUN_DIR_PATTERN = re.compile(r"^Run_(\d+)$")
    _EMPTY_PATH_VALUES = {"", ".", "./"}
    _TRACKING_CORE_FIELDS = (
        "diameter",
        "use_adaptive_diameter",
        "stepsize",
        "iterations",
        "jitter",
        "use_rotations",
        "use_ultrack",
        "dim",
    )
    _TRACKING_TRACING_FIELDS = (
        "adapt_diam_lower",
        "adapt_diam_upper",
        "adapt_window",
        "scale_jitter",
        "scale_stepsize",
    )
    _TRACKING_FLOAT_FIELDS = {
        "diameter",
        "adapt_diam_lower",
        "adapt_diam_upper",
        "scale_jitter",
        "scale_stepsize",
    }
    _TRACKING_INT_FIELDS = {"stepsize", "iterations", "jitter", "dim", "adapt_window"}
    _TRACKING_BOOL_FIELDS = {"use_adaptive_diameter", "use_rotations", "use_ultrack"}
    _TRACKING_LABELS = {
        "dim": "Sampling Window Size (dim)",
        "iterations": "Iterations",
        "use_ultrack": "Enable Ultrack Troubleshooting",
        "adapt_diam_lower": "Adaptive Diameter Lower Bound",
        "adapt_diam_upper": "Adaptive Diameter Upper Bound",
        "adapt_window": "Adaptive Diameter Averaging Window",
        "scale_jitter": "Adaptive Jitter Scale Factor",
        "scale_stepsize": "Adaptive Step Size Scale Factor",
    }

    def __init__(self, viewer):
        """Initialize the instance state."""
        super().__init__()
        self.viewer = viewer
        self.name = 'Tubule Tracing Widget'
        self.layout_main = QVBoxLayout()
        self._current_source_meta = None

        self.step1_label = QLabel("Step 1. Trace Tubule")
        self.step1_label.setStyleSheet("font-weight: bold;")
        self.step1_hint = QLabel(
            "Choose starting points and image source, tune tracing parameters, then run and stop tracking as needed."
        )
        self.step1_hint.setWordWrap(True)
        self.layout_main.addWidget(self.step1_label)
        self.layout_main.addWidget(self.step1_hint)
        
        # Create dropdowns
        self.kp_source_label = QLabel("Starting points source")
        self.kp_source = QComboBox()
        self.kp_source.addItems(["JSON file", "Points layer"])
        self.kp_source.currentIndexChanged.connect(self.update_widgets)
        
        self.data_source_label = QLabel("Image data source")
        self.data_source = QComboBox()
        self.data_source.addItems(["Zarr folder path", "Image layer"])
        self.data_source.currentIndexChanged.connect(self.update_widgets)

        # Add compact source rows (label + control in one line)
        kp_row = QHBoxLayout()
        kp_row.addWidget(self.kp_source_label)
        kp_row.addWidget(self.kp_source, 1)
        self.layout_main.addLayout(kp_row)

        data_row = QHBoxLayout()
        data_row.addWidget(self.data_source_label)
        data_row.addWidget(self.data_source, 1)
        self.layout_main.addLayout(data_row)

        # Main-panel tracking control: iterations
        self.iterations_label = QLabel("Iterations")
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 100000)
        iterations_comment = self._parameter_comment("iterations")
        if iterations_comment:
            self.iterations_label.setToolTip(iterations_comment)
            self.iterations_spinbox.setToolTip(iterations_comment)
        iterations_row = QHBoxLayout()
        iterations_row.addWidget(self.iterations_label)
        iterations_row.addWidget(self.iterations_spinbox, 1)
        self.layout_main.addLayout(iterations_row)
        
        # Add the run_trace magicgui widget
        self.run_trace_widget = tracking.run_trace
        if hasattr(self.run_trace_widget, "call_button"):
            self.run_trace_widget.call_button.hide()
        self.layout_main.addWidget(self.run_trace_widget.native)

        self._tracking_param_overrides = self._initial_tracking_parameter_values()
        self._sync_tracking_parameters_to_magicgui()
        self.iterations_spinbox.valueChanged.connect(self._on_iterations_spinbox_changed)
        self.parameters_button = QPushButton("Tracking Parameters...")
        self.parameters_button.clicked.connect(self.open_tracking_parameters_dialog)
        self.layout_main.addWidget(self.parameters_button)
        
        # Add Start and Stop buttons
        self.start_button = QPushButton("Start Tracking")
        self.stop_button = QPushButton("Stop Tracking")
        self.generate_mesh_button = QPushButton("Generate Mesh")
        self.stop_button.setEnabled(False)  # Initially disabled
        self.layout_main.addWidget(self.start_button)
        self.layout_main.addWidget(self.stop_button)

        self.step2_label = QLabel("Step 2. Generate Mesh")
        self.step2_label.setStyleSheet("font-weight: bold;")
        self.step2_hint = QLabel(
            "Build a closed tubule surface mesh (capped tube) from the final trace points and orthogonal HDF5 planes."
        )
        self.step2_hint.setWordWrap(True)
        self.layout_main.addWidget(self.step2_label)
        self.layout_main.addWidget(self.step2_hint)

        self.mesh_progress_label = QLabel("Mesh Progress")
        self.mesh_progress_bar = QProgressBar()
        self.mesh_progress_bar.setRange(0, 100)
        self.mesh_progress_bar.setValue(0)
        self.mesh_progress_bar.setFormat("%p%")
        self.mesh_progress_status = QLabel("Idle")
        self.mesh_progress_status.setWordWrap(True)
        self.layout_main.addWidget(self.mesh_progress_label)
        self.layout_main.addWidget(self.mesh_progress_bar)
        self.layout_main.addWidget(self.mesh_progress_status)

        self.layout_main.addWidget(self.generate_mesh_button)

        self.step2b_label = QLabel("Step 2.b Load OBJ")
        self.step2b_label.setStyleSheet("font-weight: bold;")
        self.step2b_hint = QLabel(
            "Load a generated OBJ mesh into napari for 3D visualization."
        )
        self.step2b_hint.setWordWrap(True)
        self.load_obj_button = QPushButton("Load OBJ")
        self.layout_main.addWidget(self.step2b_label)
        self.layout_main.addWidget(self.step2b_hint)
        self.layout_main.addWidget(self.load_obj_button)

        self.step2c_label = QLabel("Step 2.c Generate Mask")
        self.step2c_label.setStyleSheet("font-weight: bold;")
        self.step2c_hint = QLabel(
            "Convert OBJ meshes into a dense label mask zarr aligned to the selected loaded volume bounds."
        )
        self.step2c_hint.setWordWrap(True)
        self.generate_mask_button = QPushButton("Generate Mask")
        self.mask_progress_label = QLabel("Mask Progress")
        self.mask_progress_bar = QProgressBar()
        self.mask_progress_bar.setRange(0, 100)
        self.mask_progress_bar.setValue(0)
        self.mask_progress_bar.setFormat("%p%")
        self.mask_progress_status = QLabel("Idle")
        self.mask_progress_status.setWordWrap(True)
        self.layout_main.addWidget(self.step2c_label)
        self.layout_main.addWidget(self.step2c_hint)
        self.layout_main.addWidget(self.generate_mask_button)
        self.layout_main.addWidget(self.mask_progress_label)
        self.layout_main.addWidget(self.mask_progress_bar)
        self.layout_main.addWidget(self.mask_progress_status)
        
        self.setLayout(self.layout_main)
        self.update_kp_layer_choices()
        self.update_widgets()
        self._connect_run_control_events()
        self._sync_run_resolution_controls()
        
        # Connect signals
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.generate_mesh_button.clicked.connect(self.generate_mesh_from_track_folder)
        self.load_obj_button.clicked.connect(self.load_obj_into_viewer)
        self.generate_mask_button.clicked.connect(self.generate_mask_from_obj_folder)
        
        self.viewer.layers.events.inserted.connect(lambda event: self.update_kp_layer_choices())
        self.viewer.layers.events.removed.connect(lambda event: self.update_kp_layer_choices())
        
        self.worker = None  # Placeholder for the worker
        self.mesh_worker = None  # Worker for mesh generation
        self._mesh_result_handled = False
        self.mask_worker = None  # Worker for mesh-to-mask generation
        self._mask_result_handled = False
        self._should_cancel = False  # Cancellation flag

    def update_widgets(self):
        """Update widgets."""
        if self.kp_source.currentText() == "JSON file":
            self.run_trace_widget.kp_path.show()
            self.run_trace_widget.kp_layer.hide()
            self.run_trace_widget.kp_source.value = True
        else:
            self.run_trace_widget.kp_path.hide()
            self.run_trace_widget.kp_layer.show()
            self.run_trace_widget.kp_source.value = False

        if self.data_source.currentText() == "Zarr folder path":
            self.run_trace_widget.data_set_path.show()
            self.run_trace_widget.data_layer.hide()
            self.run_trace_widget.data_source.value = True
        else:
            self.run_trace_widget.data_set_path.hide()
            self.run_trace_widget.data_layer.show()
            self.run_trace_widget.data_source.value = False
        self._sync_run_resolution_controls()

    def update_kp_layer_choices(self):
        """Update kp layer choices."""
        layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Points)]
        if layer_names:
            self.run_trace_widget.kp_layer.choices = layer_names
        else:
            self.run_trace_widget.kp_layer.choices = [self._NO_POINTS_LAYER]
                
        layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        if layer_names:
            self.run_trace_widget.data_layer.choices = layer_names
        else:
            self.run_trace_widget.data_layer.choices = [self._NO_IMAGE_LAYER]
        self._sync_run_resolution_controls()

    def _connect_run_control_events(self):
        """Compute connect run control events."""
        for widget_name in ("data_set_path", "data_layer", "run_level"):
            widget = getattr(self.run_trace_widget, widget_name, None)
            if widget is not None and hasattr(widget, "changed"):
                widget.changed.connect(lambda _=None: self._sync_run_resolution_controls())

    def _safe_inspect_source(self, source_path):
        """Compute safe inspect source."""
        try:
            return inspect_zarr_source(source_path)
        except Exception:
            return None

    def _infer_source_meta_from_layer(self, layer):
        """Infer source meta from layer."""
        metadata = getattr(layer, "metadata", None)
        if isinstance(metadata, dict):
            cached = metadata.get("tubulemap_source_resolution")
            if isinstance(cached, dict) and cached.get("levels"):
                return cached

        source_path = getattr(layer.source, "path", None)
        if source_path in (None, ""):
            return None
        source_path = str(source_path).rstrip("/\\")

        candidates = []
        if source_path:
            candidates.append(source_path)
            parent_1 = str(Path(source_path).parent)
            parent_2 = str(Path(parent_1).parent)
            candidates.extend([parent_1, parent_2])

        seen = set()
        resolved = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not os.path.exists(candidate):
                continue
            meta = self._safe_inspect_source(candidate)
            if meta is not None:
                resolved.append(meta)

        if not resolved:
            return None

        resolved.sort(
            key=lambda meta: (
                1 if meta.get("source_kind") == "ome" else 0,
                int(len(meta.get("levels", []))),
            ),
            reverse=True,
        )
        chosen = resolved[0]
        if isinstance(metadata, dict):
            metadata["tubulemap_source_resolution"] = chosen
        return chosen

    def _resolve_current_source_meta(self):
        """Resolve current source meta."""
        if self.data_source.currentText() == "Zarr folder path":
            source_path = str(self.run_trace_widget.data_set_path.value).strip()
            if source_path in self._EMPTY_PATH_VALUES or not os.path.exists(source_path):
                return None
            return self._safe_inspect_source(source_path)

        data_layer = str(self.run_trace_widget.data_layer.value).strip()
        if not data_layer or data_layer == self._NO_IMAGE_LAYER or data_layer not in self.viewer.layers:
            return None
        return self._infer_source_meta_from_layer(self.viewer.layers[data_layer])

    def _set_spinbox_bounds(self, widget, upper_bound):
        """Set spinbox bounds."""
        if widget is None:
            return
        if hasattr(widget, "min"):
            widget.min = 0
        if hasattr(widget, "max"):
            widget.max = max(0, int(upper_bound))

    def _sync_run_resolution_controls(self):
        """Compute sync run resolution controls."""
        source_meta = self._resolve_current_source_meta()
        self._current_source_meta = source_meta

        run_level_widget = getattr(self.run_trace_widget, "run_level", None)
        run_time_widget = getattr(self.run_trace_widget, "run_time_index", None)
        run_channel_widget = getattr(self.run_trace_widget, "run_channel_index", None)

        if source_meta is None:
            if run_level_widget is not None:
                run_level_widget.choices = [0]
                run_level_widget.value = 0
            if run_time_widget is not None:
                run_time_widget.value = 0
                run_time_widget.hide()
            if run_channel_widget is not None:
                run_channel_widget.value = 0
                run_channel_widget.hide()
            return

        levels = source_meta.get("levels", [])
        level_choices = list(range(len(levels))) if levels else [0]
        if run_level_widget is not None:
            run_level_widget.choices = level_choices
            current_value = int(run_level_widget.value) if str(run_level_widget.value).strip() else 0
            if current_value not in level_choices:
                run_level_widget.value = level_choices[0]

        selected_level = int(run_level_widget.value) if run_level_widget is not None else 0
        t_size = get_axis_size_for_level(source_meta, selected_level, "t")
        c_size = get_axis_size_for_level(source_meta, selected_level, "c")

        if run_time_widget is not None:
            if t_size is None:
                run_time_widget.value = 0
                run_time_widget.hide()
            else:
                self._set_spinbox_bounds(run_time_widget, t_size - 1)
                if int(run_time_widget.value) >= t_size:
                    run_time_widget.value = 0
                run_time_widget.show()

        if run_channel_widget is not None:
            if c_size is None:
                run_channel_widget.value = 0
                run_channel_widget.hide()
            else:
                self._set_spinbox_bounds(run_channel_widget, c_size - 1)
                if int(run_channel_widget.value) >= c_size:
                    run_channel_widget.value = 0
                run_channel_widget.show()

    def _normalize_widget_params(self, params):
        """Normalize widget params."""
        for key in ("kp_path", "data_set_path", "save_dir"):
            if key in params and params[key] is not None:
                params[key] = str(params[key]).strip()
        for key in ("run_level", "run_time_index", "run_channel_index"):
            if key in params:
                params[key] = int(params.get(key, 0) or 0)
        params["auto_scale_for_level"] = bool(params.get("auto_scale_for_level", True))

        if not str(params.get("name", "")).strip():
            if params.get("kp_source", True):
                kp_path = params.get("kp_path", "")
                params["name"] = Path(kp_path).stem if kp_path not in self._EMPTY_PATH_VALUES else "trace_job"
            else:
                layer_name = str(params.get("kp_layer", "")).strip()
                params["name"] = layer_name if layer_name and layer_name != self._NO_POINTS_LAYER else "trace_job"

        return params

    def _tracking_parameter_fields(self):
        """Compute tracking parameter fields."""
        seen = set()
        ordered = []
        for key in list(self._TRACKING_CORE_FIELDS) + list(self._TRACKING_TRACING_FIELDS):
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def _parameter_comment(self, key):
        """Compute parameter comment."""
        info = ALL_PARAMETERS.get(key, {})
        return str(info.get("comment", "")).strip()

    def _parameter_label(self, key):
        """Compute parameter label."""
        if key in self._TRACKING_LABELS:
            return self._TRACKING_LABELS[key]
        return key.replace("_", " ").title()

    def _default_tracking_parameter_values(self):
        """Build the default tracking parameter values."""
        defaults = {}
        for key in self._tracking_parameter_fields():
            defaults[key] = ALL_PARAMETERS.get(key, {}).get("default")
        return defaults

    def _initial_tracking_parameter_values(self):
        """Compute initial tracking parameter values."""
        values = self._default_tracking_parameter_values()
        for key in self._tracking_parameter_fields():
            widget = getattr(self.run_trace_widget, key, None)
            if widget is not None and hasattr(widget, "value"):
                values[key] = widget.value
        return values

    def _sync_tracking_parameters_to_magicgui(self):
        """Compute sync tracking parameters to magicgui."""
        for key, value in self._tracking_param_overrides.items():
            widget = getattr(self.run_trace_widget, key, None)
            if widget is not None and hasattr(widget, "value"):
                widget.value = value
        self._sync_iterations_control_from_overrides()

    def _sync_iterations_control_from_overrides(self):
        """Keep the main-panel iterations spinbox aligned with tracking overrides."""
        default_iterations = int(ALL_PARAMETERS.get("iterations", {}).get("default", 20))
        iterations_value = int(self._tracking_param_overrides.get("iterations", default_iterations))
        self.iterations_spinbox.blockSignals(True)
        self.iterations_spinbox.setValue(iterations_value)
        self.iterations_spinbox.blockSignals(False)

    def _on_iterations_spinbox_changed(self, value):
        """Apply main-panel iterations changes to the active tracking parameter set."""
        value = int(value)
        self._tracking_param_overrides["iterations"] = value
        widget = getattr(self.run_trace_widget, "iterations", None)
        if widget is not None and hasattr(widget, "value"):
            widget.value = value

    def _build_tracking_parameter_editor(self, key, value):
        """Compute build tracking parameter editor."""
        if key in self._TRACKING_BOOL_FIELDS:
            editor = QCheckBox()
            editor.setChecked(bool(value))
            return editor

        if key in self._TRACKING_INT_FIELDS:
            editor = QSpinBox()
            if key == "dim":
                editor.setRange(8, 4096)
            elif key == "stepsize":
                editor.setRange(1, 500)
            elif key == "iterations":
                editor.setRange(1, 100000)
            elif key == "jitter":
                editor.setRange(0, 500)
            elif key == "adapt_window":
                editor.setRange(1, 500)
            else:
                editor.setRange(-1000000, 1000000)
            editor.setValue(int(value))
            return editor

        if key in self._TRACKING_FLOAT_FIELDS:
            editor = QDoubleSpinBox()
            editor.setDecimals(4)
            editor.setSingleStep(0.5)
            if key == "diameter":
                editor.setRange(0.01, 5000.0)
            elif key in {"adapt_diam_lower", "adapt_diam_upper"}:
                editor.setRange(0.0, 5000.0)
            elif key in {"scale_jitter", "scale_stepsize"}:
                editor.setRange(0.01, 1000.0)
            else:
                editor.setRange(-1000000.0, 1000000.0)
            editor.setValue(float(value))
            return editor

        # Fallback for any future fields.
        editor = QDoubleSpinBox()
        editor.setDecimals(4)
        editor.setRange(-1000000.0, 1000000.0)
        editor.setValue(float(value))
        return editor

    def _editor_value(self, editor):
        """Compute editor value."""
        if isinstance(editor, QCheckBox):
            return bool(editor.isChecked())
        if isinstance(editor, QSpinBox):
            return int(editor.value())
        if isinstance(editor, QDoubleSpinBox):
            return float(editor.value())
        return None

    def _populate_tracking_parameter_editors(self, editors, values):
        """Compute populate tracking parameter editors."""
        for key, editor in editors.items():
            value = values.get(key)
            if isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QSpinBox):
                editor.setValue(int(value))
            elif isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(value))

    def _validate_tracking_parameter_values(self, params):
        """Compute validate tracking parameter values."""
        if float(params.get("diameter", 0)) <= 0:
            return "Diameter must be greater than 0."
        if int(params.get("stepsize", 0)) <= 0:
            return "Step size must be greater than 0."
        if int(params.get("iterations", 0)) <= 0:
            return "Iterations must be greater than 0."
        if int(params.get("jitter", 0)) < 0:
            return "Jitter must be greater than or equal to 0."
        if int(params.get("dim", 0)) <= 0:
            return "Sampling window size (dim) must be greater than 0."
        if int(params.get("adapt_window", 0)) <= 0:
            return "Adaptive diameter averaging window must be greater than 0."
        if float(params.get("adapt_diam_lower", 0)) < 0:
            return "Adaptive diameter lower bound must be greater than or equal to 0."
        if float(params.get("adapt_diam_upper", 0)) < float(params.get("adapt_diam_lower", 0)):
            return "Adaptive diameter upper bound must be greater than or equal to the lower bound."
        if float(params.get("scale_jitter", 0)) <= 0:
            return "Adaptive jitter scale factor must be greater than 0."
        if float(params.get("scale_stepsize", 0)) <= 0:
            return "Adaptive step size scale factor must be greater than 0."
        return None

    def open_tracking_parameters_dialog(self):
        """Open tracking parameters dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Tracking Parameters")
        dialog_layout = QVBoxLayout(dialog)

        info = QLabel(
            "Adjust tracking behavior parameters for this run. "
            "Hover over each parameter name to see its description."
        )
        info.setWordWrap(True)
        dialog_layout.addWidget(info)

        editors = {}
        section_specs = [
            ("Core Tracking", self._TRACKING_CORE_FIELDS),
            ("Tracing Parameters", self._TRACKING_TRACING_FIELDS),
        ]

        for section_title, fields in section_specs:
            group = QGroupBox(section_title)
            form_layout = QFormLayout(group)
            for key in fields:
                label = QLabel(self._parameter_label(key))
                comment = self._parameter_comment(key)
                if comment:
                    label.setToolTip(comment)
                editor = self._build_tracking_parameter_editor(
                    key=key,
                    value=self._tracking_param_overrides.get(key),
                )
                if comment:
                    editor.setToolTip(comment)
                form_layout.addRow(label, editor)
                editors[key] = editor
            dialog_layout.addWidget(group)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        defaults_button = button_box.addButton("Restore Defaults", QDialogButtonBox.ResetRole)
        defaults_button.clicked.connect(
            lambda: self._populate_tracking_parameter_editors(editors, self._default_tracking_parameter_values())
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        updated = {key: self._editor_value(editor) for key, editor in editors.items()}
        validation_error = self._validate_tracking_parameter_values(updated)
        if validation_error:
            QMessageBox.warning(self, "Invalid Tracking Parameter", validation_error)
            return

        self._tracking_param_overrides.update(updated)
        self._sync_tracking_parameters_to_magicgui()

    def _validate_params(self, params):
        """Compute validate params."""
        if params.get("kp_source", True):
            kp_path = params.get("kp_path", "")
            if kp_path in self._EMPTY_PATH_VALUES or not os.path.isfile(kp_path):
                return "Choose a valid Starting Points JSON file."
            try:
                with open(kp_path, "r") as fp:
                    payload = json.load(fp)
                points = payload.get("points", []) if isinstance(payload, dict) else []
                if len(points) < 2:
                    return "Starting points JSON must contain at least 2 points."
            except Exception:
                return "Starting points JSON could not be read."
        else:
            kp_layer = str(params.get("kp_layer", "")).strip()
            if not kp_layer or kp_layer == self._NO_POINTS_LAYER or kp_layer not in self.viewer.layers:
                return "Choose a valid points layer from the viewer."
            kp_layer_obj = self.viewer.layers[kp_layer]
            if len(kp_layer_obj.data) == 0:
                return "The selected points layer is empty."
            if len(kp_layer_obj.data) < 2:
                return "The selected points layer must contain at least 2 points."
            first_point = kp_layer_obj.data[0]
            if len(first_point) < 3:
                return "Points layer must contain points as [z,y,x] or [t,c,z,y,x]."

        if params.get("data_source", True):
            data_set_path = params.get("data_set_path", "")
            if data_set_path in self._EMPTY_PATH_VALUES or not os.path.isdir(data_set_path):
                return "Choose a valid image zarr folder."
        else:
            data_layer = str(params.get("data_layer", "")).strip()
            if not data_layer or data_layer == self._NO_IMAGE_LAYER or data_layer not in self.viewer.layers:
                return "Choose a valid image layer from the viewer."
            layer_obj = self.viewer.layers[data_layer]
            source_meta = self._infer_source_meta_from_layer(layer_obj)
            source_path = getattr(layer_obj.source, "path", None)
            if source_meta is None and source_path in (None, ""):
                return (
                    "The selected image layer has no resolvable zarr metadata or source path. "
                    "Use 'Zarr folder path' mode for this layer."
                )
            if source_path not in (None, ""):
                source_path = str(source_path).rstrip("/\\")
                if source_path and not os.path.exists(source_path):
                    return "The selected image layer path no longer exists on disk."

        save_dir = params.get("save_dir", "")
        if save_dir in self._EMPTY_PATH_VALUES:
            return "Choose an output folder."
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            return "Output folder could not be created."

        source_meta = self._current_source_meta or self._resolve_current_source_meta()
        if source_meta is None:
            return (
                "Could not resolve zarr source metadata. "
                "Use 'Zarr folder path' mode or choose an image layer loaded from a zarr path."
            )

        run_level = int(params.get("run_level", 0))
        level_count = len(source_meta.get("levels", []))
        if run_level < 0 or run_level >= level_count:
            return f"Run level must be in range [0, {max(0, level_count - 1)}]."

        if run_level > 0 and has_translation_mismatch(source_meta, level_idx=run_level):
            return (
                "Selected run level has a translation mismatch across levels. "
                "Please choose run level 0 for this source."
            )

        t_size = get_axis_size_for_level(source_meta, run_level, "t")
        c_size = get_axis_size_for_level(source_meta, run_level, "c")
        run_time_index = int(params.get("run_time_index", 0))
        run_channel_index = int(params.get("run_channel_index", 0))
        if t_size is not None and not (0 <= run_time_index < t_size):
            return f"Run Time Index must be in range [0, {t_size-1}]."
        if c_size is not None and not (0 <= run_channel_index < c_size):
            return f"Run Channel Index must be in range [0, {c_size-1}]."

        parameter_validation_error = self._validate_tracking_parameter_values(params)
        if parameter_validation_error:
            return parameter_validation_error

        return None

    def start_tracking(self):
        """Start tracking."""
        params = self._normalize_widget_params(self.run_trace_widget.asdict())
        params.update(self._tracking_param_overrides)
        validation_error = self._validate_params(params)
        if validation_error:
            QMessageBox.warning(self, "Missing Input", validation_error)
            return

        # Disable the start button and enable the stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_trace_widget.native.setEnabled(False)
        self._should_cancel = False

        # Create the should_cancel callback
        def should_cancel():
            """Return whether cancel is requested."""
            return self._should_cancel

        # Start the worker
        self.worker = start_run_trace(params, should_cancel)
        self.worker.yielded.connect(self.update_viewer)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.start()

    def stop_tracking(self):
        """Stop tracking."""
        if self.worker is not None:
            self._should_cancel = True  # Signal cancellation
            # self.worker.quit()  # Stop the worker thread
            self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_trace_widget.native.setEnabled(True)

    def on_tracking_finished(self):
        """Reset controls when the background tracking worker finishes."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_trace_widget.native.setEnabled(True)
        self._should_cancel = False
        self.worker = None

    def _resolve_volume_override_for_mesh(self):
        """Best-effort image path for mesh-plane filling when run parameters are incomplete."""
        source_meta = self._current_source_meta or self._resolve_current_source_meta()
        if isinstance(source_meta, dict):
            source_path = str(source_meta.get("path", "")).strip()
            if source_path and os.path.exists(source_path):
                return source_path

        data_set_path = str(getattr(self.run_trace_widget.data_set_path, "value", "")).strip()
        if data_set_path not in self._EMPTY_PATH_VALUES and os.path.exists(data_set_path):
            return data_set_path

        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Image):
                continue
            layer_meta = self._infer_source_meta_from_layer(layer)
            if isinstance(layer_meta, dict):
                layer_path = str(layer_meta.get("path", "")).strip()
                if layer_path and os.path.exists(layer_path):
                    return layer_path
            source_path = getattr(layer.source, "path", None)
            if source_path not in (None, ""):
                source_path = str(source_path).rstrip("/\\")
                if source_path and os.path.exists(source_path):
                    return source_path

        return None

    def generate_mesh_from_track_folder(self):
        """Open a track folder chooser and build a mesh from the latest run."""
        if self.mesh_worker is not None:
            QMessageBox.information(self, "Mesh Generation In Progress", "A mesh job is already running.")
            return

        track_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Track Folder (contains Run_* folders)",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not track_folder:
            return
        track_path = Path(track_folder)
        run_dirs = [
            entry
            for entry in track_path.iterdir()
            if entry.is_dir() and self._RUN_DIR_PATTERN.match(entry.name)
        ]
        if not run_dirs:
            self._update_mesh_progress(0, "No Run_* folders found in selected track folder.")
            QMessageBox.warning(
                self,
                "Invalid Track Folder",
                f"No Run_* folders found in:\n{track_folder}",
            )
            return

        volume_override = self._resolve_volume_override_for_mesh()
        self.generate_mesh_button.setEnabled(False)
        self._mesh_result_handled = False
        self._update_mesh_progress(0, "Starting mesh generation...")

        self.mesh_worker = start_generate_mesh(track_folder, volume_override)
        if hasattr(self.mesh_worker, "yielded"):
            self.mesh_worker.yielded.connect(self._on_mesh_generation_yielded)
        if hasattr(self.mesh_worker, "returned"):
            self.mesh_worker.returned.connect(self._on_mesh_generation_returned)
        if hasattr(self.mesh_worker, "errored"):
            self.mesh_worker.errored.connect(self._on_mesh_generation_error)
        self.mesh_worker.finished.connect(self._on_mesh_generation_finished)
        self.mesh_worker.start()

    def _update_mesh_progress(self, progress, message):
        """Update the mesh progress bar and status text."""
        progress_value = int(max(0, min(100, int(progress))))
        self.mesh_progress_bar.setValue(progress_value)
        self.mesh_progress_status.setText(str(message))

    def _on_mesh_generation_yielded(self, payload):
        """Handle incremental mesh-generation progress events."""
        if not isinstance(payload, dict):
            return
        payload_type = payload.get("type")
        if payload_type == "progress":
            self._update_mesh_progress(payload.get("progress", 0), payload.get("message", "Working..."))
        elif payload_type == "result":
            self._on_mesh_generation_returned(payload.get("report"))

    def _on_mesh_generation_returned(self, report):
        """Process mesh result exactly once."""
        if self._mesh_result_handled:
            return
        self._mesh_result_handled = True
        self._on_mesh_generation_result(report)

    def _on_mesh_generation_result(self, report):
        """Display mesh-generation summary."""
        if not isinstance(report, dict):
            QMessageBox.warning(self, "Mesh Generation", "Mesh generation finished with an unexpected result.")
            return

        mesh_output = report.get("mesh_output", "")
        report_path = report.get("report_path", "")
        missing_before = report.get("missing_indices_before", [])
        generated = report.get("generated_indices", [])
        missing_after = report.get("missing_indices_after", [])
        failed = report.get("failed_indices", [])

        summary = (
            f"Mesh generated:\n{mesh_output}\n\n"
            f"Report:\n{report_path}\n\n"
            f"Missing masks before: {len(missing_before)}\n"
            f"Generated masks: {len(generated)}\n"
            f"Missing masks after: {len(missing_after)}\n"
            f"Failed fills: {len(failed)}"
        )
        self._update_mesh_progress(100, "Mesh generation complete.")
        QMessageBox.information(self, "Mesh Generation Complete", summary)

    def _on_mesh_generation_error(self, error):
        """Display mesh-generation errors."""
        if isinstance(error, tuple) and error:
            error_text = str(error[0])
        else:
            error_text = str(error)
        self._update_mesh_progress(self.mesh_progress_bar.value(), f"Mesh generation failed: {error_text}")
        self.generate_mesh_button.setEnabled(True)
        self.mesh_worker = None
        QMessageBox.critical(self, "Mesh Generation Failed", error_text)

    def _on_mesh_generation_finished(self):
        """Reset mesh-generation button state."""
        self.generate_mesh_button.setEnabled(True)
        self.mesh_worker = None

    def _list_loaded_volume_candidates(self):
        """List loaded image layers that can provide a resolvable source path."""
        candidates = []
        seen_paths = set()

        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Image):
                continue

            source_path = None
            layer_meta = self._infer_source_meta_from_layer(layer)
            if isinstance(layer_meta, dict):
                meta_path = str(layer_meta.get("path", "")).strip()
                if meta_path and os.path.exists(meta_path):
                    source_path = meta_path

            if source_path is None:
                raw_source = getattr(layer.source, "path", None)
                if raw_source not in (None, ""):
                    raw_source = str(raw_source).rstrip("/\\")
                    if raw_source and os.path.exists(raw_source):
                        source_path = raw_source

            if source_path is None:
                continue

            resolved = str(Path(source_path).expanduser().resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            candidates.append(
                {
                    "layer_name": str(layer.name),
                    "path": resolved,
                }
            )

        return candidates

    def _choose_volume_source_for_mask(self):
        """Choose which loaded volume defines bounds for mesh-to-mask generation."""
        candidates = self._list_loaded_volume_candidates()
        if not candidates:
            message = "No volume loaded; load a volume first to define mask bounds."
            self._update_mask_progress(0, message)
            QMessageBox.warning(self, "Missing Volume", message)
            return None

        if len(candidates) == 1:
            return candidates[0]["path"]

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Volume for Mask Bounds")
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Multiple volumes are loaded. Select which volume should define "
                "the mask output bounds."
            )
        )
        combo = QComboBox(dialog)
        for candidate in candidates:
            label = f"{candidate['layer_name']}  ({candidate['path']})"
            combo.addItem(label, userData=candidate["path"])
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return str(combo.currentData())

    def _prepare_mask_output_path(self, output_path):
        """Prompt before overwrite and clear existing output store when confirmed."""
        output = Path(output_path)
        if not output.exists():
            return True

        answer = QMessageBox.question(
            self,
            "Output Already Exists",
            f"Mask output already exists:\n{output}\n\nOverwrite it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False

        try:
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Overwrite Failed",
                f"Could not remove existing output:\n{output}\n\n{exc}",
            )
            return False
        return True

    def generate_mask_from_obj_folder(self):
        """Choose an OBJ folder and generate a dense labels zarr mask."""
        if self.mask_worker is not None:
            QMessageBox.information(self, "Mask Generation In Progress", "A mask job is already running.")
            return

        obj_folder = QFileDialog.getExistingDirectory(
            self,
            "Select OBJ Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if not obj_folder:
            return

        volume_source_path = self._choose_volume_source_for_mask()
        if not volume_source_path:
            return

        from tubulemap.cellpose_tracker.mesh_to_mask_pipeline import derive_default_mask_output_path

        output_path = derive_default_mask_output_path(volume_source_path, Path(obj_folder).name)
        if not self._prepare_mask_output_path(output_path):
            self._update_mask_progress(0, "Mask generation cancelled (existing output was not overwritten).")
            return

        self.generate_mask_button.setEnabled(False)
        self._mask_result_handled = False
        self._update_mask_progress(0, "Starting mesh-to-mask generation...")

        self.mask_worker = start_generate_mask(obj_folder, volume_source_path, str(output_path))
        if hasattr(self.mask_worker, "yielded"):
            self.mask_worker.yielded.connect(self._on_mask_generation_yielded)
        if hasattr(self.mask_worker, "returned"):
            self.mask_worker.returned.connect(self._on_mask_generation_returned)
        if hasattr(self.mask_worker, "errored"):
            self.mask_worker.errored.connect(self._on_mask_generation_error)
        self.mask_worker.finished.connect(self._on_mask_generation_finished)
        self.mask_worker.start()

    def _update_mask_progress(self, progress, message):
        """Update the mask progress bar and status text."""
        progress_value = int(max(0, min(100, int(progress))))
        self.mask_progress_bar.setValue(progress_value)
        self.mask_progress_status.setText(str(message))

    def _on_mask_generation_yielded(self, payload):
        """Handle incremental mask-generation progress events."""
        if not isinstance(payload, dict):
            return
        payload_type = payload.get("type")
        if payload_type == "progress":
            self._update_mask_progress(payload.get("progress", 0), payload.get("message", "Working..."))
        elif payload_type == "result":
            self._on_mask_generation_returned(payload.get("report"))

    def _on_mask_generation_returned(self, report):
        """Process mask-generation result exactly once."""
        if self._mask_result_handled:
            return
        self._mask_result_handled = True
        self._on_mask_generation_result(report)

    def _on_mask_generation_result(self, report):
        """Display a summary for successful mask generation."""
        if not isinstance(report, dict):
            QMessageBox.warning(self, "Mask Generation", "Mask generation finished with an unexpected result.")
            return

        output_path = report.get("output_path", "")
        obj_count = int(report.get("obj_count", 0))
        processed_meshes = int(report.get("processed_meshes", 0))
        failed_meshes = int(report.get("failed_meshes", 0))
        elapsed_sec = report.get("elapsed_sec", None)
        pyramid_report = report.get("pyramid", {}) if isinstance(report.get("pyramid"), dict) else {}
        if pyramid_report.get("built"):
            pyramid_state = "built"
        elif pyramid_report.get("skipped"):
            pyramid_state = "reused"
        else:
            pyramid_state = "not requested"

        summary = (
            f"Mask generated:\n{output_path}\n\n"
            f"OBJ files found: {obj_count}\n"
            f"Processed meshes: {processed_meshes}\n"
            f"Failed meshes: {failed_meshes}\n"
            f"Pyramid: {pyramid_state}\n"
            f"Elapsed (s): {elapsed_sec}"
        )
        self._update_mask_progress(100, "Mask generation complete.")
        QMessageBox.information(self, "Mask Generation Complete", summary)

    def _on_mask_generation_error(self, error):
        """Display mask-generation errors."""
        if isinstance(error, tuple) and error:
            error_text = str(error[0])
        else:
            error_text = str(error)
        self._update_mask_progress(self.mask_progress_bar.value(), f"Mask generation failed: {error_text}")
        self.generate_mask_button.setEnabled(True)
        self.mask_worker = None
        QMessageBox.critical(self, "Mask Generation Failed", error_text)

    def _on_mask_generation_finished(self):
        """Reset mask-generation button state."""
        self.generate_mask_button.setEnabled(True)
        self.mask_worker = None

    def load_obj_into_viewer(self):
        """Load an OBJ mesh file and display it as a napari surface layer."""
        obj_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OBJ Mesh",
            "",
            "OBJ Files (*.obj);;All Files (*)",
        )
        if not obj_path:
            return

        try:
            import trimesh
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load OBJ Failed",
                f"Could not import trimesh:\n{exc}",
            )
            return

        try:
            loaded = trimesh.load(obj_path, force="mesh", process=False)
            if isinstance(loaded, trimesh.Trimesh):
                mesh = loaded
            elif isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            else:
                loaded = trimesh.load(obj_path, process=False)
                if isinstance(loaded, trimesh.Scene):
                    mesh = loaded.dump(concatenate=True)
                elif isinstance(loaded, trimesh.Trimesh):
                    mesh = loaded
                else:
                    raise ValueError("Unsupported mesh type returned from trimesh.")

            raw_vertices = np.asarray(mesh.vertices, dtype=float)
            vertices, axis_info = self._map_obj_vertices_to_display(raw_vertices)
            vertices = np.asarray(vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)

            if vertices.size == 0 or faces.size == 0:
                raise ValueError("OBJ contains no valid vertices/faces.")

            layer_name = f"Mesh: {Path(obj_path).stem}"
            existing = self.viewer.layers[layer_name] if layer_name in self.viewer.layers else None
            surface_data = (vertices, faces)
            if existing is not None and isinstance(existing, napari.layers.Surface):
                existing.data = surface_data
            else:
                surface_layer = self.viewer.add_surface(
                    surface_data,
                    name=layer_name,
                    opacity=0.6,
                    shading="flat",
                )
                if hasattr(surface_layer, "metadata") and isinstance(surface_layer.metadata, dict):
                    surface_layer.metadata["tubulemap_obj_axis_mapping"] = axis_info
            # Keep user's current display mode (2D/3D) unchanged.
            # In 2D mode, napari shows slice intersections of the surface.
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load OBJ Failed",
                f"Could not load OBJ file:\n{obj_path}\n\n{exc}",
            )

    def _display_point_axes(self):
        """Return the axis order used by the currently selected source layer."""
        source_meta = self._current_source_meta or self._resolve_current_source_meta()
        if isinstance(source_meta, dict):
            axes = [str(axis).strip().lower() for axis in source_meta.get("axes", [])]
            if len(axes) >= 3 and all(axis in axes for axis in ("z", "y", "x")):
                return axes
        return ["z", "y", "x"]

    def _display_spatial_axes(self):
        """Return active spatial axis order as a 3-axis list containing z,y,x."""
        axes = self._display_point_axes()
        spatial_axes = [axis for axis in axes if axis in {"z", "y", "x"}]
        if len(spatial_axes) == 3:
            return spatial_axes
        return ["z", "y", "x"]

    def _vertices_zyx_to_display_axes(self, vertices):
        """
        Convert canonical [z,y,x] vertices to the active display spatial axis order.
        """
        arr = np.asarray(vertices, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return arr

        spatial_axes = self._display_spatial_axes()
        if spatial_axes == ["z", "y", "x"]:
            return arr[:, :3]

        mapped = []
        for row in arr:
            z, y, x = float(row[0]), float(row[1]), float(row[2])
            coords = []
            for axis in spatial_axes:
                if axis == "z":
                    coords.append(z)
                elif axis == "y":
                    coords.append(y)
                elif axis == "x":
                    coords.append(x)
                else:
                    coords.append(0.0)
            mapped.append(coords)
        return np.asarray(mapped, dtype=float)

    def _display_spatial_shape(self):
        """Return active display spatial shape [d0,d1,d2], when available."""
        source_meta = self._current_source_meta or self._resolve_current_source_meta()
        if not isinstance(source_meta, dict):
            return None

        levels = source_meta.get("levels", [])
        if not isinstance(levels, list) or len(levels) == 0:
            return None

        run_level_widget = getattr(self.run_trace_widget, "run_level", None)
        if run_level_widget is not None and str(getattr(run_level_widget, "value", "")).strip():
            level_idx = int(run_level_widget.value)
        else:
            level_idx = 0
        level_idx = max(0, min(level_idx, len(levels) - 1))

        level_shape_zyx = levels[level_idx].get("shape_zyx")
        if not isinstance(level_shape_zyx, (list, tuple)) or len(level_shape_zyx) < 3:
            return None

        axis_sizes = {
            "z": float(level_shape_zyx[0]),
            "y": float(level_shape_zyx[1]),
            "x": float(level_shape_zyx[2]),
        }
        display_axes = self._display_spatial_axes()
        try:
            return [axis_sizes[axis] for axis in display_axes]
        except Exception:
            return None

    @staticmethod
    def _in_bounds_ratio(points, shape):
        """Compute fraction of points within a [d0,d1,d2] volume shape."""
        if shape is None:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
            return None
        d0, d1, d2 = [float(v) for v in shape]
        inside = (
            (arr[:, 0] >= 0.0) & (arr[:, 0] < d0)
            & (arr[:, 1] >= 0.0) & (arr[:, 1] < d1)
            & (arr[:, 2] >= 0.0) & (arr[:, 2] < d2)
        )
        return float(np.mean(inside))

    def _map_obj_vertices_to_display(self, raw_vertices):
        """
        Auto-map OBJ vertices to display axes by testing raw->zyx permutations.

        OBJ files can encode coordinates in a different axis order than our canonical
        z,y,x convention. We score each permutation using in-bounds ratio against the
        active source shape and select the best candidate.
        """
        arr = np.asarray(raw_vertices, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return arr, {"mode": "passthrough"}

        # Keep scoring fast on large meshes.
        sample = arr
        if len(arr) > 5000:
            idx = np.linspace(0, len(arr) - 1, 5000).astype(int)
            sample = arr[idx]

        shape_display = self._display_spatial_shape()
        perms = list(permutations((0, 1, 2)))
        best_perm = (0, 1, 2)
        best_ratio = -1.0

        for perm in perms:
            candidate_zyx = sample[:, list(perm)]
            candidate_display = self._vertices_zyx_to_display_axes(candidate_zyx)
            ratio = self._in_bounds_ratio(candidate_display, shape_display)
            score = -1.0 if ratio is None else float(ratio)
            if score > best_ratio:
                best_ratio = score
                best_perm = perm

        remapped_zyx = arr[:, list(best_perm)]
        remapped_display = self._vertices_zyx_to_display_axes(remapped_zyx)
        axis_info = {
            "raw_to_zyx_perm": [int(i) for i in best_perm],
            "display_axes": self._display_spatial_axes(),
            "in_bounds_ratio": None if best_ratio < 0 else float(best_ratio),
        }
        self.viewer.status = (
            "OBJ axis mapping selected: "
            f"raw[{best_perm[0]}],raw[{best_perm[1]}],raw[{best_perm[2]}] -> z,y,x"
        )
        return remapped_display, axis_info

    def _points_zyx_to_display_axes(self, points):
        """
        Convert canonical [z,y,x] points into the active source axis order for napari display.
        """
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return arr, ["z", "y", "x"]

        axes = self._display_point_axes()
        if axes == ["z", "y", "x"]:
            return arr, axes

        run_time_widget = getattr(self.run_trace_widget, "run_time_index", None)
        run_channel_widget = getattr(self.run_trace_widget, "run_channel_index", None)
        run_time_index = int(run_time_widget.value) if run_time_widget is not None else 0
        run_channel_index = int(run_channel_widget.value) if run_channel_widget is not None else 0

        mapped = []
        for row in arr:
            z, y, x = float(row[0]), float(row[1]), float(row[2])
            coords = []
            for axis in axes:
                if axis == "t":
                    coords.append(float(run_time_index))
                elif axis == "c":
                    coords.append(float(run_channel_index))
                elif axis == "z":
                    coords.append(z)
                elif axis == "y":
                    coords.append(y)
                elif axis == "x":
                    coords.append(x)
                else:
                    coords.append(0.0)
            mapped.append(coords)
        return np.asarray(mapped, dtype=float), axes

    def update_viewer(self, data):
        """Update viewer."""
        if data is None:
            return
        # Update the viewer with data emitted from run_trace
        viewer = self.viewer

        # Update or add the points layer
        points = data.get('points')
        points_name = data.get('points_name', 'Tracked Points')
        if points is not None:
            display_points, display_axes = self._points_zyx_to_display_axes(points)
            if points_name in viewer.layers:
                points_layer = viewer.layers[points_name]
                points_layer.data = display_points
            else:
                points_layer = viewer.add_points(display_points, size=30, face_color='red', name=points_name)
            if hasattr(points_layer, "metadata") and isinstance(points_layer.metadata, dict):
                points_layer.metadata["tubulemap_point_axes"] = display_axes

        # Update or add the image layer
        rectified_data = data.get('rectified_data')
        if rectified_data is not None:
            if '3D Volume' in viewer.layers:
                viewer.layers['3D Volume'].data = rectified_data
            else:
                viewer.add_image(rectified_data, name='3D Volume', colormap='gray')

        # Update or add the labels layer
        rectified_mask = data.get('rectified_mask')
        if rectified_mask is not None:
            if '3D Mask' in viewer.layers:
                viewer.layers['3D Mask'].data = rectified_mask
            else:
                viewer.add_labels(rectified_mask, name='3D Mask')
