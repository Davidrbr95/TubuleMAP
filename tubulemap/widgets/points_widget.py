import os
import json
import time
import tempfile
import subprocess
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd

from qtpy.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QSpinBox, 
    QHBoxLayout, QLabel, QColorDialog, QListWidget, QFileDialog, QInputDialog
)

import napari

from tubulemap.utils.misc_utils import is_excel_running
from tubulemap.cellpose_tracker.io_utils import normalize_points_to_zyx
from tubulemap.utils.zarr_resolution import inspect_zarr_source
from tubulemap.widgets.downsample_control_widget import (
    get_downsample_factor,
    is_downsample_enabled,
    to_downsample_points,
    to_original_points,
)


class PointsWidget(QWidget):
    """
    A QWidget-based class for managing points in a Napari viewer.

    This widget provides functionality to load, save, and reverse_points in a list.
    PointsWidget contains PointsListWidget, which expands its capabilities by providing
    the functionality to edit points, change their size, and navigate through them.
    Points can also be exported to and edited in Excel.
    """

    def __init__(self, viewer):
        """
        Initialize the PointsWidget.

        Parameters:
        viewer (napari.Viewer): The Napari viewer instance.
        """
        super().__init__()
        self.viewer = viewer
        self.name = 'Points Widget'
        self._points_axis_reference_layer_name = None

        # Initialize UI elements
        self.load_button = QPushButton('Load Points from JSON')
        self.save_button = QPushButton('Save Points to JSON')

        self.points_list_widget = PointsListWidget(viewer)
        self.color_button = QPushButton('Change Color')
        self.next_button = QPushButton('Next Point')
        self.prev_button = QPushButton('Previous Point')
        self.delete_button = QPushButton('Delete Point')
        self.add_button = QPushButton('Add Point')

        self._initialize_layout()
        self._connect_signals()
        self.viewer.layers.events.inserted.connect(self._on_layers_inserted)
        self.viewer.layers.events.removed.connect(self._on_layers_removed)

    def _initialize_layout(self):
        """Set up the layout for the widget."""
        layout = QVBoxLayout()
        nav_layout = QHBoxLayout()

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.points_list_widget)
        layout.addWidget(self.color_button)
        layout.addLayout(nav_layout)
        layout.addWidget(self.delete_button)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect UI signals to their respective slots (event handlers)."""
        self.load_button.clicked.connect(self.load_points)
        self.save_button.clicked.connect(self.save_points)
        self.next_button.clicked.connect(self.points_list_widget.next_point)
        self.prev_button.clicked.connect(self.points_list_widget.prev_point)
        self.delete_button.clicked.connect(self.points_list_widget.delete_point)
        self.add_button.clicked.connect(self.points_list_widget.add_point)
        self.color_button.clicked.connect(self.points_list_widget.change_color)

    def _image_layers(self):
        """Return all image layers in the current viewer."""
        return [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

    def _layer_by_name(self, layer_name):
        """Return a viewer layer by name, if present."""
        if not layer_name:
            return None
        if layer_name in self.viewer.layers:
            return self.viewer.layers[layer_name]
        return None

    def _get_reference_image_layer(self):
        """Return the selected reference image layer for axis mapping."""
        layer = self._layer_by_name(self._points_axis_reference_layer_name)
        if isinstance(layer, napari.layers.Image):
            return layer
        return None

    def _prompt_points_axis_reference_layer(self):
        """Prompt user to choose which loaded volume defines point axis order."""
        image_layers = self._image_layers()
        if len(image_layers) < 2:
            return

        layer_names = [layer.name for layer in image_layers]
        current_name = self._points_axis_reference_layer_name
        if current_name not in layer_names:
            current_name = layer_names[0]
        current_index = layer_names.index(current_name)

        choice, ok = QInputDialog.getItem(
            self,
            "Points Axis Reference Volume",
            "Multiple volumes are loaded.\nChoose the volume that defines point axis order:",
            layer_names,
            current_index,
            False,
        )
        if ok and choice:
            self._points_axis_reference_layer_name = str(choice)

    def _reorder_points_layer_to_axes(self, points_layer, target_axes):
        """Reorder a points layer into the specified axis order."""
        if not isinstance(points_layer, napari.layers.Points):
            return
        if not target_axes or len(target_axes) < 3:
            return

        metadata = getattr(points_layer, "metadata", {}) or {}
        source_axes = None
        if isinstance(metadata, dict):
            axes_candidate = metadata.get("tubulemap_point_axes")
            if isinstance(axes_candidate, (list, tuple)):
                source_axes = [str(axis).strip().lower() for axis in axes_candidate]

        # Do not mutate unlabeled manual points layers.
        if source_axes is None:
            return

        target_axes = [str(axis).strip().lower() for axis in target_axes]
        if source_axes == target_axes:
            return

        try:
            points_zyx = normalize_points_to_zyx(points_layer.data.tolist(), source_axes=source_axes)
            mapped = self._map_zyx_to_axes(points_zyx, target_axes)
        except Exception:
            return

        points_layer.data = np.asarray(mapped, dtype=float)
        if isinstance(metadata, dict):
            metadata["tubulemap_point_axes"] = target_axes

    def _reorder_all_points_layers_for_current_volume(self):
        """Reorder all axis-aware points layers to the active reference volume axes."""
        target_axes = self._resolve_display_axes()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._reorder_points_layer_to_axes(layer, target_axes)

    def _on_layers_inserted(self, event):
        """Handle layer insertion events to keep point axes in sync with loaded volumes."""
        layer = getattr(event, "value", None)
        if not isinstance(layer, napari.layers.Image):
            return

        image_layers = self._image_layers()
        if len(image_layers) == 1:
            self._points_axis_reference_layer_name = layer.name
        elif len(image_layers) == 2:
            self._prompt_points_axis_reference_layer()

        self._reorder_all_points_layers_for_current_volume()

    def _on_layers_removed(self, event):
        """Handle layer removal events and maintain a valid axis reference layer."""
        layer = getattr(event, "value", None)
        if not isinstance(layer, napari.layers.Image):
            return

        if layer.name == self._points_axis_reference_layer_name:
            self._points_axis_reference_layer_name = None
            image_layers = self._image_layers()
            if image_layers:
                self._points_axis_reference_layer_name = image_layers[0].name

    def _resolve_display_axes(self):
        """Resolve the active image axis order used for point display."""
        active_layer = self.viewer.layers.selection.active
        reference_layer = self._get_reference_image_layer()
        candidates = []
        if isinstance(reference_layer, napari.layers.Image):
            candidates.append(reference_layer)
        if isinstance(active_layer, napari.layers.Image):
            candidates.append(active_layer)
        candidates.extend(
            [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image) and layer not in candidates
            ]
        )

        for layer in candidates:
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            source_meta = metadata.get("tubulemap_source_resolution")
            if not isinstance(source_meta, dict):
                source_path = getattr(layer.source, "path", None)
                if source_path:
                    source_path = str(source_path).rstrip("/\\")
                    inspect_candidates = [source_path]
                    parent_1 = str(Path(source_path).parent)
                    parent_2 = str(Path(parent_1).parent)
                    inspect_candidates.extend([parent_1, parent_2])
                    seen = set()
                    for inspect_path in inspect_candidates:
                        if not inspect_path or inspect_path in seen or not os.path.exists(inspect_path):
                            continue
                        seen.add(inspect_path)
                        try:
                            inferred_meta = inspect_zarr_source(inspect_path)
                        except Exception:
                            continue
                        if isinstance(inferred_meta, dict):
                            source_meta = inferred_meta
                            if isinstance(metadata, dict):
                                metadata["tubulemap_source_resolution"] = inferred_meta
                            break
                if not isinstance(source_meta, dict):
                    continue
            axes = [str(axis).strip().lower() for axis in source_meta.get("axes", [])]
            if len(axes) >= 3 and all(axis in axes for axis in ("z", "y", "x")):
                layer_shape = self._layer_shape(layer)
                inferred_axes = self._infer_display_axes_from_layer_shape(source_meta, layer_shape)
                if (
                    isinstance(inferred_axes, list)
                    and len(inferred_axes) >= 3
                    and all(axis in inferred_axes for axis in ("z", "y", "x"))
                ):
                    return inferred_axes
                return axes
        return ["z", "y", "x"]

    @staticmethod
    def _layer_shape(layer):
        """Return the visible image shape tuple for an image layer."""
        data = getattr(layer, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None and isinstance(data, (list, tuple)) and len(data) > 0:
            shape = getattr(data[0], "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(v) for v in shape)
        except Exception:
            return None

    @staticmethod
    def _infer_display_axes_from_layer_shape(source_meta, layer_shape):
        """
        Infer axis order used by the loaded napari layer from shape matching.

        This handles cases where source metadata axes (for example, x,y,z) do not
        match the in-memory array order shown in napari (for example, z,y,x).
        """
        if not isinstance(source_meta, dict):
            return None
        if layer_shape is None:
            return None
        axes = [str(axis).strip().lower() for axis in source_meta.get("axes", [])]
        levels = source_meta.get("levels", [])
        if not axes or not levels:
            return None
        source_shape = levels[0].get("shape", [])
        if len(source_shape) != len(axes) or len(layer_shape) != len(axes):
            return None

        # Match each displayed dimension to one source axis by unique size.
        used_source_idx = set()
        display_to_source = []
        for dim in layer_shape:
            matches = [
                idx
                for idx, src_dim in enumerate(source_shape)
                if idx not in used_source_idx and int(src_dim) == int(dim)
            ]
            if len(matches) != 1:
                return None
            src_idx = matches[0]
            used_source_idx.add(src_idx)
            display_to_source.append(src_idx)

        return [axes[src_idx] for src_idx in display_to_source]

    def _resolve_source_meta(self):
        """Resolve source metadata for the active image layer when available."""
        active_layer = self.viewer.layers.selection.active
        candidates = []
        if isinstance(active_layer, napari.layers.Image):
            candidates.append(active_layer)
        candidates.extend(
            [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image) and layer is not active_layer]
        )

        for layer in candidates:
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            source_meta = metadata.get("tubulemap_source_resolution")
            if not isinstance(source_meta, dict):
                source_path = getattr(layer.source, "path", None)
                if source_path:
                    source_path = str(source_path).rstrip("/\\")
                    inspect_candidates = [source_path]
                    parent_1 = str(Path(source_path).parent)
                    parent_2 = str(Path(parent_1).parent)
                    inspect_candidates.extend([parent_1, parent_2])
                    seen = set()
                    for inspect_path in inspect_candidates:
                        if not inspect_path or inspect_path in seen or not os.path.exists(inspect_path):
                            continue
                        seen.add(inspect_path)
                        try:
                            inferred_meta = inspect_zarr_source(inspect_path)
                        except Exception:
                            continue
                        if isinstance(inferred_meta, dict):
                            source_meta = inferred_meta
                            metadata["tubulemap_source_resolution"] = inferred_meta
                            break
            if isinstance(source_meta, dict):
                return source_meta
        return None

    @staticmethod
    def _shape_zyx_from_source_meta(source_meta):
        """Extract level-0 shape in zyx order from source metadata."""
        if not isinstance(source_meta, dict):
            return None
        axes = [str(axis).strip().lower() for axis in source_meta.get("axes", [])]
        levels = source_meta.get("levels", [])
        if not levels:
            return None
        shape = levels[0].get("shape", [])
        if not axes or not shape:
            return None
        axis_to_index = {axis: idx for idx, axis in enumerate(axes)}
        if not all(axis in axis_to_index for axis in ("z", "y", "x")):
            return None
        z_idx, y_idx, x_idx = axis_to_index["z"], axis_to_index["y"], axis_to_index["x"]
        if any(idx >= len(shape) for idx in (z_idx, y_idx, x_idx)):
            return None
        return (int(shape[z_idx]), int(shape[y_idx]), int(shape[x_idx]))

    @staticmethod
    def _in_bounds_ratio(points_zyx, shape_zyx):
        """Compute fraction of points within zyx bounds."""
        if shape_zyx is None:
            return 0.0
        if points_zyx is None or len(points_zyx) == 0:
            return 0.0
        z_max, y_max, x_max = [int(v) for v in shape_zyx]
        valid = 0
        for point in np.asarray(points_zyx, dtype=float):
            if point.shape[0] < 3:
                continue
            z, y, x = float(point[0]), float(point[1]), float(point[2])
            if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
                valid += 1
        return valid / float(len(points_zyx))

    def _infer_point_axes_for_json(self, points_data):
        """
        Infer axis order for 3D point rows when point_axes is missing.

        Returns axis names for raw rows (for normalize_points_to_zyx source_axes), or None.
        """
        source_meta = self._resolve_source_meta()
        shape_zyx = self._shape_zyx_from_source_meta(source_meta)
        if shape_zyx is None:
            return None
        if points_data is None or len(points_data) == 0:
            return None
        if any(not isinstance(point, (list, tuple)) or len(point) != 3 for point in points_data):
            return None

        raw = np.asarray(points_data, dtype=float)
        # Base assumption in legacy files: already z,y,x
        best_axes = ["z", "y", "x"]
        best_ratio = self._in_bounds_ratio(raw[:, [0, 1, 2]], shape_zyx)
        base_ratio = best_ratio

        # Evaluate all raw->(z,y,x) index assignments.
        for perm in permutations((0, 1, 2)):
            candidate = raw[:, [perm[0], perm[1], perm[2]]]
            ratio = self._in_bounds_ratio(candidate, shape_zyx)
            if ratio > best_ratio:
                best_ratio = ratio
                best_axes = [f"dim_{perm[0]}", f"dim_{perm[1]}", f"dim_{perm[2]}"]

        # If best mapping is equivalent to base zyx, no inference needed.
        if best_axes == ["z", "y", "x"]:
            return None

        # Only infer when clearly better to avoid changing already-good datasets.
        if (best_ratio - base_ratio) < 0.3:
            return None

        # Convert inferred dim indices to axis names expected by normalize_points_to_zyx.
        dim_to_axis = {
            best_axes[0]: "z",
            best_axes[1]: "y",
            best_axes[2]: "x",
        }
        # Build raw axis labels: each raw column gets one axis name.
        raw_axes = [""] * 3
        for dim_label, axis_name in dim_to_axis.items():
            raw_idx = int(dim_label.split("_")[1])
            raw_axes[raw_idx] = axis_name
        if all(raw_axes):
            self.viewer.status = (
                "Inferred point axis order from bounds: "
                f"raw[{raw_axes.index('z')}],raw[{raw_axes.index('y')}],raw[{raw_axes.index('x')}] -> z,y,x"
            )
            return raw_axes
        return None

    @staticmethod
    def _map_zyx_to_axes(points_zyx, axes):
        """Map canonical [z,y,x] points to the requested axis order."""
        arr = np.asarray(points_zyx, dtype=float)
        if arr.size == 0:
            return np.empty((0, len(axes)), dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return arr
        if list(axes) == ["z", "y", "x"]:
            return arr

        mapped = []
        for row in arr:
            z, y, x = float(row[0]), float(row[1]), float(row[2])
            coords = []
            for axis in axes:
                if axis == "z":
                    coords.append(z)
                elif axis == "y":
                    coords.append(y)
                elif axis == "x":
                    coords.append(x)
                elif axis in {"t", "c"}:
                    coords.append(0.0)
                else:
                    coords.append(0.0)
            mapped.append(coords)
        return np.asarray(mapped, dtype=float)


    def load_points(self):
        """
        Load points from multiple JSON files and add them to the viewer.
        
        Opens a file dialog for selecting one or more JSON files containing the points.
        """
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Open Points JSON", "", 
            "JSON Files (*.json);;All Files (*)", options=options
        )

        if file_names:
            for file_name in file_names:
                with open(file_name, 'r') as f:
                    payload = json.load(f)
                base_name = os.path.basename(file_name) 
                if isinstance(payload, dict):
                    points_data = payload.get('points', [])
                    point_axes = payload.get('point_axes')
                else:
                    points_data = payload
                    point_axes = None

                if not isinstance(point_axes, (list, tuple)):
                    point_axes = self._infer_point_axes_for_json(points_data)

                normalized_points = normalize_points_to_zyx(points_data, source_axes=point_axes)
                normalized_points = np.asarray(normalized_points, dtype=float)

                if is_downsample_enabled(self.viewer):
                    factor = get_downsample_factor(self.viewer)
                    points = np.array(to_downsample_points(normalized_points, factor), dtype=float)
                    output_axes = ["t", "c", "z", "y", "x"]
                else:
                    output_axes = self._resolve_display_axes()
                    points = self._map_zyx_to_axes(normalized_points, output_axes)

                layer = self.viewer.add_points(points, size=30, face_color='red', name=f'{base_name}')
                if hasattr(layer, "metadata") and isinstance(layer.metadata, dict):
                    layer.metadata["tubulemap_point_axes"] = [str(axis).strip().lower() for axis in output_axes]

    def save_points(self):
        """
        Save the currently selected points layer to a JSON file.
        
        Opens a file dialog for saving the points data to a JSON file.
        """
        points_layer = self.viewer.layers.selection.active
        if isinstance(points_layer, napari.layers.Points):
            points = points_layer.data.tolist()
            points_layer_meta = getattr(points_layer, "metadata", {}) or {}
            point_axes = None
            if isinstance(points_layer_meta, dict):
                axes_candidate = points_layer_meta.get("tubulemap_point_axes")
                if isinstance(axes_candidate, (list, tuple)):
                    point_axes = [str(axis).strip().lower() for axis in axes_candidate]

            if is_downsample_enabled(self.viewer):
                factor = get_downsample_factor(self.viewer)
                points = to_original_points(points, factor)

            # If layer metadata is missing, infer a stable axis declaration
            # from the currently resolved image/source axis order.
            if point_axes is None:
                inferred_axes = [str(axis).strip().lower() for axis in self._resolve_display_axes()]
                point_dim = len(points[0]) if points and isinstance(points[0], (list, tuple)) else 3
                if point_dim == 3 and len(inferred_axes) > 3:
                    spatial_axes = [axis for axis in inferred_axes if axis in {"z", "y", "x"}]
                    if len(spatial_axes) == 3:
                        point_axes = spatial_axes
                    else:
                        point_axes = ["z", "y", "x"]
                elif len(inferred_axes) == point_dim:
                    point_axes = inferred_axes
                elif point_dim >= 5:
                    point_axes = ["t", "c", "z", "y", "x"]
                else:
                    point_axes = ["z", "y", "x"]

            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Points JSON", "", 
                "JSON Files (*.json);;All Files (*)", options=options
            )

            if file_name:
                points_data = {
                    'points': points,
                    'point_axes': point_axes,
                }
                with open(file_name, 'w') as f:
                    json.dump(points_data, f, indent=4)
        else:
            print("No points layer selected")

class PointsListWidget(QWidget):
    """
    A QWidget-based class for displaying and interacting with a list of points.

    This widget provides functionality to edit points, change their size, and 
    navigate through them. Points can also be exported to and edited in Excel.
    """

    def __init__(self, viewer):
        """
        Initialize the PointsListWidget.

        Parameters:
        viewer (napari.Viewer): The Napari viewer instance.
        """
        super().__init__()
        self.viewer = viewer
        self.points_layer = None
        self.current_index = 0
        self.changing_list = True

        self.list_widget = QListWidget()
        self.center_button = QPushButton('Center on Selected Point')
        self.refresh_button = QPushButton('Refresh Points List')
        self.size_spinbox = QSpinBox()
        self.edit_excel_button = QPushButton('Edit in Excel')

        self._initialize_layout()
        self._connect_signals()

    def _initialize_layout(self):
        """Set up the layout for the widget."""
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel('Points List'))
        layout.addWidget(self.list_widget)
        layout.addWidget(self.center_button)
        layout.addWidget(self.refresh_button)
        layout.addWidget(QLabel('Point Size'))
        layout.addWidget(self.size_spinbox)
        layout.addWidget(self.edit_excel_button)
        
        self.setLayout(layout)
        
        self.size_spinbox.setMinimum(1)
        self.size_spinbox.setValue(5)  # Default point size

    def _connect_signals(self):
        """Connect UI signals to their respective slots (event handlers)."""
        # Connect button-related signals
        self.center_button.clicked.connect(self.center_on_point)
        self.refresh_button.clicked.connect(self.update_points_list)
        self.edit_excel_button.clicked.connect(self.edit_points_in_excel)

        # Connect layer-related signals to update the points list
        self.viewer.layers.events.inserted.connect(self.update_layer)
        self.viewer.layers.events.removed.connect(self.update_layer)
        self.viewer.layers.events.changed.connect(self.update_layer)
        self.viewer.layers.selection.events.active.connect(self.update_layer)

        # Connect changes in slides for point size
        self.size_spinbox.valueChanged.connect(self.change_size)

        # Connect changes in the list viewer
        self.list_widget.currentRowChanged.connect(self.update_selection)

    def update_layer(self, event=None):
        """
        Update the points layer reference and refresh the points list.
        
        This method is triggered when the active layer changes in the viewer.
        """
        self.points_layer = self.viewer.layers.selection.active
        if not isinstance(self.points_layer, napari.layers.Points):
            self.points_layer = None

        self.update_points_list()
        if isinstance(self.points_layer, napari.layers.Points):
            self.points_layer.events.data.connect(self.update_points_list)

    def edit_points_in_excel(self):
        """
        Export the points to an Excel file, open the file for editing,
        and re-import the data after the file is closed.
        """
        if self.points_layer is not None:
            temp_dir = tempfile.gettempdir()  # Use the system's temp directory
            temp_file = os.path.join(temp_dir, "points.xlsx")

            use_downsample = is_downsample_enabled(self.viewer)

            data = self.points_layer.data
            point_dim = int(data.shape[1]) if getattr(data, "ndim", 0) == 2 else 0
            if use_downsample and point_dim >= 5:
                df = pd.DataFrame(data[:, :5], columns=["time", "channel", "Z", "Y", "X"])
            elif point_dim >= 3:
                df = pd.DataFrame(data[:, :3], columns=["Z", "Y", "X"])
            else:
                df = pd.DataFrame(data)
                
            df.to_excel(temp_file, index=False)
            
            # Open the Excel file for editing
            self._open_excel_file(temp_file)
            
            # Re-import the data after Excel is closed
            updated_df = pd.read_excel(temp_file)
            self.points_layer.data = updated_df.values

            # Update the points list in the UI
            self.update_points_list()

    def _open_excel_file(self, temp_file):
        """Open a temporary Excel file using the appropriate command for the OS."""
        if os.name == 'nt':  # For Windows
            subprocess.Popen(['start', 'excel', temp_file], shell=True)
        elif os.name == 'posix':  # For macOS/Linux
            subprocess.Popen(['open', temp_file])
        else:
            raise RuntimeError("Unsupported OS")

        # Polling loop to wait until the Excel file is closed
        time.sleep(10)  # Initial delay to ensure Excel has started
        while is_excel_running():
            print('Microsoft Excel is running')
            continue

    def update_points_list(self, event=None):
        """
        Refresh the points list in the widget based on the current points layer.
        
        This method is triggered when the points layer data changes. Ex. after new layer
        is selected, after the points are edited with excel, or when the refresh button 
        is clicked.
        """
        self.list_widget.clear()
        if self.points_layer is not None:
            for i, point in enumerate(self.points_layer.data):
                self.list_widget.addItem(f"Point {i}: {point}")
            self.list_widget.setCurrentRow(self.current_index)

    def center_on_point(self):
        """
        Center the viewer's camera on the currently selected point.
        
        This method uses the current dimension order in the viewer to determine
        which coordinates to use for centering. This allows us to center on
        the correct coordinate regardless of the current view [xy, zy, xz].
        """
        if self.points_layer is not None:
            current_item = self.list_widget.currentItem()
            if current_item:
                point_index = int(current_item.text().split(':')[0].split(' ')[1])
                self.current_index = point_index
                point = self.points_layer.data[point_index]

                # dims_order = self.viewer.dims.order

                # # Center the camera on the selected point
                # self.viewer.camera.center = [point[dims_order[1]], point[dims_order[2]]]
                # self.viewer.dims.set_current_step(dims_order[0], round(point[dims_order[0]]))

                if is_downsample_enabled(self.viewer) and len(point) >= 5:
                    factor = get_downsample_factor(self.viewer)
                    z = float(point[2])
                    y = float(point[3]) * factor
                    x = float(point[4]) * factor
                    self.viewer.camera.center = [y, x]
                    self.viewer.dims.set_current_step(2, round(z))
                else:
                    dims_order = self.viewer.dims.order
                    self.viewer.camera.center = [point[dims_order[1]], point[dims_order[2]]]
                    self.viewer.dims.set_current_step(dims_order[0], round(point[dims_order[0]]))
                    
                # Update the selection to highlight the current point
                self.points_layer.selected_data = {self.current_index}

    def next_point(self):
        """
        Move to the next point in the list and center the viewer on it.
        
        The list wraps around when the end is reached.
        """
        if self.points_layer is not None:
            self.current_index = (self.current_index + 1) % len(self.points_layer.data)
            self.list_widget.setCurrentRow(self.current_index)
            self.center_on_point()

    def prev_point(self):
        """
        Move to the previous point in the list and center the viewer on it.
        
        The list wraps around when the beginning is reached.
        """
        if self.points_layer is not None:
            self.current_index = (self.current_index - 1) % len(self.points_layer.data)
            self.list_widget.setCurrentRow(self.current_index)
            self.center_on_point()

    def delete_point(self):
        """
        Delete the currently selected point from the points layer.

        The selection moves to the previous point in the list.
        """
        if self.points_layer is not None and len(self.points_layer.data) > 0:
            self.changing_list = False
            save_index = self.current_index
            self.points_layer.data = np.delete(self.points_layer.data, self.current_index, axis=0)
            self.current_index = max(0, save_index - 1)
            print('Table of data current after deletion')
            print(self.points_layer.data)
            self.update_points_list()  # Update the list widget here
            if len(self.points_layer.data) > 0:
                self.points_layer.selected_data = set()
                # Ensure current_index is within valid range
                if self.current_index < len(self.points_layer.data):
                    print('self.current_index', self.current_index)
                    self.points_layer.selected_data = {self.current_index}
                else:
                    # Adjust current_index if out of bounds
                    self.current_index = len(self.points_layer.data) - 1
                self.points_layer.selected_data = {self.current_index}

    def add_point(self):
        """
        Add a new point after the currently selected point, duplicating its position.
        """
        if self.points_layer is not None:
            current_point = self.points_layer.data[self.current_index]
            self.changing_list = False
            save_index = self.current_index
            if len(self.points_layer.data) > 0:
                points = np.insert(self.points_layer.data, self.current_index + 1, current_point, axis=0)
            else:
                points = np.insert(self.points_layer.data, self.current_index, current_point, axis=0)
            self.points_layer.data = points
            self.current_index = save_index + 1
            self.update_points_list()  # Update the list widget here
            if len(self.points_layer.data) > 0:
                self.points_layer.selected_data = set()
                # Ensure current_index is within valid range
                if self.current_index < len(self.points_layer.data):
                    self.points_layer.selected_data = {self.current_index}
                else:
                    self.current_index = len(self.points_layer.data) - 1
                    self.points_layer.selected_data = {self.current_index}


    def change_color(self):
        """
        Open a color dialog to change the face and edge color of the points.
        """
        if self.points_layer is not None:
            color = QColorDialog.getColor()
            if color.isValid():
                self.points_layer.border_color = color.name()
                self.points_layer.face_color = color.name()
                self._update_default_size_and_color()

    def change_size(self):
        """
        Change the size of the points in the points layer.
        """
        if self.points_layer is not None:
            size = self.size_spinbox.value()
            self.points_layer.size = size
            self._update_default_size_and_color()

    def _update_default_size_and_color(self):
        """
        Update the points layer's current size and color attributes.

        This ensures that new points added to the layer use the updated
        size and color.
        """
        if self.points_layer is not None:
            self.points_layer.current_face_color = self.points_layer.face_color
            self.points_layer.current_border_color = self.points_layer.border_color
            self.points_layer.current_size = self.points_layer.size

    def update_selection(self, event):
        """
        Update the current selection index when the user selects a point in the list.
        """
        if self.changing_list:
            self.current_index = self.list_widget.currentRow()
            # Ensure current_index is within valid range
            if self.current_index >= len(self.points_layer.data):
                self.current_index = len(self.points_layer.data) - 1
        self.changing_list = True

