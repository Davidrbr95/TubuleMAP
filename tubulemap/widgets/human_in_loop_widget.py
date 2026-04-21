import os
import json

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QCheckBox,
    QFileDialog, QMessageBox, QLineEdit, QLabel, 
    QHBoxLayout, QInputDialog
)
import napari
from tubulemap.widgets.downsample_control_widget import (
    get_downsample_factor,
    is_downsample_enabled,
    to_downsample_points,
    to_original_points,
)

def read_status(status_path):
    """Return 'status' field from JSON, or 'unknown' if file is missing/invalid."""
    if not os.path.exists(status_path):
        return "unknown"
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        return data.get("status", "unknown")
    except:
        return "unknown"

def update_status(status_path, new_status):
    """Load a status JSON and set 'status' to new_status, overwriting the file."""
    if not os.path.exists(status_path):
        return  # If there's no status file, skip or handle as needed
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        data['status'] = new_status
        with open(status_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Warning: could not set status to {new_status}. {e}")

class HumanInLoopWidget(QWidget):
    """
    A QWidget-based class for working with a 'Human_in_loop' style folder:

      - Choose a folder containing job subfolders (e.g. Track_nephron_1.json).
      - Each subfolder can have multiple Run_X subfolders.
      - Load result_trace.json or corrected_points.json from the latest run folder.
      - Filter loaded jobs by status with two mutually-exclusive checkboxes.
        * Tracking finished traces (status == "done")
        * Reviewed and accepted traces (status == "all_complete")
      - Save points layers back as corrected_points.json and set the status 
        to either 'rerun' or 'all_complete' per user choice.
      - Delete all loaded points layers from the viewer.
    """

    def __init__(self, viewer: napari.Viewer):
        """Initialize the instance state."""
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Human In Loop Widget")

        # { layer_name: {"run_folder": ..., "status_path": ...} }
        self.layer_run_map = {}
        self.name = 'Human in Loop Widget'
        # Internal folder path storage
        self.folder_path = None

        # --- UI Elements ---
        self.folder_path_label = QLabel("Selected Folder:")
        self.folder_path_lineedit = QLineEdit()
        self.folder_path_lineedit.setPlaceholderText("No folder selected yet")
        self.folder_path_lineedit.setReadOnly(True)

        self.choose_folder_button = QPushButton("Choose Folder")

        self.load_button = QPushButton("Load Tracking Results")

        # --- New checkboxes ---
        self.checkbox_complete_tracking = QCheckBox("Load Traces with Tracking Finished")
        self.checkbox_complete_tracking.setToolTip(
            "Loads jobs with status 'done' (tracking process finished, not necessarily reviewed)."
        )
        self.checkbox_finalized_traces = QCheckBox("Load Reviewed and Accepted Traces")
        self.checkbox_finalized_traces.setToolTip(
            "Loads jobs with status 'all_complete' (reviewed and accepted/finalized)."
        )

        # (Mutually-exclusive logic)
        self.checkbox_complete_tracking.toggled.connect(self.on_complete_tracking_toggled)
        self.checkbox_finalized_traces.toggled.connect(self.on_finalized_traces_toggled)

        self.load_corrected_button = QPushButton("Load Corrected Points")
        self.save_all_button = QPushButton("Save All Corrected Points")
        self.save_active_button = QPushButton("Save Selected Corrected Points")
        self.finalize_selected_button = QPushButton("Finalize Selected Traces")
        self.delete_points_button = QPushButton("Delete All Points Layers")

        # --- Layout ---
        main_layout = QVBoxLayout()

        self.step1_label = QLabel("1. Select Review Folder")
        self.step1_label.setStyleSheet("font-weight: bold;")
        self.step1_hint = QLabel(
            "Choose the Human_in_loop folder containing job subfolders and *_status.json files."
        )
        self.step1_hint.setWordWrap(True)

        self.step2_label = QLabel("2. Load Traces and Review End Points")
        self.step2_label.setStyleSheet("font-weight: bold;")
        self.step2_hint = QLabel(
            "Load tracking or corrected traces, then inspect each track. "
            "If there are endpoint errors, correct the end points before saving."
        )
        self.step2_hint.setWordWrap(True)

        self.step3_label = QLabel("3. Save Corrected Points and Set Review Status")
        self.step3_label.setStyleSheet("font-weight: bold;")
        self.step3_hint = QLabel(
            "Save selected or all corrected points, then choose either "
            "'Needs Re-Run' or 'Reviewed and Accepted (Finalized)'."
        )
        self.step3_hint.setWordWrap(True)

        self.step4_label = QLabel("4. Finalize Selected Traces (Set all_complete)")
        self.step4_label.setStyleSheet("font-weight: bold;")
        self.step4_hint = QLabel(
            "If selected tracks are fully reviewed and accepted, click finalize to set "
            "status to 'all_complete' for all selected layers."
        )
        self.step4_hint.setWordWrap(True)

        main_layout.addWidget(self.step1_label)
        main_layout.addWidget(self.step1_hint)

        # Folder selection row
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_path_label)
        folder_layout.addWidget(self.folder_path_lineedit)
        main_layout.addLayout(folder_layout)

        main_layout.addWidget(self.choose_folder_button)

        main_layout.addWidget(self.step2_label)
        main_layout.addWidget(self.step2_hint)
        main_layout.addWidget(self.load_button)
        main_layout.addWidget(self.checkbox_complete_tracking)
        main_layout.addWidget(self.checkbox_finalized_traces)

        main_layout.addWidget(self.load_corrected_button)

        main_layout.addWidget(self.step3_label)
        main_layout.addWidget(self.step3_hint)
        main_layout.addWidget(self.save_all_button)
        main_layout.addWidget(self.save_active_button)

        main_layout.addWidget(self.step4_label)
        main_layout.addWidget(self.step4_hint)
        main_layout.addWidget(self.finalize_selected_button)

        main_layout.addWidget(self.delete_points_button)

        self.setLayout(main_layout)

        # --- Connect signals ---
        self.choose_folder_button.clicked.connect(self.choose_folder)
        self.load_button.clicked.connect(self.load_latest_runs)
        self.load_corrected_button.clicked.connect(self.load_corrected_points)
        self.save_all_button.clicked.connect(self.save_all_points)
        self.save_active_button.clicked.connect(self.save_active_points)
        self.finalize_selected_button.clicked.connect(self.finalize_selected_traces)
        self.delete_points_button.clicked.connect(self.delete_points_layers)

    # ---------------------------
    #   Mutually-exclusive boxes
    # ---------------------------
    def on_complete_tracking_toggled(self, checked):
        """If this box is checked, uncheck the 'reviewed and accepted' box."""
        if checked:
            self.checkbox_finalized_traces.setChecked(False)

    def on_finalized_traces_toggled(self, checked):
        """If this box is checked, uncheck the 'tracking finished' box."""
        if checked:
            self.checkbox_complete_tracking.setChecked(False)

    # ---------------------------
    #   Folder selection
    # ---------------------------
    def choose_folder(self):
        """Open a folder selection dialog for the user."""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Human_in_loop Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        if chosen:
            self.folder_path = chosen
            self.folder_path_lineedit.setText(chosen)
        else:
            self.folder_path = None
            self.folder_path_lineedit.setText("")

    # ---------------------------
    #   Load - result_trace
    # ---------------------------
    def load_latest_runs(self):
        """
        For each job folder in self.folder_path, find the latest Run_X subfolder
        and load the 'result_trace.json' file as a points layer.

        - If 'Load Traces with Tracking Finished' is checked -> load only status == "done"
        - If 'Load Reviewed and Accepted Traces' is checked -> load only status == "all_complete"
        - Else load any status.
        """
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please choose a folder first.")
            return

        loaded_count = 0
        max_load = 5

        # Determine filters based on checkboxes
        # done -> tracking finished
        # all_complete -> reviewed and accepted/finalized
        only_done = self.checkbox_complete_tracking.isChecked()
        only_all_complete = self.checkbox_finalized_traces.isChecked()

        for sub in os.listdir(self.folder_path):
           
            if loaded_count >= max_load:
                break

            job_folder = os.path.join(self.folder_path, sub)
            if not os.path.isdir(job_folder):
                continue

            # Construct the status.json path
            core_name = sub[:-5] if sub.endswith('.json') else sub
            status_path = os.path.join(self.folder_path, f"{core_name}_status.json")

            job_status = read_status(status_path)

            # Filter logic
            if only_done and job_status != "done":
                continue
            if only_all_complete and job_status != "all_complete":
                continue

            # Find the latest Run_X subfolder
            latest_run = self._get_latest_run_folder(job_folder)
            if not latest_run:
                continue

            # Look for 'result_trace.json'
            result_trace = os.path.join(latest_run, "result_trace.json")
            if os.path.exists(result_trace):
                with open(result_trace, 'r') as f:
                    data = json.load(f)
                # points = np.array(data['points'], dtype=float)
                if is_downsample_enabled(self.viewer):
                    factor = get_downsample_factor(self.viewer)
                    points = np.array(to_downsample_points(data["points"], factor), dtype=float)
                else:
                    points = np.array(data['points'], dtype=float)
                layer_name = f"{os.path.basename(job_folder)}_{os.path.basename(latest_run)}"
                layer = self.viewer.add_points(points, size=30, face_color='red', name=layer_name)

                # Store info
                self.layer_run_map[layer.name] = {
                    "run_folder": latest_run,
                    "status_path": status_path
                }
                loaded_count += 1

        QMessageBox.information(
            self,
            "Load Done",
            f"Loaded results from {loaded_count} folder(s)."
        )

    # ---------------------------
    #   Load - corrected_points
    # ---------------------------
    def load_corrected_points(self):
        """
        Similar to 'load_latest_runs' but attempts to load 'corrected_points.json'
        in the latest Run_X folder for each job folder.
        Applies the same status-based filtering as above.
        """
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please choose a folder first.")
            return

        loaded_count = 0

        # Determine filters based on checkboxes
        # done -> tracking finished
        # all_complete -> reviewed and accepted/finalized
        only_done = self.checkbox_complete_tracking.isChecked()
        only_all_complete = self.checkbox_finalized_traces.isChecked()

        for sub in os.listdir(self.folder_path):
            job_folder = os.path.join(self.folder_path, sub)
            if not os.path.isdir(job_folder):
                continue

            # Construct the status path
            core_name = sub[:-5] if sub.endswith('.json') else sub
            status_path = os.path.join(self.folder_path, f"{core_name}_status.json")

            job_status = read_status(status_path)

            # Filter logic
            if only_done and job_status != "done":
                continue
            if only_all_complete and job_status != "all_complete":
                continue

            latest_run = self._get_latest_run_folder(job_folder)
            if not latest_run:
                continue

            corrected_path = os.path.join(latest_run, "corrected_points.json")
            if os.path.exists(corrected_path):
                with open(corrected_path, 'r') as f:
                    data = json.load(f)
                # points = np.array(data['points'], dtype=float)
                if is_downsample_enabled(self.viewer):
                    factor = get_downsample_factor(self.viewer)
                    points = np.array(to_downsample_points(data["points"], factor), dtype=float)
                else:
                    points = np.array(data['points'], dtype=float)
                layer_name = f"{os.path.basename(job_folder)}_{os.path.basename(latest_run)}_corrected"
                layer = self.viewer.add_points(points, size=30, face_color='green', name=layer_name)

                # Store info
                self.layer_run_map[layer.name] = {
                    "run_folder": latest_run,
                    "status_path": status_path
                }
                loaded_count += 1

        QMessageBox.information(
            self,
            "Load Corrected",
            f"Loaded corrected points from {loaded_count} folder(s)."
        )

    # ---------------------------
    #   Save / Prompt for status
    # ---------------------------
    def ask_user_for_save_status(self):
        """
        Ask the user whether they want to save these points as 'rerun' or 'all_complete'.
        Returns the selected status or None if the user cancels.
        """
        display_to_status = {
            "Needs Re-Run": "rerun",
            "Reviewed and Accepted (Finalized)": "all_complete",
        }
        status_options = list(display_to_status.keys())
        default_index = 0  # "Needs Re-Run" is the default
        chosen_display, ok = QInputDialog.getItem(
            self,
            "Select Save Status",
            "Choose review status for these traces:",
            status_options,
            default_index,
            editable=False
        )
        if ok:
            return display_to_status.get(chosen_display)
        return None

    def save_all_points(self):
        """
        For each loaded layer, save the points to 'corrected_points.json'
        in that layer's run folder. Then set the corresponding status.json
        to either 'rerun' or 'all_complete', depending on user input.
        """
        if not self.layer_run_map:
            QMessageBox.information(self, "No Layers", "No points layers are loaded by this widget.")
            return

        # Prompt for status
        chosen_status = self.ask_user_for_save_status()
        if chosen_status is None:
            return  # User canceled

        saved_count = 0
        for layer_name, info_dict in self.layer_run_map.items():
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                if isinstance(layer, napari.layers.Points):
                    run_folder = info_dict["run_folder"]
                    status_path = info_dict["status_path"]

                    out_path = os.path.join(run_folder, "corrected_points.json")
                    # downsample to original points
                    points = layer.data.tolist()
                    if is_downsample_enabled(self.viewer):
                        factor = get_downsample_factor(self.viewer)
                        points = to_original_points(points, factor)

                    point_axes = None
                    layer_meta = getattr(layer, "metadata", {}) or {}
                    axes_candidate = layer_meta.get("tubulemap_point_axes")
                    if isinstance(axes_candidate, (list, tuple)):
                        point_axes = [str(axis).strip().lower() for axis in axes_candidate]
                    else:
                        point_dim = layer.data.shape[1] if layer.data.ndim == 2 else 0
                        if point_dim >= 5:
                            point_axes = ["t", "c", "z", "y", "x"]
                        else:
                            point_axes = ["z", "y", "x"]

                    with open(out_path, 'w') as f:
                        json.dump({'points': points, 'point_axes': point_axes}, f, indent=4)

                    # Update status 
                    update_status(status_path, chosen_status)
                    saved_count += 1

        QMessageBox.information(
            self,
            "Save All Points",
            f"Saved corrected_points.json for {saved_count} layer(s) and set status to '{chosen_status}'."
        )

    def save_active_points(self):
        """
        Save only the currently active points layer as 'corrected_points.json',
        then set the corresponding status.json to user-chosen status.
        """
        active_layer = self.viewer.layers.selection.active
        if not isinstance(active_layer, napari.layers.Points):
            QMessageBox.information(self, "No Points Layer", "Please select a points layer to save.")
            return

        layer_name = active_layer.name
        info_dict = self.layer_run_map.get(layer_name)
        if not info_dict:
            QMessageBox.warning(
                self,
                "Layer Not Found in Map", 
                f"Layer '{layer_name}' was not loaded by this widget."
            )
            return

        # Prompt for status
        chosen_status = self.ask_user_for_save_status()
        if chosen_status is None:
            return  # User canceled

        run_folder = info_dict["run_folder"]
        status_path = info_dict["status_path"]

        out_path = os.path.join(run_folder, "corrected_points.json")
        # downsample to original points
        points = active_layer.data.tolist()
        if is_downsample_enabled(self.viewer):
            factor = get_downsample_factor(self.viewer)
            points = to_original_points(points, factor)

        point_axes = None
        layer_meta = getattr(active_layer, "metadata", {}) or {}
        axes_candidate = layer_meta.get("tubulemap_point_axes")
        if isinstance(axes_candidate, (list, tuple)):
            point_axes = [str(axis).strip().lower() for axis in axes_candidate]
        else:
            point_dim = active_layer.data.shape[1] if active_layer.data.ndim == 2 else 0
            if point_dim >= 5:
                point_axes = ["t", "c", "z", "y", "x"]
            else:
                point_axes = ["z", "y", "x"]

        with open(out_path, 'w') as f:
            json.dump({'points': points, 'point_axes': point_axes}, f, indent=4)

        # Update status
        update_status(status_path, chosen_status)

        QMessageBox.information(
            self,
            "Save Selected Corrected Points",
            f"Saved selected layer '{layer_name}' to corrected_points.json in: {out_path}\n"
            f"Status file set to '{chosen_status}'."
        )

    def finalize_selected_traces(self):
        """
        Set status='all_complete' for all currently selected points layers that were
        loaded by this widget.
        """
        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            QMessageBox.information(self, "No Selection", "Select one or more points layers to finalize.")
            return

        finalized_count = 0
        skipped_count = 0
        for layer in selected_layers:
            if not isinstance(layer, napari.layers.Points):
                skipped_count += 1
                continue
            info_dict = self.layer_run_map.get(layer.name)
            if not info_dict:
                skipped_count += 1
                continue
            status_path = info_dict["status_path"]
            update_status(status_path, "all_complete")
            finalized_count += 1

        if finalized_count == 0:
            QMessageBox.information(
                self,
                "Finalize Selected Traces",
                "No selected layers were eligible for finalization.\n"
                "Select points layers loaded by this widget.",
            )
            return

        QMessageBox.information(
            self,
            "Finalize Selected Traces",
            f"Set status to 'all_complete' for {finalized_count} selected layer(s)."
            + (f"\nSkipped {skipped_count} layer(s)." if skipped_count else ""),
        )

    # ---------------------------
    #   Delete
    # ---------------------------
    def delete_points_layers(self):
        """
        Remove all points layers that were loaded via this widget from the Napari viewer.
        """
        if not self.layer_run_map:
            QMessageBox.information(self, "No Layers", "No points layers loaded by this widget.")
            return

        removed_count = 0
        for layer_name in list(self.layer_run_map.keys()):
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
                removed_count += 1
            # Remove from map regardless
            self.layer_run_map.pop(layer_name, None)

        QMessageBox.information(
            self,
            "Delete Layers",
            f"Removed {removed_count} layer(s) from the viewer."
        )

    # ---------------------------
    #   Utility
    # ---------------------------
    def _get_latest_run_folder(self, job_folder: str) -> str:
        """
        Given a job folder, e.g. 'Human_in_loop/Track_nephron_1.json',
        find subdirectories named 'Run_X' and return the one with the max X.

        Returns:
            str: path to the latest run folder, or None if none found.
        """
        run_folders = []
        for d in os.listdir(job_folder):
            full_d = os.path.join(job_folder, d)
            if os.path.isdir(full_d) and d.startswith("Run_"):
                # Parse run index
                try:
                    run_index = int(d.split('_')[1])
                    run_folders.append((run_index, full_d))
                except (IndexError, ValueError):
                    pass
        if not run_folders:
            return None
        # Sort by run index descending
        run_folders.sort(key=lambda x: x[0], reverse=True)
        return run_folders[0][1]
