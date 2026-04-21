from pathlib import Path
import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
)

from tubulemap.utils.zarr_resolution import inspect_zarr_source, open_level_array


class ZarrLoaderWidget(QWidget):
    def __init__(self, viewer):
        """Initialize the instance state."""
        super().__init__()
        self.viewer = viewer

        # Layout and button
        layout = QVBoxLayout()
        self.load_button = QPushButton('Load Zarr Volume')

        layout.addWidget(self.load_button)
        self.setLayout(layout)

        # Connect button to function
        self.load_button.clicked.connect(self.load_zarr_volume)

    def _sample_contrast_limits(self, data):
        """Estimate contrast limits from a downsampled in-memory sample."""
        shape = getattr(data, "shape", None)
        if shape is None:
            return None

        try:
            sampling = []
            for dim in shape:
                dim = int(dim)
                if dim <= 256:
                    sampling.append(slice(None))
                else:
                    step = max(1, dim // 256)
                    sampling.append(slice(0, dim, step))
            sample = np.asarray(data[tuple(sampling)])
        except Exception:
            return None

        if sample.size == 0:
            return None
        finite = np.isfinite(sample)
        if not np.any(finite):
            return None
        values = sample[finite]
        try:
            low = float(np.percentile(values, 1.0))
            high = float(np.percentile(values, 99.0))
        except Exception:
            return None
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        if high <= low:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            if vmax <= vmin:
                return None
            low, high = vmin, vmax
        return (low, high)

    @staticmethod
    def _dtype_contrast_limits(data):
        """Return dtype-based limits as a fallback."""
        dtype = getattr(data, "dtype", None)
        if dtype is None:
            return None
        try:
            np_dtype = np.dtype(dtype)
        except Exception:
            return None

        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            if info.max > info.min:
                return (float(info.min), float(info.max))
        return None

    def _auto_adjust_contrast(self, layer, source_meta=None):
        """Apply robust contrast limits so newly loaded volumes are visible."""
        data = getattr(layer, "data", None)
        if data is None:
            return

        candidates = []

        # Prefer level-0 source data (raw intensities) when metadata is available.
        if isinstance(source_meta, dict):
            try:
                candidates.append(open_level_array(source_meta, 0))
            except Exception:
                pass

        # Then consider layer data levels as loaded by napari/plugin.
        if isinstance(data, (list, tuple)) and len(data) > 0:
            candidates.append(data[0])   # finest level
            if len(data) > 1:
                candidates.append(data[-1])  # coarsest as fallback
        else:
            candidates.append(data)

        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            key = id(candidate)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            limits = self._sample_contrast_limits(candidate)
            if limits is not None:
                layer.contrast_limits = limits
                return

        # Fallback to dtype limits when sampling cannot estimate a valid range.
        for candidate in unique_candidates:
            limits = self._dtype_contrast_limits(candidate)
            if limits is not None:
                layer.contrast_limits = limits
                return

        # Last resort, ask napari to recompute.
        try:
            layer.reset_contrast_limits()
        except Exception:
            pass

    def _set_source_metadata(self, layers, source_meta, *, adjust_contrast):
        """Attach resolved source metadata to loaded layers and optionally normalize contrast."""
        for layer in layers or []:
            if hasattr(layer, "metadata"):
                layer.metadata["tubulemap_source_resolution"] = source_meta
            if not adjust_contrast:
                continue
            layer_type = type(layer).__name__.lower()
            if "image" in layer_type:
                self._auto_adjust_contrast(layer, source_meta=source_meta)

    def _open_with_napari_default(self, directory):
        """Use napari default reader resolution to match drag-and-drop behavior."""
        try:
            return self.viewer.open(directory)
        except Exception:
            return None

    def load_zarr_volume(self):
        """Load zarr volume."""
        # Open a file dialog to select a Zarr folder
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Open Zarr Volume", "", options=options)

        if directory:
            try:
                source_meta = inspect_zarr_source(directory)
            except Exception as exc:
                QMessageBox.critical(self, "Load Failed", f"Could not inspect zarr source:\n{exc}")
                return

            # First try napari's default reader resolution path; this most closely matches drag-and-drop.
            opened_layers = self._open_with_napari_default(directory)
            if opened_layers:
                # Keep plugin-selected contrast behavior to avoid dark/black multiscale OME views.
                self._set_source_metadata(opened_layers, source_meta, adjust_contrast=False)
                return

            if source_meta.get("source_kind") == "ome":
                try:
                    opened_layers = self.viewer.open(directory, plugin="napari-ome-zarr")
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "OME-Zarr Reader Missing",
                        "Failed to open with plugin 'napari-ome-zarr'.\n"
                        "Install the plugin and retry.\n\n"
                        f"Error: {exc}",
                    )
                    return

                # For OME, keep plugin-selected rendering defaults (contrast/gamma/scale).
                self._set_source_metadata(opened_layers, source_meta, adjust_contrast=False)
                return

            levels = source_meta.get("levels", [])
            if not levels:
                QMessageBox.critical(self, "Load Failed", "No levels were found in the selected zarr source.")
                return

            layer_name = Path(directory).name
            if len(levels) > 1:
                multiscale_data = [open_level_array(source_meta, idx) for idx in range(len(levels))]
                layer = self.viewer.add_image(multiscale_data, multiscale=True, name=layer_name)
            else:
                image_data = open_level_array(source_meta, 0)
                layer = self.viewer.add_image(image_data, name=layer_name)

            layer.metadata["tubulemap_source_resolution"] = source_meta
            self._auto_adjust_contrast(layer, source_meta=source_meta)
