import os
import sys
import tempfile
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )


# Keep napari/numba runtime defaults local and writable across launch contexts.
os.environ.setdefault(
    "NAPARI_ASYNC",
    "1",
)

os.environ.setdefault(
    "NAPARI_OCTREE",
    "0",
)

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(
        tempfile.gettempdir(),
        "numba_cache",
    ),
)


import napari

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from tubulemap.widgets import (
    PointsWidget,
    ZarrLoaderWidget,
    TubuleTrackerWidget,
    HumanInLoopWidget,
    DownsampleControlWidget,
)


# ============================================================
# Demo settings
# ============================================================

DEMO_DIR = Path(
    __file__
).resolve().parent


DEMO_ZARR_PATH = (
    DEMO_DIR
    / "Kidney_demo.zarr"
)


DEMO_SEED_PATH = (
    DEMO_DIR
    / "test_seed_points"
    / "test1.json"
)


DEMO_OUTPUT_DIR = (
    DEMO_DIR
    / "test_tracking_results"
)


# ============================================================
# Tracking parameters
# ============================================================

# Tracking parameters used for the demo.
#
# These values are selected specifically for Kidney_demo.zarr
# so that the demo runs quickly and produces a representative
# tubule trace.

DEMO_TRACKING_PARAMETERS = {

    # --------------------------------------------------------
    # Core tracking
    # --------------------------------------------------------

    "diameter": 100.0,

    "use_adaptive_diameter": True,

    "stepsize": 15,

    "iterations": 10,

    "jitter": 30,

    "use_rotations": True,

    "use_ultrack": True,

    "dim": 500,


    # --------------------------------------------------------
    # Adaptive tracing parameters
    # --------------------------------------------------------

    "adapt_diam_lower": 30.0,

    "adapt_diam_upper": 150.0,

    "adapt_window": 10,

    "scale_jitter": 1.0,

    "scale_stepsize": 1.0,
}


# ============================================================
# Zarr resolution settings
# ============================================================

DEMO_RUN_LEVEL = 0

DEMO_TIME_INDEX = 0

DEMO_CHANNEL_INDEX = 0


# ============================================================
# Demo data loading
# ============================================================

def load_demo_zarr(viewer):
    """Load the bundled Kidney_demo.zarr dataset into napari."""

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DEMO_ZARR_PATH.is_dir():
        raise FileNotFoundError(
            f"Demo Zarr dataset was not found:\n"
            f"{DEMO_ZARR_PATH}"
        )


    # --------------------------------------------------------
    # Load Zarr into napari
    # --------------------------------------------------------

    loaded_layers = viewer.open(
        str(DEMO_ZARR_PATH),
        plugin="napari-ome-zarr",
    )


    # --------------------------------------------------------
    # Set layer name
    # --------------------------------------------------------

    zarr_name = (
        DEMO_ZARR_PATH.name
    )


    if len(loaded_layers) == 1:

        loaded_layers[0].name = (
            zarr_name
        )

    else:

        for index, layer in enumerate(
            loaded_layers
        ):

            layer.name = (
                f"{zarr_name} [{index}]"
            )


# ============================================================
# Demo configuration
# ============================================================

def configure_demo(tubulemap_widget):
    """Configure TubuleMAP with the bundled demo dataset."""


    # --------------------------------------------------------
    # Check demo files
    # --------------------------------------------------------

    if not DEMO_ZARR_PATH.is_dir():
        raise FileNotFoundError(
            f"Demo Zarr dataset was not found:\n"
            f"{DEMO_ZARR_PATH}"
        )


    if not DEMO_SEED_PATH.is_file():
        raise FileNotFoundError(
            f"Demo seed point file was not found:\n"
            f"{DEMO_SEED_PATH}"
        )


    DEMO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Input source
    # --------------------------------------------------------

    tubulemap_widget.kp_source.setCurrentText(
        "JSON file"
    )


    tubulemap_widget.data_source.setCurrentText(
        "Zarr folder path"
    )


    # --------------------------------------------------------
    # Run widget
    # --------------------------------------------------------

    run_widget = (
        tubulemap_widget.run_trace_widget
    )


    # --------------------------------------------------------
    # Demo paths
    # --------------------------------------------------------

    run_widget[
        "kp_path"
    ].value = str(
        DEMO_SEED_PATH
    )


    run_widget[
        "data_set_path"
    ].value = str(
        DEMO_ZARR_PATH
    )


    run_widget[
        "save_dir"
    ].value = str(
        DEMO_OUTPUT_DIR
    )


    run_widget[
        "name"
    ].value = "test1"


    # --------------------------------------------------------
    # Zarr level
    # --------------------------------------------------------

    if hasattr(
        run_widget,
        "run_level",
    ):
        run_widget.run_level.value = (
            DEMO_RUN_LEVEL
        )


    # --------------------------------------------------------
    # Time index
    # --------------------------------------------------------

    if hasattr(
        run_widget,
        "run_time_index",
    ):
        run_widget.run_time_index.value = (
            DEMO_TIME_INDEX
        )


    # --------------------------------------------------------
    # Channel index
    # --------------------------------------------------------

    if hasattr(
        run_widget,
        "run_channel_index",
    ):
        run_widget.run_channel_index.value = (
            DEMO_CHANNEL_INDEX
        )


    # --------------------------------------------------------
    # Automatic coordinate scaling
    # --------------------------------------------------------

    if hasattr(
        run_widget,
        "auto_scale_for_level",
    ):
        run_widget.auto_scale_for_level.value = (
            True
        )


    # --------------------------------------------------------
    # Tracking parameters
    # --------------------------------------------------------

    tubulemap_widget._tracking_param_overrides.update(
        DEMO_TRACKING_PARAMETERS
    )


    tubulemap_widget._sync_tracking_parameters_to_magicgui()


    # --------------------------------------------------------
    # Refresh widget state
    # --------------------------------------------------------

    tubulemap_widget.update_widgets()


    tubulemap_widget._sync_run_resolution_controls()


# ============================================================
# Create widgets
# ============================================================

def create_widgets(viewer):
    """Create and return the custom widgets."""


    # --------------------------------------------------------
    # Downsample control
    # --------------------------------------------------------

    downsample_control_widget = (
        DownsampleControlWidget(
            viewer
        )
    )


    # --------------------------------------------------------
    # Points widget
    # --------------------------------------------------------

    points_widget = (
        PointsWidget(
            viewer
        )
    )


    # --------------------------------------------------------
    # Zarr loader
    # --------------------------------------------------------

    zarr_loader_widget = (
        ZarrLoaderWidget(
            viewer
        )
    )


    # --------------------------------------------------------
    # Tubule tracker
    # --------------------------------------------------------

    tubulemap_widget = (
        TubuleTrackerWidget(
            viewer
        )
    )


    # --------------------------------------------------------
    # Human-in-the-loop
    # --------------------------------------------------------

    human_in_loop_widget = (
        HumanInLoopWidget(
            viewer
        )
    )


    # --------------------------------------------------------
    # Configure TubuleMAP for bundled demo
    # --------------------------------------------------------

    configure_demo(
        tubulemap_widget
    )


    # --------------------------------------------------------
    # Left widgets
    # --------------------------------------------------------

    left_widgets = [

        downsample_control_widget,

        zarr_loader_widget,

    ]


    # --------------------------------------------------------
    # Right widgets
    # --------------------------------------------------------

    right_widgets = [

        points_widget,

        tubulemap_widget,

        human_in_loop_widget,

    ]


    return (
        left_widgets,
        right_widgets,
    )


# ============================================================
# Setup layout
# ============================================================

def setup_layout(
    left_widgets,
    right_widgets,
):
    """Setup the layout and add the provided widgets."""


    # --------------------------------------------------------
    # Left layout
    # --------------------------------------------------------

    layout_left = QWidget()


    left_layout = QVBoxLayout(
        layout_left
    )


    for widget in left_widgets:

        left_layout.addWidget(
            widget
        )


    # --------------------------------------------------------
    # Right tab layout
    # --------------------------------------------------------

    layout_right = QTabWidget()


    for widget in right_widgets:

        tab = QWidget()


        tab_layout = QVBoxLayout()


        tab_layout.addWidget(
            widget
        )


        tab.setLayout(
            tab_layout
        )


        layout_right.addTab(
            tab,
            widget.name,
        )


    return (
        layout_left,
        layout_right,
    )


# ============================================================
# Main
# ============================================================

def main():
    """Launch napari with the bundled TubuleMAP demo."""


    # --------------------------------------------------------
    # Create napari viewer
    # --------------------------------------------------------

    viewer = napari.Viewer()


    # --------------------------------------------------------
    # Load bundled Kidney_demo.zarr
    # --------------------------------------------------------

    load_demo_zarr(
        viewer
    )


    # --------------------------------------------------------
    # Create custom widgets
    # --------------------------------------------------------

    left_widgets, right_widgets = (
        create_widgets(
            viewer
        )
    )


    # --------------------------------------------------------
    # Setup UI
    # --------------------------------------------------------

    layout_left, layout_right = (
        setup_layout(
            left_widgets,
            right_widgets,
        )
    )


    # --------------------------------------------------------
    # Add left dock
    # --------------------------------------------------------

    viewer.window.add_dock_widget(
        layout_left,
        area="left",
        name="Main operations",
    )


    # --------------------------------------------------------
    # Add right dock
    # --------------------------------------------------------

    viewer.window.add_dock_widget(
        layout_right,
        area="right",
        name="Tracking",
    )


    # --------------------------------------------------------
    # Start napari
    # --------------------------------------------------------

    napari.run()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()