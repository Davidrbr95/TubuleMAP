from pathlib import Path
import numpy as np
import napari
from magicgui import magicgui

from tubulemap.cellpose_tracker.parameters import TracingParameters

DEFAULT_MODEL_SUITE = Path(__file__).resolve().parent / "models"
AVAILABLE_MODELS = (
    sorted(
        model_path.name
        for model_path in DEFAULT_MODEL_SUITE.iterdir()
        if model_path.is_file() and not model_path.name.startswith(".")
    )
    if DEFAULT_MODEL_SUITE.exists()
    else ["CUBICcortex2"]
)

if "CUBICcortex2" in AVAILABLE_MODELS:
    DEFAULT_STARTING_MODEL = "CUBICcortex2"
elif AVAILABLE_MODELS:
    DEFAULT_STARTING_MODEL = AVAILABLE_MODELS[0]
else:
    DEFAULT_STARTING_MODEL = "CUBICcortex2"

# Define a function that accepts the parameters
@magicgui(
    auto_call=False,
    call_button="Run Tubule Trace",
    name={"label": "Trace Name"},
    kp_path={"label": "Starting Points JSON", "mode": "r", "filter": "*.json"},
    kp_layer={"label": "Starting Points Layer", "widget_type": "ComboBox", "choices": [""]},
    data_set_path={"label": "Image Zarr Folder", "mode": "d"},
    data_layer={"label": "Image Layer", "widget_type": "ComboBox", "choices": [""]},
    run_level={"label": "Run Resolution Level", "widget_type": "ComboBox", "choices": [0]},
    run_time_index={"label": "Run Time Index"},
    run_channel_index={"label": "Run Channel Index"},
    auto_scale_for_level={"label": "Auto-scale Parameters For Level"},
    save_dir={"label": "Output Folder", "mode": "d"},
    starting_model={"label": "Starting Segmentation Model", "choices": AVAILABLE_MODELS},
    kp_source={"visible": False},
    data_source={"visible": False},
    should_cancel={"visible": False},
    multiprocessing={"visible": False},
    model_suite={"visible": False},
    w={"visible": False},
    stepsize={"visible": False},
    diameter={"visible": False},
    jitter={"visible": False},
    iterations={"visible": False},
    save_rate={"visible": False},
    dim={"visible": False},
    trace_savename={"visible": False},
    use_rotations={"visible": False},
    use_ultrack={"visible": False},
    use_adaptive_diameter={"visible": False},
    use_recenter_point={"visible": False},
    vector_method={"visible": False},
    overwite_w_rot={"visible": False},
    break_distance={"visible": False},
    adapt_diam_lower={"visible": False},
    adapt_diam_upper={"visible": False},
    adapt_window={"visible": False},
    scale_jitter={"visible": False},
    scale_stepsize={"visible": False},
)
def run_trace(
    data_source: bool = True,
    kp_source: bool = True,
    name: str = "demo",
    kp_path: Path = Path(""),
    kp_layer: str = "",
    data_set_path: Path = Path(""),
    data_layer: str = "",
    run_level: int = 0,
    run_time_index: int = 0,
    run_channel_index: int = 0,
    auto_scale_for_level: bool = True,
    save_dir: Path = Path("Demo"),
    starting_model: str = DEFAULT_STARTING_MODEL,
    should_cancel=None,
    model_suite: str = str(DEFAULT_MODEL_SUITE),
    multiprocessing: bool = False,
    w: float = 0.7,
    stepsize: int = 15,
    diameter: float = 81.0,
    jitter: int = 30,
    iterations: int = 20,
    save_rate: int = 5,
    dim: int = 200,
    trace_savename: str = "result_trace",
    use_rotations: bool = True,
    use_ultrack: bool = True,
    use_adaptive_diameter: bool = True,
    use_recenter_point: bool = True,
    vector_method: str = "traditional",
    overwite_w_rot: bool = True,
    break_distance: int = 10,
    adapt_diam_lower: float = 5.0,
    adapt_diam_upper: float = 120.0,
    adapt_window: int = 15,
    scale_jitter: float = 3.0,
    scale_stepsize: float = 5.0,
):
    """Run trace."""
    trace_parameters = locals()
    run_core_function(**trace_parameters)

def run_core_function(**params):
    """Run core function."""
    from tubulemap.cellpose_tracker.initialization import (
        initialize_tracking_state,
        load_data,
        setup_logging_and_folders,
    )
    from tubulemap.cellpose_tracker.core import looping_through_points
    import json
    import time

    # Create a class that stores all the run parameters
    # trace = TracingParameters(**locals())
    trace = TracingParameters(**params)
    # Setup the folder structure and logging
    setup_logging_and_folders(trace)
    trace.log.info('Start tracing')

    # Generate napari viewer object
    trace.napari_viewer =napari.current_viewer()

    # Load data key points and zarr volume
    load_data(trace)


    # Set parameters related to loaded trace and load model
    initialize_tracking_state(trace)

    trace.log_attribute_names()

    trace.log.info("Starting the trace")
    starting_trace = time.time()
    for x in looping_through_points(trace):
        yield x
    ending_trace = time.time()
    
    trace.duration = ending_trace-starting_trace # TODO: ADD VARIABLE
    trace.record_current_node_params()
    trace.write_detailed_parameters()

    ## Load points that were just completed into the napari GUI
    with open(trace.result_trace_path+'.json', 'r') as f:
        points_data = json.load(f)
        
    points = np.array(points_data['points'])

    trace_path = Path(trace.result_trace_path)
    points_name = f"{trace_path.parent.name}_{trace_path.name}"

    data_to_yield = {
        'points': points,
        'points_name': points_name,
    }

    print('Final yield')
    if not trace.multiprocessing:
        yield data_to_yield
