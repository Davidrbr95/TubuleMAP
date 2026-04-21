from magicgui import magicgui
from pathlib import Path
import numpy as np
import napari
from tubulemap.cellpose_tracker.initialization import *
from tubulemap.cellpose_tracker.core_post_processing import *
from tubulemap.cellpose_tracker.parameters import TracingParameters
from line_profiler import LineProfiler
# Define a function that accepts the parameters
@magicgui(
    data_set_path={"label": "Data Set Path", "mode": "d"},  # 'd' mode for directory
    kp_source={"visible": False},
    data_source={"visible": False},
    auto_call=False
)
def post_process_trace(
    data_source: bool = True,
    kp_source: bool = True,
    data_set_path: Path = Path(""),
    name: str = 'test',
    stepsize: int = 10,
    diameter: int = 50,
    jitter: int = 10,
    save_rate: int = 5,
    resample_step_size: int = 5,
    dim: int = 150,
    use_jitter: bool = True,
    use_rotations: bool = True,
    use_adaptive_diameter: bool = True,
    vector_method: str = 'traditional',
    multiprocessing = True,
    model_suite = "cellpose_tracker/models",
    save_dir = "./Traces",
    starting_model="CUBICcortex2",
    kp_path = ".",
    write_ply = True
    ):
    """Compute post process trace."""
    trace_parameters = locals()
    run_post_process_trace(**trace_parameters)

def run_post_process_trace(
    data_source: bool = True,
    kp_source: bool = True,
    name: str = 'test',
    stepsize: int = 10,
    diameter: int = 50,
    resample_step_size: int = 5,
    jitter: int = 10,
    save_rate: int = 5,
    dim: int = 150,
    use_jitter: bool = True,
    use_rotations: bool = True,
    use_adaptive_diameter: bool = True,
    data_set_path: Path = Path(""),
    vector_method: str = 'traditional',
    should_cancel=None,
    model_suite = "cellpose_tracker/models",
    multiprocessing = True,
    save_dir = "./Traces",
    starting_model="CUBICcortex2",
    kp_path = ".",
    write_ply = True,
    **kwargs
    ):

    # Create a class that stores all the run parameters
    """Run post process trace."""
    trace = TracingParameters(**locals())

    # Setup the folder structure and logging
    setup_logging_and_folders(trace)
    trace.log.info('Start tracing')

    # Generate napari viewer object
    trace.napari_viewer =napari.current_viewer()
    print('HERE1')
    print(trace.kp_path)
    # Load data key points and zarr volume
    load_data(trace)

    # Set parameters related to loaded trace and load model
    initialize_tracking_state(trace)

    trace.log_attribute_names()

    trace.log.info("Starting the trace")
    starting_trace = time.time()
    for x in looping_through_points(trace):
        yield x
        
    trace.record_current_node_params()
    trace.write_detailed_parameters()
    ending_trace = time.time()
    
