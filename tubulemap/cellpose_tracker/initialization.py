import numpy as np
import pandas as pd
import logging
import json
import shutil
import os
from tubulemap.cellpose_tracker.io_utils import *
from tubulemap.cellpose_tracker.geometry import *
from tubulemap.utils.zarr_resolution import (
    apply_parameter_scaling_to_trace,
    create_run_volume_view,
    get_axis_size_for_level,
    has_translation_mismatch,
    inspect_zarr_source,
    open_level_array,
    scale_curve_nodes_xyz,
)

def _as_bool(value):
    """Return a robust boolean for flags that may come from JSON or CLI inputs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_cellpose_runtime(trace, torch):
    """Resolve a safe Cellpose runtime device and GPU flag for this environment."""
    force_cpu = os.getenv("tubulemap_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_cpu:
        trace.log.warning("tubulemap_FORCE_CPU is set. Forcing CPU execution for Cellpose.")
        return False, "cpu"

    requested_gpu = _as_bool(getattr(trace, "use_GPU", True))
    requested_device = str(getattr(trace, "cuda_device", "cuda:0") or "cuda:0")
    cuda_available = bool(torch.cuda.is_available())

    if requested_gpu and cuda_available:
        try:
            torch.device(requested_device)
            return True, requested_device
        except Exception:
            trace.log.warning(
                "Requested CUDA device '%s' is invalid. Falling back to 'cuda:0'.",
                requested_device,
            )
            return True, "cuda:0"

    if requested_gpu and not cuda_available:
        trace.log.warning(
            "use_GPU=True but CUDA is unavailable. Falling back to CPU for Cellpose."
        )

    return False, "cpu"


def _restore_points_from_downsample_if_needed(trace, points):
    """Restore downsampled napari points to original pixel coordinates when enabled."""
    if not points:
        return points
    if trace.napari_viewer is None:
        return points

    try:
        from tubulemap.widgets.downsample_control_widget import (
            get_downsample_factor,
            is_downsample_enabled,
            to_original_points,
        )
    except Exception:
        return points

    if not is_downsample_enabled(trace.napari_viewer):
        return points

    first_point = list(points[0]) if points else []
    if len(first_point) < 5:
        return points

    factor = float(get_downsample_factor(trace.napari_viewer))
    restored = to_original_points(points, factor)
    trace.log.info(
        "Converted downsampled points to original coordinates using factor=%s",
        factor,
    )
    return restored


def _in_bounds_ratio_zyx(points_zyx, shape_zyx):
    """Return fraction of points that lie within zyx bounds."""
    if not points_zyx:
        return 0.0
    if shape_zyx is None:
        return 0.0
    z_max, y_max, x_max = [int(v) for v in shape_zyx]
    ok = 0
    for point in points_zyx:
        if len(point) < 3:
            continue
        z, y, x = float(point[0]), float(point[1]), float(point[2])
        if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
            ok += 1
    return ok / float(len(points_zyx))


def _infer_3d_point_order_if_needed(trace, points_zyx, raw_points, point_axes):
    """
    Infer 3D point axis order when metadata is missing.

    If points were saved as [x,y,z] but interpreted as [z,y,x], tracing can jump
    unpredictably. We detect this by testing all 3D permutations against current
    volume bounds and selecting the best-scoring order only when clearly better.
    """
    if point_axes is not None:
        return points_zyx
    if not raw_points:
        return points_zyx

    # Only infer for plain 3D points.
    if any(len(point) != 3 for point in raw_points if isinstance(point, (list, tuple))):
        return points_zyx

    shape_zyx = getattr(trace.volume, "shape", None)
    if shape_zyx is None or len(shape_zyx) != 3:
        return points_zyx

    base_ratio = _in_bounds_ratio_zyx(points_zyx, shape_zyx)
    try:
        arr = np.asarray(raw_points, dtype=float)
    except Exception:
        return points_zyx
    if arr.ndim != 2 or arr.shape[1] != 3:
        return points_zyx

    permutations = {
        "zyx": (0, 1, 2),
        "zxy": (0, 2, 1),
        "yzx": (1, 0, 2),
        "yxz": (1, 2, 0),
        "xzy": (2, 0, 1),
        "xyz": (2, 1, 0),
    }

    best_name = "zyx"
    best_points = points_zyx
    best_ratio = base_ratio

    for name, perm in permutations.items():
        candidate = arr[:, perm].tolist()
        ratio = _in_bounds_ratio_zyx(candidate, shape_zyx)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate
            best_name = name

    # Apply only when clearly better to avoid changing already-valid data.
    if best_name != "zyx" and (best_ratio - base_ratio) >= 0.3:
        trace.log.warning(
            "Point axis order inferred as '%s' (in-bounds %.1f%% vs %.1f%% for zyx). "
            "Using inferred order for this run.",
            best_name,
            best_ratio * 100.0,
            base_ratio * 100.0,
        )
        return best_points

    return points_zyx


def setup_logging_and_folders(trace):
    """
    Set up logging and folder structure for saving results.

    This function creates the necessary directory structure for an experiment
    run based on attributes in the given `trace` object. It also configures logging
    by creating a log file in the run folder and setting up a logger.

    The following attributes are added to the `trace` object:
      - experiment_folder: The folder for the current experiment.
      - next_run_id: The next available run number.
      - next_run_folder: The folder for the current run.
      - log_path: The path to the run log file.
      - result_trace_path: The path to the result trace file.

    Args:
        trace: Object that contains the attributes for the current trace operation

    Returns:
        The updated `trace` object with new folder and logging attributes.
    """
    # Create directory structure for saving results
    # Ensure the base save directory exists
    if not os.path.isdir(trace.save_dir):
        os.mkdir(trace.save_dir)

    # Create a subdirectory for the specific experiment
    experiment_folder = os.path.join(trace.save_dir, trace.name)
    trace.experiment_folder = experiment_folder
    if not os.path.isdir(trace.experiment_folder):
        os.mkdir(trace.experiment_folder)

    # Determine the next run number and create a corresponding folder
    next_run = get_max_run_number(trace.experiment_folder) + 1
    trace.next_run_id = next_run
    run_folder = os.path.join(trace.experiment_folder, f'Run_{trace.next_run_id}')
    trace.next_run_folder = run_folder
    if not os.path.isdir(trace.next_run_folder):
        os.mkdir(trace.next_run_folder)
    
    # saving the log
    run_log_path = os.path.join(trace.next_run_folder, 'run.log')
    trace.log_path = run_log_path
    logging.basicConfig(level=logging.INFO, filename = trace.log_path, filemode="w", force=True, format="%(asctime)s - %(levelname)s - %(message)s") 
    trace.log = logging.getLogger("trace_logger")

    # name for the resulting trace file
    result_trace_path = os.path.join(trace.next_run_folder, trace.trace_savename)
    trace.result_trace_path = result_trace_path

    # Name for the status file. If name ends with .json we keep legacy stem behavior.
    job_name = str(trace.name)
    job_stem = job_name[:-5] if job_name.endswith(".json") else job_name
    trace.status_file_path = os.path.join(trace.save_dir, f"{job_stem}_status.json")

    # save run parameters file
    run_parameters_path = os.path.join(trace.next_run_folder, 'run_parameters.json')
    trace.param_json_path = run_parameters_path
    trace.dump_to_json()
    trace._ensure_save_dir()

def load_data(trace):
    """
    Load initial keypoints and Zarr volume into the trace object.

    This function sets up the trace object by loading the starting keypoints and the
    Zarr volume based on the provided configuration in the trace object. It handles
    two scenarios for both the dataset and keypoints:
      - If the dataset is provided via the interactive Napari GUI (i.e., data_source is False),
        it extracts the path from the selected data layer.
      - Otherwise, it uses the provided data_set_path.
      
    For keypoints:
      - If kp_source is True, it copies the provided keypoints file to the run folder.
      - Otherwise, it extracts keypoints from the selected Napari layer and writes them to a JSON file.

    If a ground truth file is specified (trace.ground_truth is not empty), the file is copied
    to the run folder and the ground truth keypoints are loaded; otherwise, ground_truth_curvenode
    is set to None.

    Args:
        trace: An object containing configuration parameters and state.

    Returns:
        The updated trace object with loaded keypoints, volume, and additional attributes.
    """
    trace.log.info('Loading the starting key points and zarr volume')
    # Used if the dataset comes from the interactive napari gui
    if trace.data_source is False:
        trace.log.info('Identifying the path to the volume from the data layer selected in napari')
        data_layer_obj = trace.napari_viewer.layers[trace.data_layer]
        layer_path = getattr(data_layer_obj.source, "path", None)
        layer_meta = getattr(data_layer_obj, "metadata", {}) or {}
        cached_meta = layer_meta.get("tubulemap_source_resolution") if isinstance(layer_meta, dict) else None

        if cached_meta and cached_meta.get("path"):
            trace.data_set_path = str(cached_meta["path"]).rstrip("/\\")
        elif layer_path not in (None, ""):
            trace.data_set_path = str(layer_path).rstrip("/\\")
        else:
            raise ValueError(
                "Selected image layer does not expose a source path or cached zarr metadata. "
                "Use 'Zarr folder path' data source mode."
            )

    trace.log.info('Resolving Zarr source metadata from: {%s}', str(trace.data_set_path))
    source_meta = inspect_zarr_source(trace.data_set_path)
    trace.source_metadata = source_meta
    trace.source_axes = list(source_meta.get("axes", []))

    levels = source_meta.get("levels", [])
    if not levels:
        raise ValueError(f"No resolution levels found for source: {trace.data_set_path}")

    run_level = int(trace.run_level)
    if run_level < 0 or run_level >= len(levels):
        raise ValueError(f"run_level={run_level} is out of range [0, {len(levels)-1}]")

    if run_level > 0 and has_translation_mismatch(source_meta, level_idx=run_level):
        raise ValueError(
            "Selected run_level has a different per-level translation than level 0. "
            "Tracking uses pixel coordinates only for multiscale mapping. "
            "Please run at level 0 for this dataset."
        )

    trace.run_level = run_level
    selected_level = levels[run_level]
    trace.run_level_scale_zyx = [float(v) for v in selected_level.get("scale_zyx", [1.0, 1.0, 1.0])]
    trace.run_level_translation_zyx = [
        float(v) for v in selected_level.get("translation_zyx", [0.0, 0.0, 0.0])
    ]

    t_size = get_axis_size_for_level(source_meta, run_level, "t")
    c_size = get_axis_size_for_level(source_meta, run_level, "c")
    if t_size is None:
        trace.run_time_index = 0
    else:
        trace.run_time_index = int(trace.run_time_index)
        if not (0 <= trace.run_time_index < t_size):
            raise ValueError(f"run_time_index={trace.run_time_index} is out of range [0, {t_size-1}]")

    if c_size is None:
        trace.run_channel_index = 0
    else:
        trace.run_channel_index = int(trace.run_channel_index)
        if not (0 <= trace.run_channel_index < c_size):
            raise ValueError(f"run_channel_index={trace.run_channel_index} is out of range [0, {c_size-1}]")

    level_array = open_level_array(source_meta, run_level)
    trace.volume = create_run_volume_view(
        source_array=level_array,
        axes=trace.source_axes,
        run_time_index=trace.run_time_index,
        run_channel_index=trace.run_channel_index,
    )

    trace.log.info(
        "Loaded source kind=%s run_level=%s scale_zyx=%s t_idx=%s c_idx=%s",
        source_meta.get("source_kind"),
        trace.run_level,
        trace.run_level_scale_zyx,
        trace.run_time_index,
        trace.run_channel_index,
    )

    # Check if there is an associated ground_truth file
    if trace.ground_truth != "" and trace.kp_path is None:
        gt_curvenodes_path = os.path.join(trace.next_run_folder, 'Ground_Truth.json')
        shutil.copyfile(trace.ground_truth, gt_curvenodes_path)
        gt_points_level0 = load_keypoints(trace.ground_truth)
        trace.ground_truth_curvenode = scale_curve_nodes_xyz(
            gt_points_level0,
            trace.run_level_scale_zyx,
            to_run=True,
        )
        trace.curvenode = trace.ground_truth_curvenode[:5]
        apply_parameter_scaling_to_trace(trace)
        return

    elif trace.ground_truth != "" and trace.kp_path is not None:
        gt_curvenodes_path = os.path.join(trace.next_run_folder, 'Ground_Truth.json')
        shutil.copyfile(trace.ground_truth, gt_curvenodes_path)
        gt_points_level0 = load_keypoints(trace.ground_truth)
        trace.ground_truth_curvenode = scale_curve_nodes_xyz(
            gt_points_level0,
            trace.run_level_scale_zyx,
            to_run=True,
        )

    if trace.kp_source:
        with open(trace.kp_path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            raw_points = payload.get("points", [])
            point_axes = payload.get("point_axes")
        else:
            raw_points = payload
            point_axes = None
        if isinstance(point_axes, (list, tuple)):
            point_axes = [str(axis).strip().lower() for axis in point_axes]
        else:
            point_axes = None
    else: # Used if the dataset comes from the interactive napari gui
        points_layer_call = trace.napari_viewer.layers[trace.kp_layer]
        points_layer_meta = getattr(points_layer_call, "metadata", {}) or {}
        layer_point_axes = None
        if isinstance(points_layer_meta, dict):
            axes_candidate = points_layer_meta.get("tubulemap_point_axes")
            if isinstance(axes_candidate, (list, tuple)):
                layer_point_axes = [str(axis).strip().lower() for axis in axes_candidate]

        raw_points = points_layer_call.data.tolist()
        raw_points = _restore_points_from_downsample_if_needed(trace, raw_points)
        point_axes = layer_point_axes

        # Keep legacy behavior unless the layer explicitly declares point axis order.
        # This matches April2025 behavior for manually drawn [z,y,x] napari points.
        normalization_axes = point_axes if point_axes else None
        trace.log.info(
            (
                "Reading starting points from napari layer '%s' with shape=%s, "
                "source_axes=%s, layer_point_axes=%s, normalization_axes=%s"
            ),
            trace.kp_layer,
            getattr(points_layer_call.data, "shape", None),
            trace.source_axes,
            point_axes,
            normalization_axes,
        )

    points_data = normalize_points_to_zyx(
        raw_points,
        source_axes=point_axes if point_axes else None,
    )
    points_data = _infer_3d_point_order_if_needed(
        trace=trace,
        points_zyx=points_data,
        raw_points=raw_points,
        point_axes=point_axes,
    )
    if points_data:
        trace.log.info("First normalized starting point (zyx)=%s", points_data[0])

    starting_curvenodes_path = os.path.join(trace.next_run_folder, 'Starting_Points.json')
    trace.kp_path = starting_curvenodes_path
    with open(trace.kp_path, 'w') as f:
        json.dump({'points': points_data, 'point_axes': ['z', 'y', 'x']}, f, indent=4)

    trace.log.info('Loading the Starting points from the provided file: {%s}', str(trace.kp_path))
    curvenode_level0 = load_keypoints(trace.kp_path)
    trace.curvenode = scale_curve_nodes_xyz(curvenode_level0, trace.run_level_scale_zyx, to_run=True)
    if len(trace.curvenode) < 2:
        raise ValueError(
            "At least two starting points are required to initialize tracing. "
            "Provide points as [z,y,x] or [t,c,z,y,x]."
        )
    apply_parameter_scaling_to_trace(trace)


def initialize_tracking_state(trace):
    """
    Initialize the tracking state by setting up transforms, point lists, and model parameters.

    This function performs several initialization steps:
      - Computes the center transform using generate_center_transform and adds it to the trace.
      - Creates a list of point indices based on the number of keypoints in trace.curvenode.
      - Sets the starting index (last element in the points list) and computes the chunk size.
      - Loads default model parameters from a JSON file.
      - Initializes the Cellpose model with GPU enabled and adds model information to the trace.
    
    Args:
        trace: An object containing tracking parameters and state.
    
    Returns:
        The updated trace object with initialized tracking state attributes.
    """
    trace.log.info('Initialize the tracking state')
    from cellpose import models as Models
    import torch

    trace.center_transform = generate_center_transform(trace)
    trace.points_list = list(np.arange(len(trace.curvenode)))
    if trace.ground_truth!='' and len(trace.ground_truth_curvenode)<=5:
        write_status(
            trace,
            status="all_complete",
            error_msg="Breaking due to end of ground truth +",
        )
        exit()
    trace.start_idx = trace.points_list[-1] # getting error here
    trace.chunk_size = int(trace.stepsize*trace.save_rate+trace.dim/2) # TODO: This will have to be modified whenever the step size is updated

    use_GPU, resolved_device = _resolve_cellpose_runtime(trace, torch)
    trace.use_GPU = use_GPU
    trace.cuda_device = resolved_device
    # with open('tubulemap/cellpose_tracker/default_values.json', 'r') as file:
    #     default_params = json.load(file)

    # trace.starting_model = default_params['starting_model']
    trace.model_name = trace.starting_model 
    # trace.model_suite = default_params['model_suite']
    # print(trace.model_suite, trace.model_name)
    trace.log.info("Loading the initial model {%s}", trace.model_name)
    model = Models.CellposeModel(
        gpu=trace.use_GPU,
        pretrained_model=os.path.join(trace.model_suite, trace.model_name),
        device=torch.device(trace.cuda_device),
    )
    trace.model = model

    
    


    

