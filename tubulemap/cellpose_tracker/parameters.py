import json
import pandas as pd
import os
import copy
import SimpleITK as sitk
import orjson
import threading, queue, atexit, time

DEFAULT_MODEL_SUITE = os.path.join(os.path.dirname(__file__), "models")

ALL_PARAMETERS = {
    # ---------------------
    # Tracing parameters
    # ---------------------
    "adapt_diam_lower": {
        "default": 5,
        "comment": "Minimum adaptive diameter (pixels) allowed when updating diameter."
    },
    "adapt_diam_upper": {
        "default": 120,
        "comment": "Maximum adaptive diameter (pixels) allowed when updating diameter."
    },
    "adapt_window":{
        "default": 15,
        "comment": "Number of recent diameter estimates averaged for adaptive updates."
    },
    "latest_diameters":{
        "default": [],
        "comment": "History buffer of recent diameter estimates used by adaptive updates."
    }, 
    "jitter": {
        "default": 30,
        "comment": "Diameter sweep half-width in pixels; troubleshooting tests [diameter-jitter, diameter, diameter+jitter]."
    },
    "use_adaptive_diameter": {
        "default": True,
        "comment": "If True, update diameter (and jitter) from recent segmentation measurements."
    },
    "diameter": {
        "default": 81,
        "comment": "Primary Cellpose diameter in pixels for the current node."
    },
    "scale_jitter":{
        "default":3,
        "comment": "Divisor used to derive jitter from adaptive radius (jitter = radius / scale_jitter)."
    },
    "scale_stepsize":{
        "default": 5,
        "comment": "Divisor used to derive stepsize from adaptive radius when enabled."
    },
    "node_parameter_record":{
        "default":{},
        "comment": "In-memory map of per-node parameter snapshots keyed by node index."
    },

    # ---------------------
    # Model-related
    # ---------------------
    "model": {
        "default": None,
        "comment": "Loaded Cellpose model instance used for segmentation."
    },
    "model_name": {
        "default": None,
        "comment": "Name of the currently active segmentation model file."
    },
    "model_suite": {
        "default": DEFAULT_MODEL_SUITE,
        "comment": "Directory containing available Cellpose model files."
    },
    "starting_model": {
        "default": "CUBICcortex2",
        "comment": "Model file name used to initialize tracing."
    },

    # ---------------------
    # Data reading/loading
    # ---------------------
    "data_layer": {
        "default": '',
        "comment": "Name of the napari image layer used as input when data_source is False."
    },
    "data_set_path": {
        "default": ".",
        "comment": "Filesystem path to the input zarr or OME-zarr source."
    },
    "data_source": {
        "default": True,
        "comment": "Input mode flag: True uses data_set_path, False uses the selected napari image layer."
    },
    "run_level": {
        "default": 0,
        "comment": "Resolution level index used for tracking computations."
    },
    "run_time_index": {
        "default": 0,
        "comment": "Time index used when the source has a time axis."
    },
    "run_channel_index": {
        "default": 0,
        "comment": "Channel index used when the source has a channel axis."
    },
    "auto_scale_for_level": {
        "default": True,
        "comment": "Automatically scale voxel parameters for the selected run level."
    },
    "source_metadata": {
        "default": None,
        "comment": "Resolved source metadata (axes, levels, scales, translations) for the current input."
    },
    "source_axes": {
        "default": [],
        "comment": "Axis order resolved from source metadata (for example zyx or tczyx)."
    },
    "run_level_scale_zyx": {
        "default": [1.0, 1.0, 1.0],
        "comment": "Scale factors that map run-level pixels to level-0 pixels (z,y,x)."
    },
    "run_level_translation_zyx": {
        "default": [0.0, 0.0, 0.0],
        "comment": "Per-level zyx translation values stored for multiscale validation."
    },
    "kp_layer": {
        "default": '',
        "comment": "Name of the napari points layer used as starting keypoints when kp_source is False."
    },
    "kp_path": {
        "default": ".",
        "comment": "Path to starting keypoints JSON when kp_source is True."
    },
    "kp_source": {
        "default": True,
        "comment": "Keypoint input mode flag: True uses kp_path, False uses kp_layer."
    },
    "multiprocessing": {
        "default": False,
        "comment": "True when tracing is launched from multiprocessing or batch pipelines."
    },

    # ---------------------
    # Data saving & logging
    # ---------------------
    "name": {
        "default": 'demo',
        "comment": "Trace job name used for output folders and status files."
    },
    "experiment_folder": {
        "default": None,
        "comment": "Root folder for grouped runs (when used by a workflow)."
    },
    "log": {
        "default": None,
        "comment": "Logger instance used during tracing."
    },
    "log_path": {
        "default": None,
        "comment": "Path to the run log file."
    },
    "next_run_id": {
        "default": 0,
        "comment": "Numeric run index used to create the next run folder."
    },
    "next_run_folder": {
        "default": None,
        "comment": "Output folder for the active run."
    },
    "param_json_path": {
        "default": None,
        "comment": "Path to the run_parameters.json snapshot."
    },
    "result_trace_path": {
        "default": None,
        "comment": "Path prefix for final traced points output."
    },
    "save_dir": {
        "default": "Demo",
        "comment": "Base output directory for all trace runs."
    },
    "save_rate": {
        "default": 5,
        "comment": "Save intermediate progress every N iterations."
    },
    "status_file_path": {
        "default": None,
        "comment": "Path to the status JSON file for this run."
    },
    "trace_savename": {
        "default": "result_trace",
        "comment": "Base filename for saved trace point outputs."
    },
    "napari_viewer": {
        "default": None,
        "comment": "Active napari viewer instance, when running from the GUI."
    },

    # ---------------------
    # Vector/transform
    # ---------------------
    "center_transform": {
        "default": None,
        "comment": "Centering transform applied to slice geometry before sampling."
    },
    "vectors": {
        "default": [],
        "comment": "History of direction vectors used to propagate the trace."
    },
    "vector_method": {
        "default": 'traditional',
        "comment": "Vector update strategy name."
    },
    "w": {
        "default": 0.7,
        "comment": "Blend weight between current and previous direction vectors during initialization."
    },

    # ---------------------
    # Image loading
    # ---------------------
    "chunk_size": {
        "default": 100,
        "comment": "Number of points or steps represented by each loaded image chunk."
    },
    "current_chunk": {
        "default": None,
        "comment": "Currently loaded image chunk used for sampling."
    },
    "dim": {
        "default": 200,
        "comment": "XY size (pixels) of the sampled plane/window used for segmentation."
    },
    "volume": {
        "default": None,
        "comment": "Run-level source volume used for tracing."
    },

    # ---------------------
    # Iterations
    # ---------------------
    "cummulative_iterator": {
        "default": 0,
        "comment": "Current iteration count within the active trace."
    },
    "iterations": {
        "default": 20,
        "comment": "Maximum number of tracing steps before the run stops."
    },
    "points_list": {
        "default": [],
        "comment": "Queue of point indices scheduled for processing."
    },
    "pointIndex": {
        "default": None,
        "comment": "Index pointing to the current point in the list."
    },
    "resample_step_size":{
        "default": 5,
        "comment": "Step size used for post-processing curve resampling."
    },

    # ---------------------
    # Data for tracing
    # ---------------------
    "curvenode": {
        "default": None,
        "comment": "Ordered list of traced points."
    },
    "ply_faces": {
        "default": [],
        "comment": "PLY mesh faces (if exported)."
    },
    "ply_vertices": {
        "default": [],
        "comment": "PLY mesh vertices (if exported)."
    },
    "write_ply":{
        "default": False,
        "comment": "If True, write PLY mesh outputs for the trace."
    },

    # ---------------------
    # General
    # ---------------------
    "should_cancel": {
        "default": None,
        "comment": "Callable checked each loop to stop tracing early."
    },
    "start_idx": {
        "default": 0,
        "comment": "Initial index in points_list where tracing begins."
    },
    "stepsize": {
        "default": 15,
        "comment": "Forward propagation step size in pixels."
    },
    "use_GPU": {
        "default": True,
        "comment": "If True, run model inference using GPU when available."
    },
    "cuda_device":{
        "default": "cuda:0",
        "comment": "Torch device string used for model inference (for example cuda:0)."
    },
    "use_recenter_point": {
        "default": True,
        "comment": "If True, recenter the current point to the segmented mask centroid."
    },
    "duration": {
        "default": 0,
        "comment": "Total wall-clock duration of the trace run (seconds)."
    },

    # ------------------------------------------------------
    # Backtrack parameters
    # ------------------------------------------------------
    "bktk_window_size":{
        "default": 10,
        "comment": "Number of recent points used to estimate local direction for backtrack detection."
    },
    "bktk_search_radius":{
        "default": 10,
        "comment": "Search radius (pixels) to find previously visited points near the current point."
    },
    "bktk_dir_thresh":{
        "default":-0.8,
        "comment": "Dot-product threshold for opposite-direction detection in backtrack checks."
    },
    "bktk_min_gap":{
        "default": 5,
        "comment": "Minimum index separation required before a revisit is considered backtracking."
    },

    # ------------------------------------------------------
    # Rotation parameters
    # ------------------------------------------------------
    "use_rotations": {
        "default": True,
        "comment": "If True, test rotated planes to recover or improve segmentation."
    },
    "rot_final_angle": {
        "default": 0,
        "comment": "Best rotation angle selected for the current node."
    },
    "overwite_w_rot": {
        "default": True,
        "comment": "If True, overwrite the last direction vector with the rotation-adjusted vector."
    },
    "rot_angles": {
        "default": [],
        "comment": "Candidate rotation angles evaluated for the current node."
    },
    "rot_df": {
        "default": pd.DataFrame(),
        "comment": "DataFrame of per-angle rotation evaluation metrics."
    },
    "rot_improved_ecc": {
        "default": False,
        "comment": "True when rotation improved eccentricity-based quality metrics."
    },
    "rot_vectors": {
        "default": [],
        "comment": "Rotation-adjusted candidate vectors."
    },
    "rotation_angle":{
        "default": 15,
        "comment": "Maximum absolute rotation angle (degrees) explored around the current direction.",
    },
    "angle_steps":{
        "default": 10,
        "comment": "Number of angle samples evaluated between -rotation_angle and +rotation_angle."
    },

    # ------------------------------------------------------
    # Evaluation parameters
    # ------------------------------------------------------
    "gt_window_size": {
        "default": 5,
        "comment": "Sliding comparison window size used against ground-truth points."
    },
    "ground_truth": {
        "default": "",
        "comment": "Path to ground-truth points JSON."
    },
    "ground_truth_curvenode": {
        "default": None,
        "comment": "Loaded ground-truth curve points for evaluation mode."
    },
    "monotonic_index": {
        "default": 0,
        "comment": "Index for monotonic property checks (if used)."
    },
    "ground_truth_deviation":{
        "default":[],
        "comment": "Per-node deviation records versus ground truth."
    },
    "break_distance":{
        "default": 10,
        "comment": "Maximum allowed ground-truth deviation (pixels) before ending the run."
    },

    # ------------------------------------------------------
    # Current iteration parameters
    # ------------------------------------------------------
    "centroid_ijk": {
        "default": None,
        "comment": "Current segmented centroid in local slice coordinates."
    },
    "current_mask": {
        "default": None,
        "comment": "Current segmentation mask."
    },
    "current_raw": {
        "default": None,
        "comment": "Current sampled image plane or volume passed to segmentation."
    },
    "current_slice_transform": {
        "default": None,
        "comment": "Transform used to map between slice and world coordinates for current node."
    },
    "df_current": {
        "default": None,
        "comment": "Per-node segmentation measurements table."
    },
    "found_mask": {
        "default": False,
        "comment": "True when a valid segmentation mask was found for the current node."
    },
    "loop_time": {
        "default": None,
        "comment": "Per-node processing time (seconds)."
    },
    "visited_functions": {
        "default": [],
        "comment": "Ordered list of processing functions visited for the current node."
    },
    
    # ------------------------------------------------------
    # Troubleshooting parameters
    # ------------------------------------------------------
    "ts_attempts": {
        "default": 0,
        "comment": "Number of troubleshooting attempts for the current node."
    },
    "ts_correlations": {
        "default": None,
        "comment": "Stored troubleshooting correlation metrics for the current attempt."
    },
    "ts_iter_correlations": {
        "default": [],
        "comment": "History of troubleshooting correlation metrics across attempts."
    },
    "ts_max_attempts":{
        "default": 1,
        "comment": "Maximum troubleshooting retries before switching strategy or model."
    },
    "use_ultrack":{
        "default": True,
        "comment": "If True, run Ultrack-based troubleshooting when the first segmentation attempt fails."
    }
}

# --- MODIFY: extend ALL_PARAMETERS with save-related knobs ---
ALL_PARAMETERS.update({
    "save_mode": {"default": "ndjson", "comment": "Output format for per-node details: ndjson, per_node_json, or monolith."},
    "async_io": {"default": True, "comment": "If True, write NDJSON records in a background thread."},
    "flush_every": {"default": 1, "comment": "Flush the NDJSON writer every N records."},
    "queue_maxsize": {"default": 10000, "comment": "Maximum number of pending NDJSON records before producer blocks."},
    "detail_file_name": {"default": "detail_parameters.ndjson", "comment": "Filename for NDJSON detail records."},
    "per_node_dir": {"default": "detail_parameters", "comment": "Directory name used for per-node JSON files."},
    "keep_node_parameter_record_in_memory": {
        "default": False,
        "comment": "If True, keep full per-node detail snapshots in memory."
    }
})



# --- ADD: a tiny async NDJSON writer ---
class NDJSONWriter:
    def __init__(self, path, flush_every=1, queue_maxsize=10000, json_dumps=json.dumps):
        """Initialize the instance state."""
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # line-buffered text mode; system still may buffer but flushes on '\n' often
        self._f = open(self.path, "a", buffering=1, encoding="utf-8")
        self._q = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._flush_every = max(1, int(flush_every))
        self._cnt = 0
        self._json_dumps = json_dumps
        self._t.start()
        atexit.register(self.close)

    def write(self, obj):
        # blocks briefly if queue is full; adjust if you prefer non-blocking + drop
        """Write."""
        self._q.put(obj)

    def _run(self):
        """Run."""
        while not self._stop.is_set() or not self._q.empty():
            try:
                obj = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._f.write(self._json_dumps(obj) + "\n")
            self._cnt += 1
            if self._cnt % self._flush_every == 0:
                self._f.flush()
        self._f.flush()
        self._f.close()

    def close(self):
        """Close."""
        if not self._stop.is_set():
            self._stop.set()
            self._t.join(timeout=5)
            try:
                self._f.flush(); self._f.close()
            except Exception:
                pass



class TracingParameters:
    """
    Encapsulate parameters for run_core_function.

    This class:
      1. Provides defaults (and descriptions) for each parameter.
      2. Allows overriding parameter values via constructor arguments.
      3. Tracks access counts to each attribute.
      4. Can dump its JSON-serializable attributes to a file.
      5. Offers a help method to display parameter definitions.
    """

    def __init__(self, **kwargs):
        """
        Initialize TracingParameters with defaults, overridden by kwargs.

        Args:
            **kwargs: Arbitrary keyword arguments that override default values.
        """
        import logging
        self.log = kwargs.pop("log", None) or logging.Logger(__name__)
        self._access_counts = {}

        # Set defaults, then override with user-provided kwargs
        for param_name, info_dict in ALL_PARAMETERS.items():
            default_val = info_dict["default"]
            # If the user provided a value for param_name, use it; otherwise use the default
            setattr(self, param_name, kwargs.get(param_name, default_val))

        # Any additional kwargs that are not in ALL_PARAMETERS
        # can still be stored if desired:
        for extra_key, extra_val in kwargs.items():
            if extra_key not in ALL_PARAMETERS:
                setattr(self, extra_key, extra_val)
        self.setup_sampler()
        self._json_dumps_fast = None
        try:
            self._json_dumps_fast = lambda o: orjson.dumps(o, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
        except Exception:
            self._json_dumps_fast = json.dumps  # fallback

        
        

    def _ensure_save_dir(self):
        
        """Ensure save dir."""
        folder = str(getattr(self, "next_run_folder", "."))
        os.makedirs(folder, exist_ok=True)
        # Precompute paths
        self._ndjson_path = os.path.join(folder, self.detail_file_name)
        self._pernode_dir = os.path.join(folder, self.per_node_dir)
        if self.save_mode == "per_node_json":
            os.makedirs(self._pernode_dir, exist_ok=True)
        self._writer = None
        if self.save_mode == "ndjson":
            self._start_async_writer()  # starts thread if async_io True; otherwise prepares sync path

    @staticmethod
    def _json_safe(obj):
        """Compute json safe."""
        import numpy as _np
        import pandas as _pd
        if isinstance(obj, ( _np.integer, )):
            return int(obj)
        if isinstance(obj, ( _np.floating, )):
            return float(obj)
        if isinstance(obj, ( _np.bool_, )):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, _pd.Series):
            return obj.to_dict()
        if isinstance(obj, (list, tuple, set)):
            return [TracingParameters._json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {str(TracingParameters._json_safe(k)): TracingParameters._json_safe(v) for k, v in obj.items()}
        return obj

    def _start_async_writer(self):
        """Start async writer."""
        if self.async_io and self.save_mode == "ndjson":
            self._writer = NDJSONWriter(
                self._ndjson_path,
                flush_every=self.flush_every,
                queue_maxsize=self.queue_maxsize,
                json_dumps=self._json_dumps_fast
            )
        else:
            self._writer = None  # sync mode or other save_mode

    def close_writers(self):
        """Close writers."""
        if isinstance(self._writer, NDJSONWriter):
            self._writer.close()
            self._writer = None


    def add(self, key, value):
        """
        Add or update a parameter at runtime.
        """
        setattr(self, key, value)

    def setup_sampler(self):
        """Compute setup sampler."""
        self.reference_image = sitk.Image(self.dim, self.dim, 1, sitk.sitkUInt16)
        self.interpolator = sitk.sitkLinear
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetInterpolator(sitk.sitkLinear)
        self.resampler.SetReferenceImage(self.reference_image)

    def log_attribute_names(self):
        """
        Log all attributes and their types using the .log object.
        """
        if self.log is None:
            return

        self.log.info("Trace parameters:")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                self.log.info("Param: %s | %s", key, type(value))

    def get_definition(self, param_name):
        """
        Return the user-friendly definition (comment) for a given parameter.
        """
        if param_name in ALL_PARAMETERS:
            info = ALL_PARAMETERS[param_name]
            return f"{param_name}: {info['comment']} (default={info['default']})"
        return f"No definition found for '{param_name}'."

    def __getattribute__(self, name):
        """
        Track how many times an attribute is accessed.
        """
        # Bypass normal attribute lookup to increment access count
        value = object.__getattribute__(self, name)
        if not name.startswith('_'):
            access_counts = object.__getattribute__(self, '_access_counts')
            access_counts[name] = access_counts.get(name, 0) + 1
        return value
    
    def record_current_node_params(self):
        """
        Store a per-iteration record and append it to disk without rewriting prior data.
        """
        node_count = len(self.curvenode) if self.curvenode is not None else 0
        current_index = self.pointIndex

        def is_json_serializable(val):
            """Return whether json is serializable."""
            try:
                json.dumps(val)
                return True
            except (TypeError, OverflowError):
                return False

        serializable_attributes = {}
        for key, val in self.__dict__.items():
            if is_json_serializable(val) and not key.startswith('_'):
                if key not in ["latest_diameters", "points_list", "curvenode",
                               "ply_faces", "node_parameter_record",
                               "ply_vertices", "ground_truth_curvenode"]:
                    serializable_attributes[key] = val

        if self.df_current is not None:
            serializable_attributes['df'] = self.df_current.to_dict(orient='list')

        # Build the per-node record
        record = {
            "node_index": int(current_index) if current_index is not None else int(node_count - 1),
            "timestamp": time.time(),
            "data": TracingParameters._json_safe(serializable_attributes),
        }

        # Optional: keep in-memory index (disable to save RAM)
        if self.keep_node_parameter_record_in_memory:
            self.node_parameter_record[str(record["node_index"])] = record["data"]

        # Append to disk according to save_mode
        if self.save_mode == "ndjson":
            if self._writer is not None:
                self._writer.write(record)
            else:
                # Sync append
                os.makedirs(os.path.dirname(self._ndjson_path), exist_ok=True)
                with open(self._ndjson_path, "a", encoding="utf-8") as f:
                    f.write(self._json_dumps_fast(record) + "\n")

        elif self.save_mode == "per_node_json":
            # one small JSON file per node (constant write cost, many files)
            fname = os.path.join(self._pernode_dir, f"record_{record['node_index']:06d}.json")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(self._json_dumps_fast(record))

        else:
            # 'monolith' mode retained for backwards-compat
            # (NOTE: this will still grow slower over time)
            self.node_parameter_record[str(record["node_index"])] = record["data"]
            self.write_full_snapshot()
    
    # def record_current_node_params(self):
    #     """
    #     At every iteration, store the JSON-serializable parameters into
    #     'self.node_parameter_record', keyed by the current node index,
    #     which is len(self.curvenode) - 1.
    #     """
    #     node_count = len(self.curvenode)
    #     current_index = self.pointIndex#node_count - 1 if node_count > 0 else 0
    #     def is_json_serializable(val):
    #         try:
    #             json.dumps(val)
    #             return True
    #         except (TypeError, OverflowError):
    #             return False

    #     # Filter the instance's __dict__ to only include JSON-serializable attributes.
    #     # serializable_attributes = {
    #     #     key: val
    #     #     for key, val in self.__dict__.items()
    #     #     if is_json_serializable(val) and not key.startswith('_')
    #     # }
    #     serializable_attributes = {}
    #     for key, val in self.__dict__.items():
    #         if is_json_serializable(val) and not key.startswith('_'):
    #             if  key not in ["latest_diameters", "points_list", "curvenode", "ply_faces", "node_parameter_record", "ply_vertices", "ground_truth_curvenode"]:
    #                 serializable_attributes[key] = val

    #     if self.df_current is not None:
    #         serializable_attributes['df'] = self.df_current.to_dict(orient='list')
    #     # Store it in node_parameter_record at the current_index.
    #     self.node_parameter_record[current_index] = serializable_attributes
        
    def dump_to_json(self, file_path=None):
        """
        Dump JSON-serializable attributes of this instance to a JSON file.

        If file_path is None, uses self.param_json_path if available.

        Args:
            file_path (str, optional): The path to the JSON file where the
                attributes will be saved.
        """
        def is_json_serializable(val):
            """Return whether json is serializable."""
            try:
                json.dumps(val)
                return True
            except (TypeError, OverflowError):
                return False

        if file_path is None:
            file_path = getattr(self, "param_json_path", "parameters.json")

        # Filter the instance's __dict__ to only include JSON-serializable attributes.
        serializable_attributes = {
            key: val
            for key, val in self.__dict__.items()
            if is_json_serializable(val) and not key.startswith('_')
        }
        with open(file_path, 'w') as f:
            json.dump(serializable_attributes, f, indent=4)

    # def write_detailed_parameters(self):
    #     """
    #     Writes 'self.node_parameter_record' to a file called
    #     'detail_parameters.json' in the 'next_run_folder'.
    #     """

    #     # def convert_np_types(obj):
    #     #     if isinstance(obj, dict):
    #     #         return {k: convert_np_types(v) for k, v in obj.items()}
    #     #     elif isinstance(obj, list):
    #     #         return [convert_np_types(i) for i in obj]
    #     #     elif isinstance(obj, (np.integer, np.int64, np.int32)):
    #     #         return int(obj)
    #     #     elif isinstance(obj, (np.floating, np.float64, np.float32)):
    #     #         return float(obj)
    #     #     elif isinstance(obj, np.bool_):
    #     #         return bool(obj)
    #     #     else:
    #     #         return obj
    #     def _json_safe(obj):
    #         # Values
    #         if isinstance(obj, (np.integer,)):
    #             return int(obj)
    #         if isinstance(obj, (np.floating,)):
    #             return float(obj)
    #         if isinstance(obj, (np.bool_,)):
    #             return bool(obj)
    #         if isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         if isinstance(obj, pd.DataFrame):
    #             # Avoid dict-of-dicts with int64 index keys
    #             return obj.to_dict(orient='records')
    #         if isinstance(obj, pd.Series):
    #             return obj.to_dict()
    #         if isinstance(obj, (list, tuple, set)):
    #             return [_json_safe(x) for x in obj]
    #         if isinstance(obj, dict):
    #             # Convert KEYS to str (or int(obj) if you prefer), and recurse on values
    #             return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    #         return obj  

    #     safe_data = _json_safe(self.node_parameter_record)

    #     file_name = "detail_parameters.json"
    #     # If next_run_folder is not set or not a valid path, default to current dir.
    #     folder = str(getattr(self, "next_run_folder", "."))
    #     file_path = os.path.join(folder, file_name)
    #     with open(file_path, 'w') as f:
    #         json.dump(safe_data, f, indent=4)
    
    def reset_iteration(self):
        """Reset iteration."""
        self.df_current =  None
        self.centroid_ijk =  None
        self.ts_correlations =  None
        self.found_mask = False
        self.current_mask =  None
        self.current_raw =  None
        self.current_slice_transform =  None
        self.loop_time = None
        self.visited_functions = []

        self.rot_improved_ecc = False
        self.rot_vectors = []
        self.rot_angles = []
        self.rot_df = pd.DataFrame()
        self.rot_final_angle = 0

    def reset_trouble_shooting(self):
        """Reset trouble shooting."""
        self.ts_attempts = 0
        self.ts_iter_correlations = []

    def write_full_snapshot(self):
        """
        Writes the entire node_parameter_record to a single JSON file.
        Use sparingly (e.g., at the end) because it's O(N).
        """
        folder = str(getattr(self, "next_run_folder", "."))
        file_path = os.path.join(folder, "detail_parameters_snapshot.json")
        safe_data = TracingParameters._json_safe(self.node_parameter_record)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._json_dumps_fast(safe_data))
    
    def write_detailed_parameters(self):
        """
        Backwards-compatible alias: in NDJSON mode this is a no-op because we stream per-node.
        In monolith mode, writes a full snapshot (expensive).
        """
        if self.save_mode == "monolith":
            self.write_full_snapshot()
        # else: streaming already handled per node; nothing to do here.
    # def shallow_copy(self):
    #     trace_bk = TracingParameters()
    #     trace.current_slice_transform = trace_bk.current_slice_transform
    #     trace.current_raw = trace_bk.current_raw
    #     trace.current_mask = trace_bk.current_mask
    #     trace.df_current = trace_bk.df_current
    #     trace.found_mask = trace_bk.found_mask
    #     trace.centroid_ijk = trace_bk.centroid_ijk
    #     trace.vectors[-1] = trace_bk.vectors[-1]
    #     return trace_bk

    # --- Lightweight rotation state backup/restore -----------------------------

    def make_rotation_backup(self):
        """
        Snapshot only what we actually revert on failure.
        Avoid copying SWIG/GUI/logging objects.
        """
        # deep-ish copies only for pure-Python data (arrays/frames)
        df = self.df_current.copy(deep=True) if getattr(self, "df_current", None) is not None else None

        # numpy arrays / lists are fine to deepcopy; SWIG objects are not.
        return {
            "current_slice_transform": self.current_slice_transform,  # keep by reference (usually replaced, not mutated)
            "current_raw": copy.deepcopy(self.current_raw),
            "current_mask": copy.deepcopy(self.current_mask),
            "df_current": df.copy(),
            "found_mask": bool(self.found_mask),
            "centroid_ijk": copy.deepcopy(self.centroid_ijk),
            "last_vector": copy.deepcopy(self.vectors[-1]) if getattr(self, "vectors", []) else None,
        }

    def restore_rotation_backup(self, bk):
        """Restore rotation backup."""
        self.current_slice_transform = bk["current_slice_transform"]
        self.current_raw   = bk["current_raw"]
        self.current_mask  = bk["current_mask"]
        self.df_current    = bk["df_current"]
        self.found_mask    = bk["found_mask"]
        self.centroid_ijk  = bk["centroid_ijk"]
        if bk["last_vector"] is not None and getattr(self, "vectors", None):
            self.vectors[-1] = bk["last_vector"]
