from pathlib import Path
from tubulemap.cellpose_tracker.io_utils import *
from tubulemap.cellpose_tracker.evaluation import *
from tubulemap.cellpose_tracker.vector_ops import *
from tubulemap.cellpose_tracker.geometry import *
from tubulemap.cellpose_tracker.plane_rotations import *
from tubulemap.cellpose_tracker.backtrack import *
from tubulemap.cellpose_tracker.core_post_processing import save_images_to_hdf5

_SEGMENTATION_IMPORTS_READY = False


def _ensure_segmentation_imports():
    """Ensure segmentation imports."""
    global _SEGMENTATION_IMPORTS_READY
    global analyze_segmenation, run_cellpose, trouble_shooting, trouble_shooting_diameter_only

    if _SEGMENTATION_IMPORTS_READY:
        return

    from tubulemap.cellpose_tracker.segmentation import (
        analyze_segmenation as _analyze_segmenation,
        run_cellpose as _run_cellpose,
        trouble_shooting as _trouble_shooting,
        trouble_shooting_diameter_only as _trouble_shooting_diameter_only,
    )

    analyze_segmenation = _analyze_segmenation
    run_cellpose = _run_cellpose
    trouble_shooting = _trouble_shooting
    trouble_shooting_diameter_only = _trouble_shooting_diameter_only
    _SEGMENTATION_IMPORTS_READY = True

def record_call(func):
    """Decorator to record a function call into trace.visited_functions."""
    def wrapper(trace, *args, **kwargs):
        # Ensure trace has a visited_functions attribute; initialize if needed.
        """Wrap the target function and preserve pipeline behavior."""
        trace.visited_functions.append(func.__name__)
        return func(trace, *args, **kwargs)
    return wrapper

def check_user_cancellation(trace):
    """
    Check if the user has requested cancellation of the tracking process.

    This function checks the cancellation flag in the trace object. If cancellation is requested,
    it performs the following steps:
      - Logs that cancellation has been initiated.
      - If ground truth keypoints are available, calls master_compare for evaluation.
      - Saves the current curve nodes with the reset flag.
      - Logs that tracking has been stopped by the user.
      - Writes a final status to disk indicating that tracking was halted.

    Args:
        trace: An object containing tracking parameters and state. It is expected to have the
            following attributes:
            - should_cancel (callable): A function that returns True if cancellation is requested.
            - log (logging.Logger): Logger used for status messages.
            - ground_truth_curvenode: Optional ground truth data for comparison.
            - status_file_path (str): File path where the status JSON should be written.

    Returns:
        bool: True if cancellation was requested and processed, otherwise False.
    """
    if trace.should_cancel is not None and trace.should_cancel():
        trace.log.info('Canceling the trace')
        if trace.ground_truth_curvenode:
            master_compare(trace)
        save_curve_nodes(trace, reset = True)
        trace.log.info('Tracking stopped by user.')
        write_status(
            status_path=trace.status_file_path,
            status="done",
            error_msg="Tracking stopped by user",
        )
        return True
    else:
        return False

def intialize_trace(trace):
    """
    Initialize the trace by calculating the direction vector and updating state.

    This function computes the direction vector between the last two points in the trace
    and updates the trace object accordingly. When the pointIndex equals trace.start_idx - 1,
    it calculates the initial direction vector and appends it to the list of vectors.
    When the pointIndex equals trace.start_idx, it computes a linear combination of the new
    direction vector and the previously saved direction vector (for smoothing), calculates
    the new point based on this vector, and updates the curve and tracking lists.

    Args:
        trace: An object containing the trace state and parameters. Expected to have attributes:
            - log: A logger for logging messages.
            - start_idx: The starting index for the trace.
            - vectors: A list to store computed direction vectors.
            - w: A weight used for linear combination of direction vectors.
            - stepsize: The step size for advancing the trace.
            - curvenode: A list of points representing the trace.
            - points_list: A list to store indices of trace points.
        pointIndex (int): The index of the current point being processed.

    Returns:
        None. The function updates the trace object in place.
    """
    trace.log.info('Initializing the trace: finding the direction vector of the last 2 points')
    if trace.pointIndex == trace.start_idx-1:
        Vt, c_pt = direction_vector(trace, trace.pointIndex)
        V_last = Vt # save for future dirrection calcualtion
        trace.vectors.append(V_last)
    if trace.pointIndex == trace.start_idx:
        Vt, c_pt = direction_vector(trace, trace.pointIndex)
        #linear comb of the last dirr vector and the new dirr vector
        V = trace.w*Vt+(1-trace.w)*trace.vectors[-1] 
        V_last = V # save for future dirrection calcualtion
        n_pt = c_pt + trace.stepsize*V  # set new point in space based on a vector direction
        trace.curvenode.append(tuple(n_pt.ravel()))
        trace.points_list.append(trace.pointIndex+1)
        trace.vectors.append(V_last)
        trace.latest_diameters.append(trace.diameter)

@record_call
def first_attempt(trace):
    """Compute first attempt."""
    _ensure_segmentation_imports()
    V_last, _ = direction_vector(trace, trace.pointIndex)
    trace.vectors.append(V_last)
    set_slice_view(trace)
    get_frame(trace)
    run_cellpose(trace)
    analyze_segmenation(trace)
    return

@record_call
def ultrack_trouble_shooting_diameter(trace):
    """Compute ultrack trouble shooting diameter."""
    _ensure_segmentation_imports()
    trouble_shooting_diameter_only(trace)
    if trace.found_mask:
        analyze_segmenation(trace)   
    else:
        trace.log.info("Diameter only approach failed")

@record_call
def ultrack_trouble_shooting_full(trace):
    """Compute ultrack trouble shooting full."""
    _ensure_segmentation_imports()
    trouble_shooting(trace)
    
    if trace.found_mask:
        analyze_segmenation(trace)

        # Store the correlations
        trace.ts_iter_correlations.append(trace.ts_correlations)

        # Keep only the last 3 entries
        if len(trace.ts_iter_correlations) > 3:
            trace.ts_iter_correlations.pop(0)

        # Increment troubleshooting attempts
        trace.ts_attempts += 1
        print('Troubleshooting attemps', trace.ts_attempts)

        # Check if we have reached 3 troubleshooting attempts
        if trace.ts_attempts >= trace.ts_max_attempts:
            # Aggregate IoUs for each model over the last 3 attempts
            model_iou_dict = {}
            for corr_list in trace.ts_iter_correlations:
                for corr in corr_list:
                    model_name = corr['model']
                    iou = corr['iou']
                    if model_name in model_iou_dict:
                        model_iou_dict[model_name].append(iou)
                    else:
                        model_iou_dict[model_name] = [iou]
            
            # Calculate average IoU for each model
            avg_iou_per_model = {model_name: np.mean(iou_list) for model_name, iou_list in model_iou_dict.items()}
            print('avg_iou_per_model')
            print(avg_iou_per_model)
            # Select the model with the highest average IoU
            if avg_iou_per_model:
                from cellpose import models as Models
                import torch

                new_primary_model_name = max(avg_iou_per_model, key=avg_iou_per_model.get)
                print(f"Switching primary model to: {new_primary_model_name}")
                trace.log.info(f"Switching primary model to: {new_primary_model_name}")

                # Reinitialize the primary model
                use_gpu = bool(getattr(trace, "use_GPU", False))
                device_name = str(getattr(trace, "cuda_device", "cpu") or "cpu")
                if use_gpu and not torch.cuda.is_available():
                    trace.log.warning(
                        "Requested GPU model reload but CUDA is unavailable. Falling back to CPU."
                    )
                    use_gpu = False
                    device_name = "cpu"
                model = Models.CellposeModel(
                    gpu=use_gpu,
                    pretrained_model=os.path.join(trace.model_suite, new_primary_model_name),
                    device=torch.device(device_name),
                )
                trace.model = model ## should this be the model name?
                trace.model_name = new_primary_model_name
                trace.use_GPU = use_gpu
                trace.cuda_device = device_name

                
                # # Reset troubleshooting attempts and correlations
                # trace.ts_attempts = 0
                # trace.ts_iter_correlations = []
            else:
                trace.log.info("No valid models found in troubleshooting correlations.")

@record_call
def adapt_radius(trace, diam):
    """Compute adapt radius."""
    if diam<trace.adapt_diam_lower:
        diam = trace.adapt_diam_lower
    elif diam>trace.adapt_diam_upper:
        diam = trace.adapt_diam_upper
    return diam

@record_call
def apply_rotations(trace):
    """Apply rotations."""
    rotate_to_improve_ecc(trace)

@record_call
def apply_recentering(trace):
    """Apply recentering."""
    point_ijk = [trace.centroid_ijk[1], trace.centroid_ijk[0],  0]
    ras_points = trace.current_slice_transform.TransformPoint(point_ijk)
    
    # Compute the vector from the previous point to the detected centroid
    prev_point = np.array(trace.curvenode[trace.pointIndex - 1])
    ras_point = np.array(ras_points)
    vector = ras_point - prev_point
    distance = np.linalg.norm(vector)
    
    if distance > trace.stepsize*2:
        # Limit the movement to stepsize in the direction towards ras_point
        unit_vector = vector / distance
        new_point = prev_point + trace.stepsize * unit_vector
        trace.curvenode[trace.pointIndex] = new_point.tolist()
    else:
        # Use the ras_points as is
        trace.curvenode[trace.pointIndex] = ras_points

    if not trace.rot_improved_ecc: # Use rotation vector if rotation help otherwise use the vector calc by point locations
        trace.vectors[-1], _ = direction_vector(trace, trace.pointIndex) # finds vector from previous to current point (recentered current point)

def looping_through_points(trace):
    
    """Compute looping through points."""
    trace.log.info('Starting tracing loop')
    
    while len(trace.points_list)>0:
        start_loop_time = time.time()
        trace.reset_iteration()
        print('.')
        if check_user_cancellation(trace):
            break

        # Setup loop
        trace.pointIndex = trace.points_list.pop(0)

        trace.log.info('Tracing PointIndex: {%0.1f}', trace.pointIndex)
        
        write_status(
            trace,
            status="running",
            error_msg="Tracking",
        )

        if not trace.multiprocessing:
            yield None
        
        if trace.pointIndex <= trace.start_idx:
            trace.log.info('Initializing tracing...' + str(trace.pointIndex))
            intialize_trace(trace)
            continue
        
        if trace.current_chunk is None:
            trace.log.info('Dynamic loading of new image chunk from zarr')
            trace.current_chunk = load_image(trace)

        #pure ultrack approach remove the first attmept and go straight to troubleshooting
        first_attempt(trace)

        if not trace.found_mask:
            trace.log.info('No mask found in first_attempt: starting ultrack troubleshooting')
            if trace.use_ultrack:
                ultrack_trouble_shooting_diameter(trace)
                if not trace.found_mask:
                    ultrack_trouble_shooting_full(trace)
        else:
            trace.log.info('No troubleshooting necessary: reseting the troubleshooting parameters')
            trace.reset_trouble_shooting()
        
        if not trace.found_mask:
            trace.log.info("No mask found in trouble shooting: breaking points")
            if trace.ground_truth_curvenode:
                master_compare(trace, error="No mask found")
                write_status(
                    trace,
                    status="rerun",
                    error_msg="No mask was found",
                )
                break
            save_curve_nodes(trace, reset = True)
            write_status(
                trace,
                status="done",
                error_msg="No mask was found",
            )
            break
        
        ## TODO: ADD CRITERIA THAT CHECKS IF ECC is bellow the threshold or if you dont want to do rotations that it avoid them
        if trace.use_rotations:
            # if trace.df_current.iloc[-1]['eccentricity'] >= trace.ecc_threshold:
            apply_rotations(trace)

        if trace.use_adaptive_diameter:
            diameter = trace.df_current.iloc[-1]['equivalent_diameter_area']
            trace.latest_diameters.append(diameter)
            if len(trace.latest_diameters)<= trace.adapt_window:
                new_diameter = np.mean(trace.latest_diameters)
            else:
                new_diameter = np.mean(trace.latest_diameters[-trace.adapt_window:])
            r_v = adapt_radius(trace, new_diameter)
            trace.jitter = int(r_v/trace.scale_jitter)
            # trace.stepsize = int(r_v/trace.scale_stepsize)
            trace.log.info(f'Adaptive diameter set to: {new_diameter}, {r_v}')
            trace.diameter = r_v

        if trace.use_recenter_point: # overwrites the current point in the curvenode 
            apply_recentering(trace)

        if check_backtrack(trace):
            if trace.ground_truth_curvenode:
                master_compare(trace, error="Backtracking")
                write_status(
                    trace,
                    status="rerun",
                    error_msg="Backtracking",
                )
                break
            save_curve_nodes(trace, reset = True)
            write_status(
                    trace,
                    status="done",
                    error_msg="Backtracking identified",
                )
            break

        n_pt, _ = new_vector(trace)
        
        if trace.cummulative_iterator == trace.iterations and trace.ground_truth == "":
            save_curve_nodes(trace, reset = True)
            write_status(
                    trace,
                    status="done",
                    error_msg="All good, finshed all iterations",
                )
            break

        if trace.cummulative_iterator%trace.save_rate == 0:
            save_curve_nodes(trace, reset = True)
            # TODO: Maybe this should be written into the save_curve_nodes
            # write_ply(os.path.join(trace.save_dir, trace.name[:-5]+'_cloud.ply'), mask_point_cloud, faces_cross)
            write_status(
                    trace,
                    status="running",
                    error_msg="All good, just saved checkpoint",
                )
 
            with open(trace.result_trace_path+'.json', 'r') as f:
                points_data = json.load(f)
                
            points = np.array(points_data['points'])

            trace_path = Path(trace.result_trace_path)
            points_name = f"{trace_path.parent.name}_{trace_path.name}"

            data_to_yield = {
                'points': points,
                'points_name': points_name,
            }

            if not trace.multiprocessing:
                yield data_to_yield

            trace.current_chunk = load_image(trace)


        trace.curvenode.append(tuple(n_pt.ravel()))

        trace.points_list.append(trace.pointIndex+1)
        trace.cummulative_iterator+=1

        # print('CHECK CUMMULATIVE ITERATOR', trace.cummulative_iterator)
        # if trace.cummulative_iterator == 217:
        #     exit()
        if trace.ground_truth_curvenode:
            distance_break, end_of_track_break  = master_compare(trace)
            if distance_break:
                trace.log.info('Breaking due to deviation from ground truth')
                write_status(
                trace,
                status="rerun",
                error_msg="Breaking due to deviation from ground truth",
                )
                break
            elif end_of_track_break:
                trace.log.info('Breaking due to end of ground truth')
                write_status(
                trace,
                status="all_complete",
                error_msg="Breaking due to end of ground truth",
                )
                break
            write_status(
                trace,
                status="running",
                error_msg="All good, just saved checkpoint",
            )
        trace.loop_time = time.time() - start_loop_time 
        trace.record_current_node_params()
        if trace.current_mask is not None and trace.current_raw is not None:
            save_images_to_hdf5(
                trace,
                os.path.join(trace.next_run_folder, "ortho_planes.hdf5"),
            )
        # if trace.pointIndex == 10:
        #     break
        # TODO: Add a writing step where everything that needs to be written per iteration is written (all parameters would be good)
        # make sure to inclue a timer
        add_mask_edge_loop(trace)
    trace.close_writers()
