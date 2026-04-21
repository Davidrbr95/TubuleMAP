import time
from tubulemap.cellpose_tracker.io_utils import *
from tubulemap.cellpose_tracker.vector_ops import *
from tubulemap.cellpose_tracker.segmentation import *
from cellpose import models as Models
import numpy as np
from scipy.interpolate import interp1d
import h5py
from concurrent.futures import ThreadPoolExecutor

def load_chunk_for_points(trace, point_idx):
    """Load chunk for points."""
    points = get_point_curve_ras(trace.pointIndex, trace.curvenode).ravel()
    points_int = points.astype('int')
    lw_bnds, up_bnds = check_chunk_size(trace.volume, trace.chunk_size, points_int)
    im=trace.volume[lw_bnds[2]:up_bnds[2], lw_bnds[1]:up_bnds[1], lw_bnds[0]:up_bnds[0]].astype('uint16')
    image = sitk.GetImageFromArray(im)
    image.SetOrigin([float(i) for i in lw_bnds])
    return image

def record_call(func):
    """Decorator to record a function call into trace.visited_functions."""
    def wrapper(trace, *args, **kwargs):
        # Ensure trace has a visited_functions attribute; initialize if needed.
        """Wrap the target function and preserve pipeline behavior."""
        trace.visited_functions.append(func.__name__)
        return func(trace, *args, **kwargs)
    return wrapper


@record_call
def ultrack_trouble_shooting_diameter(trace):
    """Compute ultrack trouble shooting diameter."""
    trouble_shooting_diameter_only(trace)
    if trace.found_mask:
        analyze_segmenation(trace)   
    else:
        trace.log.info("Diameter only approach failed")

@record_call
def ultrack_trouble_shooting_full(trace):
    """Compute ultrack trouble shooting full."""
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
def first_attempt(trace, vector):
    """Compute first attempt."""
    set_slice_view(trace, vector=vector)
    get_frame(trace)
    run_cellpose(trace)
    analyze_segmenation(trace)
    return

def compute_vectors(trace):
    """Compute vectors."""
    trace.log.info('Initializing the trace: finding the direction vector of the last 2 points')
    if trace.pointIndex == 0:
        Vt, c_pt = direction_vector(trace, trace.pointIndex)
        V_last = Vt # save for future dirrection calcualtion
        trace.vectors.append(V_last)
    else:
        Vt, c_pt = direction_vector(trace, trace.pointIndex)
        #linear comb of the last dirr vector and the new dirr vector
        V = trace.w*Vt+(1-trace.w)*trace.vectors[-1] 
        V_last = V # save for future dirrection calcualtion
        trace.vectors.append(V_last)


def save_images_to_hdf5(trace, hdf5_path="trace_images.h5"):
    """
    Saves trace.current_mask and trace.current_raw images into an HDF5 file 
    for each iteration (idx) if they are not None.

    Parameters:
    - trace: object containing `current_mask` and `current_raw` images.
    - hdf5_path: str, path to the HDF5 file where images are stored.
    """

    # Ensure the directory exists
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    with h5py.File(hdf5_path, "a") as h5f:  # 'a' mode appends data if file exists
        idx = trace.pointIndex  # Current iteration index

        if trace.current_mask is not None:
            dataset_name_mask = f"mask_{idx}"
            if dataset_name_mask in h5f:
                del h5f[dataset_name_mask]  # Remove existing dataset if overwriting
            h5f.create_dataset(dataset_name_mask, data=trace.current_mask, dtype='uint16')

        if trace.current_raw is not None:
            dataset_name_raw = f"raw_{idx}"
            if dataset_name_raw in h5f:
                del h5f[dataset_name_raw]
            h5f.create_dataset(dataset_name_raw, data=trace.current_raw, dtype='uint16')

        trace.log.info(f"Saved images for iteration {idx} to {hdf5_path}")

@record_call
def resample_points(trace, step_size=5.0):
    """
    Resamples trace.points_list so that consecutive points are spaced
    approximately 'step_size' units apart.
    
    Parameters
    ----------
    trace : object
        Your trace object that has 'points_list', which should be a sequence
        of coordinates (Nx2 or Nx3). If it's Nx2 or Nx3, this function
        will treat it as (x, y) or (x, y, z) positions. 
        If your data is 3D but stored differently, adapt accordingly.
    step_size : float
        The desired spacing between resampled points.
    """
    old_points = np.array(trace.curvenode)

    # If there's fewer than 2 points, there's nothing to resample
    if len(old_points) < 2:
        trace.log.info("Not enough points to resample; skipping.")
        return

    diffs = old_points[1:] - old_points[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)

    cumulative_len = np.insert(np.cumsum(seg_lengths), 0, 0)  
    total_length = cumulative_len[-1]

    new_distances = np.arange(0, total_length, step_size)
    if total_length not in new_distances:
        new_distances = np.append(new_distances, total_length)

    new_points = []
    seg_index = 0  # which segment of old_points we are in
    for d in new_distances:
        while seg_index < len(seg_lengths) - 1 and d > cumulative_len[seg_index + 1]:
            seg_index += 1
        segment_start_dist = cumulative_len[seg_index]
        segment_end_dist   = cumulative_len[seg_index + 1]
        segment_fraction   = (d - segment_start_dist) / (segment_end_dist - segment_start_dist + 1e-12)
        p1 = old_points[seg_index]
        p2 = old_points[seg_index + 1]
        new_p = p1 + segment_fraction * (p2 - p1)
        new_points.append(new_p)
    new_points = np.array(new_points)
    trace.curvenode = new_points.tolist()
    trace.log.info(
        f"Resampled {len(old_points)} points to {len(new_points)} points at step size {step_size}."
    )


# def looping_through_points(trace):
#     resample_points(trace, step_size=trace.resample_step_size)      

#     trace.points_list = []
#     for i in range(len(trace.curvenode)):
#         trace.points_list.append(i)
#     trace.iterations = len(trace.curvenode)
#     trace.log.info('Starting tracing loop')
#     for idx in range(len(trace.points_list)):
#         start_loop_time = time.time()
#         trace.reset_iteration()

#         # Setup loop
#         trace.pointIndex = trace.points_list[idx]

#         trace.log.info('[Calculating vector] Tracing PointIndex: {%0.1f}', trace.pointIndex)
        
#         write_status(
#             trace,
#             status="running",
#             error_msg="Tracking",
#         )

#         if not trace.multiprocessing:
#             yield None
        
#         compute_vectors(trace)
    
#     success = 0 
#     for idx in range(len(trace.points_list)):
#         start_loop_time = time.time()
#         trace.reset_iteration()

#         # Setup loop
#         trace.pointIndex = trace.points_list[idx]

#         if trace.current_chunk is None:
#             trace.log.info('Dynamic loading of new image chunk from zarr')
#             trace.current_chunk = load_image(trace)

#         trace.log.info('[Actual ortho generator] Tracing PointIndex: {%0.1f}', trace.pointIndex)
        
#         write_status(
#             trace,
#             status="running",
#             error_msg="Tracking",
#         )

#         if not trace.multiprocessing:
#             yield None

#         first_attempt(trace, trace.vectors[idx])

#         if not trace.found_mask:
#             trace.log.info('No mask found in first_attempt: starting ultrack troubleshooting')
#             ultrack_trouble_shooting(trace)
        

#         try:
#             if trace.use_adaptive_diameter:
#                 diameter = trace.df_current.iloc[-1]['equivalent_diameter_area']
#                 r_v = adapt_radius(trace, diameter)
#                 trace.log.info(f'Adaptive diameter set to: {diameter}, {r_v}')
#                 trace.diameter = r_v
#         except:
#             pass
#         trace.current_chunk = load_image(trace)
#         trace.cummulative_iterator+=1
#         print('CHECK CUMMULATIVE ITERATOR', trace.cummulative_iterator)

#         trace.loop_time = time.time() - start_loop_time 
#         # TODO: Add a writing step where everything that needs to be written per iteration is written (all parameters would be good)
        
#         if trace.pointIndex == 0:
#             save_curve_nodes(trace, reset = True)
        
#         # print(trace.current_mask.shape,
#         #       trace.current_raw.shape)
        
        
#         if not trace.found_mask:
#             trace.log.info("No mask found in trouble shooting: breaking points")
#             continue
#         else:
#             success += 1
#             print('Sucess', success)

#         if trace.current_mask is not None and trace.current_raw is not None:
#             save_images_to_hdf5(trace, os.path.join(trace.next_run_folder, 'ortho_planes.hdf5'))
#         add_mask_edge_loop(trace)
#     print('Done with loop now saving')
#     save_curve_nodes(trace, reset = True)
            
#     write_status(
#         trace,
#         status="done",
#         error_msg="Tracking",
#     )

def find_extrema(trace):
    """Find extrema."""
    xs, ys, zs = zip(*trace.curvenode)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    lwbnds = np.array([xmin, ymin, zmin])
    upbnds = np.array([xmax, ymax, zmax])
    return lwbnds, upbnds
    
def looping_through_points(trace):
    """Compute looping through points."""
    resample_points(trace, step_size=trace.resample_step_size)      

    trace.log.info('Starting tracing loop')
    
    trace.points_list = []
    for i in range(len(trace.curvenode)):
        trace.points_list.append(i)
    
    trace.iterations = len(trace.curvenode)
    trace.log.info('Starting tracing loop. LENTH OF POINTS'+str(len(trace.curvenode)))

    for idx in range(len(trace.points_list)):
        start_loop_time = time.time()
        trace.reset_iteration()

        # Setup loop
        trace.pointIndex = trace.points_list[idx]

        trace.log.info('[Calculating vector] Tracing PointIndex: {%0.1f}', trace.pointIndex)
        
        write_status(
            trace,
            status="running",
            error_msg="Tracking",
        )

        if not trace.multiprocessing:
            yield None

        compute_vectors(trace)
    print(trace.points_list)
    
    success = 0
    trace.log.info('Main run')
    successful_nodes = []
    num_pts = len(trace.points_list)
    def submit_if_valid(j):
        """Compute submit if valid."""
        return pool.submit(load_image, trace, j) if (j is not None and 0 <= j < num_pts) else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        cur_idx = 0
        nxt1_idx = cur_idx + trace.save_rate
        nxt2_idx = nxt1_idx + trace.save_rate
        nxt3_idx = nxt2_idx + trace.save_rate

        current_img = load_image(trace, 0)
        future_1 = submit_if_valid(nxt1_idx) #(pool.submit(load_image, trace, nxt1_idx) if len(trace.points_list)>1 else None)
        future_2 = submit_if_valid(nxt2_idx)#(pool.submit(load_image, trace, nxt2_idx) if len(trace.points_list)>1 else None)
        future_3 = submit_if_valid(nxt3_idx) #(pool.submit(load_image, trace, nxt3_idx) if len(trace.points_list)>1 else None)
        for idx in range(len(trace.points_list)):
            start_loop_time = time.time()
            trace.reset_iteration()
            trace.current_chunk = current_img

            trace.pointIndex = trace.points_list[idx]

            if trace.current_chunk is None:
                trace.log.info('Dynamic loading of new image chunk from zarr')
                # trace.current_chunk = load_image(trace)
                # trace.current_chunk = load_image_gt(trace, lwbnds, upbnds)
            
            trace.log.info('[Actual ortho generator] Tracing PointIndex: {%0.1f}', trace.pointIndex)

            write_status(
                trace,
                status="running",
                error_msg="Tracking",
            )
            
            if not trace.multiprocessing:
                yield None

            first_attempt(trace, trace.vectors[idx])

            if not trace.found_mask:
                trace.log.info('No mask found in first_attempt: starting ultrack troubleshooting')
                if trace.use_ultrack:
                    ultrack_trouble_shooting_diameter(trace)
                    if not trace.found_mask:
                        ultrack_trouble_shooting_full(trace)
            else:
                trace.log.info('No troubleshooting necessary: reseting the troubleshooting parameters')
                trace.reset_trouble_shooting()
            
            if trace.use_adaptive_diameter and trace.df_current is not None:
                diameter = trace.df_current.iloc[-1]['equivalent_diameter_area']
                trace.log.info(f'In adaptive diameter: {diameter}')
                trace.latest_diameters.append(diameter)
                if len(trace.latest_diameters)<= trace.adapt_window:
                    new_diameter = np.mean(trace.latest_diameters)
                else:
                    diam_list = len(trace.latest_diameters)
                    trace.log.info(f"Diameters taken into account. Size of list {diam_list}:{trace.latest_diameters[-trace.adapt_window:]}")
                    new_diameter = np.mean(trace.latest_diameters[-trace.adapt_window:])
                r_v = adapt_radius(trace, new_diameter)
                trace.jitter = int(r_v/trace.scale_jitter)
                trace.stepsize = int(r_v/trace.scale_stepsize)
                trace.log.info(f'Adaptive diameter set to: {new_diameter}, {r_v}')
                trace.diameter = r_v

            trace.cummulative_iterator+=1

            if trace.cummulative_iterator%trace.save_rate == 0:
                if future_1 is not None:
                    print('Data is ready for next')
                    current_img = future_1.result()
                else:
                    current_img = None
                
                future_1 = future_2
                future_2 = future_3

                nxt3_idx += trace.save_rate
                if idx + trace.save_rate < len(trace.curvenode):
                    future_3 = submit_if_valid(nxt3_idx)#pool.submit(load_image, trace, nxt3_idx)
                else:
                    future_3 = None
                # trace.current_chunk = load_image(trace)
            #     trace.current_chunk = load_image_gt(trace, lwbnds, upbnds)


            

            print('CHECK CUMMULATIVE ITERATOR', trace.cummulative_iterator)

            trace.loop_time = time.time() - start_loop_time 

            if not trace.found_mask:
                trace.log.info("No mask found in trouble shooting: breaking points")
                continue
            else:
                success += 1.0
            
            perc_succ = success/(trace.cummulative_iterator)*100.0

            trace.log.info(f"Percentage of masks found {perc_succ}%")
            type_m = type(trace.current_mask)
            trace.log.info(f"Mask type: {type_m}")
            if trace.current_mask is not None and trace.current_raw is not None:
                save_images_to_hdf5(trace, os.path.join(trace.next_run_folder, 'ortho_planes.hdf5'))
            trace.record_current_node_params()
            add_mask_edge_loop(trace)
            save_curve_nodes(trace, reset = True)
            successful_nodes.append(trace.curvenode[trace.pointIndex])
    print('Done with loop now saving')
    trace.curvenode = successful_nodes
    trace.write_ply = True
    save_curve_nodes(trace, reset = True)
    write_status(
        trace,
        status="done",
        error_msg="Tracking",
    )
