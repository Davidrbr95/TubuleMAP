import os
import zarr
import pandas as pd
import numpy as np
import skimage.measure as measure
import skimage.segmentation as skseg
import scipy.ndimage as scipy_ndi
from ultrack.utils.array import array_apply, create_zarr
from tubulemap.cellpose_tracker.geometry import *
from ultrack.utils.cuda import import_module, to_cpu
from ultrack.imgproc.segmentation import reconstruction_by_dilation, Cellpose
from ultrack.imgproc import normalize
from numpy.typing import ArrayLike
from ultrack.utils import labels_to_contours
from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.config import MainConfig
import shutil
from pathlib import Path

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def _labels_to_contours_safe(labels, sigma, trace):
    """
    Run ``labels_to_contours`` with a CPU fallback for CuPy/skimage incompatibility.

    Some ultrack + cupy combinations route label frames to scikit-image morphology,
    which expects NumPy arrays and can fail with:
    "Implicit conversion to a NumPy array is not allowed..."
    """
    def _run_labels_to_contours(input_labels):
        # Use fresh stores per call attempt to avoid zarr path collisions on retry.
        detection_store = zarr.TempStore()
        edges_store = zarr.TempStore()
        return labels_to_contours(
            input_labels,
            sigma=sigma,
            detection_store_or_path=detection_store,
            edges_store_or_path=edges_store,
            overwrite=True,
        )

    def _labels_to_contours_cpu(input_labels):
        """Compute contour maps fully on CPU as a compatibility fallback."""
        if not input_labels:
            raise ValueError("labels_to_contours fallback received an empty labels list.")

        shape = tuple(np.asarray(input_labels[0]).shape)
        for lb in input_labels[1:]:
            if tuple(np.asarray(lb).shape) != shape:
                raise ValueError("All labels must have the same shape for contour fallback.")

        detection = create_zarr(shape, bool, zarr.TempStore(), overwrite=True)
        contours = create_zarr(shape, np.float32, zarr.TempStore(), overwrite=True)

        n_labels = float(len(input_labels))
        for t in range(shape[0]):
            foreground_frame = np.zeros(shape[1:], dtype=bool)
            contours_frame = np.zeros(shape[1:], dtype=np.float32)

            for lb in input_labels:
                lb_frame = np.asarray(lb[t])
                foreground_frame |= lb_frame > 0
                contours_frame += skseg.find_boundaries(lb_frame, mode="outer").astype(np.float32, copy=False)

            contours_frame /= n_labels
            if sigma is not None:
                contours_frame = scipy_ndi.gaussian_filter(contours_frame, sigma)
                max_val = float(np.max(contours_frame))
                if max_val > 0:
                    contours_frame = contours_frame / max_val

            detection[t] = foreground_frame
            contours[t] = contours_frame.astype(np.float32, copy=False)

        return detection, contours

    try:
        return _run_labels_to_contours(labels)
    except TypeError as exc:
        message = str(exc)
        if (
            "Implicit conversion to a NumPy array is not allowed" not in message
            and "cupy._core.core._ndarray_base" not in message
        ):
            raise

        trace.log.warning(
            "labels_to_contours failed with CuPy/NumPy mismatch; using CPU contour fallback."
        )

        labels_cpu = [np.asarray(to_cpu(label)) for label in labels]
        return _labels_to_contours_cpu(labels_cpu)

# def run_cellpose(trace):
#     """
#     Robust wrapper around Cellpose eval.
#     Accepts (H,W), (N,H,W) or list[(H,W)] as input.
#     Returns masks as (H,W) for single image or (N,H,W) for batches.
#     """
#     channels = [0, 0]  # grayscale
#     imgs = trace.current_raw

#     # ---- normalize input to a list of (H, W) ----
#     if isinstance(imgs, list):
#         X = [np.ascontiguousarray(im) for im in imgs]
#         H, W = X[0].shape
#     else:
#         arr = np.ascontiguousarray(imgs)
#         if arr.ndim == 2:
#             H, W = arr.shape
#             X = [arr]
#         elif arr.ndim == 3:
#             N, H, W = arr.shape
#             # Use a Python list: Cellpose will still honor batch_size
#             X = [arr[i] for i in range(N)]
#         else:
#             raise ValueError(f"Unsupported image shape {arr.shape}")

#     bs = min(getattr(trace, "cp_batch_size", 8), len(X))

#     with torch.inference_mode():
#         masks, flows, styles = trace.model.eval(
#             X,
#             diameter=trace.diameter,
#             flow_threshold=0.4,
#             cellprob_threshold=0.3,
#             channels=channels,
#             batch_size=bs,
#             augment=False,
#             net_avg=False,
#         )

#     # ---- normalize output to (N,H,W) or (H,W) ----
#     if isinstance(masks, list):
#         # Expected for list input; ensure 2D items then stack
#         if not all(m.ndim == 2 for m in masks):
#             shapes = [m.shape for m in masks]
#             raise ValueError(f"Unexpected mask dims from Cellpose: {shapes}")
#         masks = np.stack(masks, axis=0)
#     else:
#         # Single array returned
#         if masks.ndim == 2 and masks.shape == (H, W):
#             # single image case
#             trace.current_mask = masks
#             return
#         elif masks.ndim == 3 and masks.shape[1:] == (H, W):
#             pass  # already (N,H,W)
#         elif masks.ndim == 2 and masks.shape[0] == len(X) and masks.shape[1] in (H, W):
#             # Rare: (N, H) or (N, W) -> reshape using input (H,W)
#             try:
#                 masks = masks.reshape((len(X), H, W))
#             except Exception:
#                 raise ValueError(f"Unexpected mask shape {masks.shape} for input {(len(X), H, W)}")
#         else:
#             raise ValueError(f"Unexpected mask shape {masks.shape}")

#     # For batches, keep (N,H,W) so your rotation loop can index each mask
#     trace.current_mask = masks if masks.shape[0] > 1 else masks[0]
    
    ### TODO: MIGHT NEED TO EDIT THIS TO BE A DIFFERENT BEHAVIOR DURING ROTATIONS


def run_cellpose(trace):
    """Run cellpose."""
    channels = [0,0] # IF YOU HAVE GRAYSCALE

    mask, _, _ = trace.model.eval(trace.current_raw.copy(),
                                         diameter=trace.diameter,
                                         flow_threshold=0.4,
                                         cellprob_threshold=0.3,
                                         channels=channels)
    
    trace.current_mask = mask
    ### TODO: MIGHT NEED TO EDIT THIS TO BE A DIFFERENT BEHAVIOR DURING ROTATIONS

def analyze_segmenation(trace, idx=None):
    """Compute analyze segmenation."""
    control_pt_ijk =[int(trace.dim/2), int(trace.dim/2)]
    if idx is None:
        mask = trace.current_mask
    else:
        mask = trace.current_mask[idx]
    target_blob = mask[control_pt_ijk[1], control_pt_ijk[0]]
    if target_blob == 0:
        # print('NO TARGET BLOB')
        # Update trace parameters
        trace.df_current = None
        trace.centroid_ijk = None
        trace.found_mask = False
        return

    # get properties of identified components 
    props = measure.regionprops_table(mask, intensity_image=None, properties=['label',
                                                                                      'centroid',
                                                                                      'eccentricity',
                                                                                      'axis_major_length',
                                                                                      'axis_minor_length',
                                                                                      'orientation', 'equivalent_diameter_area'])
    df_coord = pd.DataFrame(props)
    centroid_ijk = df_coord.loc[df_coord['label']==target_blob, ['centroid-0' , 'centroid-1']].values[0]

    df_coord = df_coord[df_coord['label']==target_blob]
    df_coord['angle'] = 0

    v = df_coord['equivalent_diameter_area']
    # Update trace parameters
    trace.log.info(f"In analysis diameter: {v}")

    trace.df_current = df_coord
    trace.found_mask = True
    trace.centroid_ijk = centroid_ijk

def remove_background(image: ArrayLike, sigma=15.0) -> ArrayLike:
    """
    Removes background using morphological reconstruction by dilation.
    Reconstruction seeds are an extremely blurred version of the input.

    Parameters
    ----------
    imgs : ArrayLike
        Raw image.

    Returns
    -------
    ArrayLike
        Foreground image.
    """
    image = xp.asarray(image)
    ndi = import_module("scipy", "ndimage")
    seeds = ndi.gaussian_filter(image, sigma=sigma)
    background = reconstruction_by_dilation(seeds, image, iterations=100)
    foreground = np.maximum(image, background) - background
    return to_cpu(foreground)

def compute_mask_correlation(cellpose_labels_list, labels_t0, center_coords=(75, 75)):
    """
    Computes the correlation between the object at center_coords in labels_t0
    and the corresponding object at center_coords in each mask of cellpose_labels_list.

    Parameters:
    - cellpose_labels_list: list of dicts with keys 'mask', 'model', 'diameter'
    - labels_t0: label mask (Zarr array) at time t=0
    - center_coords: tuple of (y, x) coordinates

    Returns:
    - correlations: list of dicts containing overlap metrics and corresponding model and diameter
    """
    center_y, center_x = center_coords

    # Read labels_t0 data into NumPy array
    labels_t0_np = labels_t0[:]

    # Get the label at the center point in labels_t0
    center_label_t0 = labels_t0_np[int(center_y), int(center_x)]
    if center_label_t0 == 0:
        return None

    # Create a binary mask for the object of interest in labels_t0
    object_mask_t0 = (labels_t0_np == center_label_t0).astype(np.uint8)

    correlations = []

    for idx, entry in enumerate(cellpose_labels_list):
        mask = entry['mask']
        model = entry['model']
        diameter = entry['diameter']

        # Read mask data into NumPy array
        mask_np = mask[:][0, :, :]  # Assuming the first time point

        # Get the label at the center point in the current mask
        center_label_i = mask_np[int(center_y), int(center_x)]

        if center_label_i == 0:
            iou = 0.0
        else:
            # Create a binary mask for the object at center_coords in the current mask
            object_mask_i = (mask_np == center_label_i).astype(np.uint8)
            intersection = np.logical_and(object_mask_t0, object_mask_i).sum()
            union = np.logical_or(object_mask_t0, object_mask_i).sum()

            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union

        correlations.append({
            'mask_index': idx,
            'iou': iou,
            'model': model,
            'diameter': diameter
        })
    return correlations

def delete_zarr_temp_files(trace):
    """Delete zarr temp files."""
    folder = Path(trace.next_run_folder)
    for file in folder.iterdir():
        if file.name.startswith("cellpose_labels_"):
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

def trouble_shooting(trace):
    """Compute trouble shooting."""
    import torch

    trace.log.warning('Standard approach did not find nephron')
    # try:

    trace.log.warning('Trying ultracks approach')

    # Set up the volume for ultracks
    slice_transform, P, _ = set_slice_view_ut(trace)
    _, volume_array = get_volume(P, slice_transform, trace.current_chunk, size=[trace.dim, trace.dim, 10], spacing=[1.0, 1.0, 1.0])
    
    # Process the volume using ultracks functions
    chunks = (64, 64, 32)
    foreground = create_zarr(volume_array.shape, volume_array.dtype, os.path.join(trace.next_run_folder, "foreground.zarr"), chunks=chunks, overwrite=True)
    array_apply(
        volume_array.copy(),
        out_array=foreground,
        func=remove_background,
        sigma=200.0,
        axis=(0),
    )

    normalized = create_zarr(volume_array.shape, np.float16, os.path.join(trace.next_run_folder, "normalized.zarr"), chunks=chunks, overwrite=True)
    array_apply(
        foreground,
        out_array=normalized,
        func=normalize,
        gamma=0.1,
        axis=(0),
    )

    # Run cellpose at mutiple diameters and combine labels
    cellpose_labels_list = []
    diameters = [trace.diameter-trace.jitter, trace.diameter, trace.diameter+trace.jitter]
    for d in diameters:
        for model in [f for f in os.listdir(trace.model_suite) if not f.startswith('.')]:#[ "tubulemap/cellpose_tracker/models/CUBICcortex2", "tubulemap/cellpose_tracker/models/CUBICcortex", "tubulemap/cellpose_tracker/models/FLARE_cortex", "tubulemap/cellpose_tracker/models/TAL", "cyto2"]:
            cellpose_labels = create_zarr(volume_array.shape, np.uint16, os.path.join(trace.next_run_folder, f"cellpose_labels_{model}_{d}.zarr"), chunks=chunks, overwrite=True)
            array_apply(
                volume_array,
                out_array=cellpose_labels,
                func=Cellpose(
                    gpu=bool(getattr(trace, "use_GPU", False)),
                    pretrained_model=os.path.join(trace.model_suite, model),
                    device=torch.device(getattr(trace, "cuda_device", "cpu")),
                ),
                axis=(0),
                diameter=d,
                tile=False,
                normalize=True,
            )
            cellpose_labels_list.append({
                'mask': cellpose_labels,
                'model': model,
                'diameter': d
            })

    # Read zarr arrays into NumPy arrays and sum them
    combined_labels = np.sum([entry['mask'][:] for entry in cellpose_labels_list], axis=0)

    mask_index = []
    for i in range(combined_labels.shape[0]):
        if combined_labels[i, int(trace.dim/2), int(trace.dim/2)] != 0:
            print('mask_index added', i)
            mask_index.append(i)

    if len(mask_index) == 1:
        mask_index = [mask_index[0], mask_index[0]]

    if len(mask_index) == 0:
        trace.log.info('Cellpose segmentation in ultracks found no masks in center of volume')
        trace.df_current = None
        trace.centroid_ijk = None
        trace.current_mask = None
        trace.found_mask = False
        trace.ts_correlations = None
        return

    combined_labels = combined_labels[mask_index]
    volume_array_new = volume_array[mask_index]

    cellpose_labels_list_new = []
    for entry in cellpose_labels_list:
        new_mask = entry['mask'][mask_index]
        entry['mask'] = new_mask
        cellpose_labels_list_new.append(entry)
        
    cellpose_labels_list = cellpose_labels_list_new

    foreground = create_zarr(volume_array_new.shape, volume_array_new.dtype, os.path.join(trace.next_run_folder, "foreground.zarr"), chunks=chunks, overwrite=True)
    array_apply(
        volume_array_new.copy(),
        out_array=foreground,
        func=remove_background,
        sigma=200.0,
        axis=(0),
    )

    normalized = create_zarr(volume_array_new.shape, np.float16, os.path.join(trace.next_run_folder, "normalized.zarr"), chunks=chunks, overwrite=True)
    array_apply(
        foreground,
        out_array=normalized,
        func=normalize,
        gamma=0.1,
        axis=(0),
    )

    # Get detection and contours (CPU fallback handles CuPy/skimage mismatch).
    detection, contours = _labels_to_contours_safe(
        [combined_labels],
        sigma=4.0,
        trace=trace,
    )

    # Set up ultracks configuration
    config = MainConfig()
    config.segmentation_config.n_workers = 1
    config.segmentation_config.min_area = 50
    config.segmentation_config.min_frontier = -0.1
    config.linking_config.max_neighbors = 5
    config.linking_config.max_distance = 25
    config.linking_config.n_workers = 1
    config.tracking_config.division_weight = -100000 #check with git
    config.tracking_config.disappear_weight = -10
    config.tracking_config.appear_weight = -5
    config.tracking_config.window_size = 45
    config.tracking_config.overlap_size = 3
    config.tracking_config.solution_gap = 0.01
    config.data_config.working_dir = trace.next_run_folder
    
    print('Tracking')
    # Run tracking with ultracks
    track(
        config,
        detection=detection,
        edges=contours,
        images=[normalized],
        overwrite=True
    )
    print('Track completed')
    tracks_df, lineage_graph = to_tracks_layer(config)
    # print("check df if exit:", tracks_df)
    tracking_labels = tracks_to_zarr(config, tracks_df)

    print('Check tracking labels')
    print(tracking_labels.shape)

    # Now find the track whose mask at t=0 overlaps with the center point
    center_y, center_x = int(trace.dim/2), int(trace.dim/2)  # Center of the image

    # Get the labels image at t=0
    labels_t0 = tracking_labels[0]  # Assuming time dimension is first

    # Get the label at the center point
    center_label = labels_t0[int(center_y), int(center_x)]

    print('Labels and center', labels_t0, center_label)

    if center_label == 0:
        trace.log.warning('No mask at center point found in ultracks labels at t=0')
        trace.log.info('Cellpose segmentation in ultracks found no masks in center of volume')

        # Update trace parameters
        trace.df_current = None
        trace.centroid_ijk = None
        trace.current_mask = None
        trace.found_mask = False
        trace.ts_correlations = None
        return
    
    # Now compute the correlation between cellpose_labels_list and labels_t0
    correlations = compute_mask_correlation(cellpose_labels_list, labels_t0, center_coords=(int(trace.dim/2), int(trace.dim/2)))

    # Print or log the correlations
    for corr in correlations:
        print(f"Mask {corr['model']}_{corr['diameter']}: IoU with labels_t0 object at center = {corr['iou']:.4f}")

    # Get the track_id corresponding to this label
    track_ids = tracks_df[(tracks_df['t'] == 0) & (tracks_df['track_id'] == center_label)]['track_id'].values

    track_id = track_ids[0]

    # Get the centroid of the mask at t=0
    track_data = tracks_df[(tracks_df['track_id'] == track_id) & (tracks_df['t'] == 0)]
    y = track_data['y'].values[0]
    x = track_data['x'].values[0]

    centroid_ijk = np.array([y, x])

    # Prepare the output dataframe
    df = pd.DataFrame({
        'label': [center_label],
        'centroid-0': [y],
        'centroid-1': [x],
        'eccentricity': [0],  # Placeholder
        'axis_major_length': [0],  # Placeholder
        'axis_minor_length': [0],  # Placeholder
        'orientation': [0],  # Placeholder
        'equivalent_diameter_area': [0],  # Placeholder
        'angle': [0],
    })

    # Update trace parameters
    trace.df_current = df
    trace.centroid_ijk = centroid_ijk
    trace.current_mask = labels_t0
    trace.found_mask = True
    trace.ts_correlations = correlations

    trace.log.info('Troubleshoot was successful')
    # except:
    #     trace.log.info('Random error on ultrack not accounted for')
    #     trace.df_current = None
    #     trace.centroid_ijk = None
    #     trace.current_mask = None
    #     trace.found_mask = False
    #     trace.ts_correlations = None

    delete_zarr_temp_files(trace)
    


def trouble_shooting_diameter_only(trace):
    """Compute trouble shooting diameter only."""
    import torch

    trace.log.warning('Standard approach did not find nephron')
    try:

        trace.log.warning('Trying ultracks approach')

        # Set up the volume for ultracks
        slice_transform, P, _ = set_slice_view_ut(trace)
        _, volume_array = get_volume(P, slice_transform, trace.current_chunk, size=[trace.dim, trace.dim, 10], spacing=[1.0, 1.0, 1.0])
        
        # Process the volume using ultracks functions
        chunks = (64, 64, 32)
        foreground = create_zarr(volume_array.shape, volume_array.dtype, os.path.join(trace.next_run_folder, "foreground.zarr"), chunks=chunks, overwrite=True)
        array_apply(
            volume_array.copy(),
            out_array=foreground,
            func=remove_background,
            sigma=200.0,
            axis=(0),
        )

        normalized = create_zarr(volume_array.shape, np.float16, os.path.join(trace.next_run_folder, "normalized.zarr"), chunks=chunks, overwrite=True)
        array_apply(
            foreground,
            out_array=normalized,
            func=normalize,
            gamma=0.1,
            axis=(0),
        )

        # Run cellpose at mutiple diameters and combine labels
        cellpose_labels_list = []
        diameters = [trace.diameter-trace.jitter, trace.diameter, trace.diameter+trace.jitter]
        for d in diameters:
            for model in [trace.model_name]:#[ "tubulemap/cellpose_tracker/models/CUBICcortex2", "tubulemap/cellpose_tracker/models/CUBICcortex", "tubulemap/cellpose_tracker/models/FLARE_cortex", "tubulemap/cellpose_tracker/models/TAL", "cyto2"]:
                cellpose_labels = create_zarr(volume_array.shape, np.uint16, os.path.join(trace.next_run_folder, f"cellpose_labels_{model}_{d}.zarr"), chunks=chunks, overwrite=True)
                array_apply(
                    volume_array,
                    out_array=cellpose_labels,
                    func=Cellpose(
                        gpu=bool(getattr(trace, "use_GPU", False)),
                        pretrained_model=os.path.join(trace.model_suite, model),
                        device=torch.device(getattr(trace, "cuda_device", "cpu")),
                    ),
                    axis=(0),
                    diameter=d,
                    tile=False,
                    normalize=True,
                )
                cellpose_labels_list.append({
                    'mask': cellpose_labels,
                    'model': model,
                    'diameter': d
                })

        # Read zarr arrays into NumPy arrays and sum them
        combined_labels = np.sum([entry['mask'][:] for entry in cellpose_labels_list], axis=0)

        mask_index = []
        for i in range(combined_labels.shape[0]):
            if combined_labels[i, int(trace.dim/2), int(trace.dim/2)] != 0:
                print('mask_index added', i)
                mask_index.append(i)

        if len(mask_index) == 1:
            mask_index = [mask_index[0], mask_index[0]]
        
        if len(mask_index) == 0:
            trace.log.info('Cellpose segmentation in ultracks found no masks in center of volume')
            trace.df_current = None
            trace.centroid_ijk = None
            trace.current_mask = None
            trace.found_mask = False
            trace.ts_correlations = None
            return

        combined_labels = combined_labels[mask_index]
        volume_array_new = volume_array[mask_index]

        cellpose_labels_list_new = []
        for entry in cellpose_labels_list:
            new_mask = entry['mask'][mask_index]
            entry['mask'] = new_mask
            cellpose_labels_list_new.append(entry)
            
        cellpose_labels_list = cellpose_labels_list_new

        foreground = create_zarr(volume_array_new.shape, volume_array_new.dtype, os.path.join(trace.next_run_folder, "foreground.zarr"), chunks=chunks, overwrite=True)
        array_apply(
            volume_array_new.copy(),
            out_array=foreground,
            func=remove_background,
            sigma=200.0,
            axis=(0),
        )

        normalized = create_zarr(volume_array_new.shape, np.float16, os.path.join(trace.next_run_folder, "normalized.zarr"), chunks=chunks, overwrite=True)
        array_apply(
            foreground,
            out_array=normalized,
            func=normalize,
            gamma=0.1,
            axis=(0),
        )

        # Get detection and contours
        detection, contours = _labels_to_contours_safe(
            [combined_labels],
            sigma=4.0,
            trace=trace,
        )

        # Set up ultracks configuration
        config = MainConfig()
        config.segmentation_config.n_workers = 1
        config.segmentation_config.min_area = 50
        config.segmentation_config.min_frontier = -0.1
        config.linking_config.max_neighbors = 5
        config.linking_config.max_distance = 25
        config.linking_config.n_workers = 1
        config.tracking_config.division_weight = -100000 #check with git
        config.tracking_config.disappear_weight = -10
        config.tracking_config.appear_weight = -5
        config.tracking_config.window_size = 45
        config.tracking_config.overlap_size = 3
        config.tracking_config.solution_gap = 0.01
        config.data_config.working_dir = trace.next_run_folder
        
        print('Tracking')
        # Run tracking with ultracks
        track(
            config,
            detection=detection,
            edges=contours,
            images=[normalized],
            overwrite=True
        )
        print('Track completed')
        tracks_df, lineage_graph = to_tracks_layer(config)
        # print("check df if exit:", tracks_df)
        tracking_labels = tracks_to_zarr(config, tracks_df)

        print('Check tracking labels')
        print(tracking_labels.shape)

        # Now find the track whose mask at t=0 overlaps with the center point
        center_y, center_x = int(trace.dim/2), int(trace.dim/2)  # Center of the image

        # Get the labels image at t=0
        labels_t0 = tracking_labels[0]  # Assuming time dimension is first

        # Get the label at the center point
        center_label = labels_t0[int(center_y), int(center_x)]

        print('Labels and center', labels_t0, center_label)

        if center_label == 0:
            trace.log.warning('No mask at center point found in ultracks labels at t=0')
            trace.log.info('Cellpose segmentation in ultracks found no masks in center of volume')

            # Update trace parameters
            trace.df_current = None
            trace.centroid_ijk = None
            trace.current_mask = None
            trace.found_mask = False
            trace.ts_correlations = None
            return
        
        # Now compute the correlation between cellpose_labels_list and labels_t0
        correlations = compute_mask_correlation(cellpose_labels_list, labels_t0, center_coords=(int(trace.dim/2), int(trace.dim/2)))

        # Print or log the correlations
        for corr in correlations:
            print(f"Mask {corr['model']}_{corr['diameter']}: IoU with labels_t0 object at center = {corr['iou']:.4f}")

        # Get the track_id corresponding to this label
        track_ids = tracks_df[(tracks_df['t'] == 0) & (tracks_df['track_id'] == center_label)]['track_id'].values

        track_id = track_ids[0]

        # Get the centroid of the mask at t=0
        track_data = tracks_df[(tracks_df['track_id'] == track_id) & (tracks_df['t'] == 0)]
        y = track_data['y'].values[0]
        x = track_data['x'].values[0]

        centroid_ijk = np.array([y, x])

        # Prepare the output dataframe
        df = pd.DataFrame({
            'label': [center_label],
            'centroid-0': [y],
            'centroid-1': [x],
            'eccentricity': [0],  # Placeholder
            'axis_major_length': [0],  # Placeholder
            'axis_minor_length': [0],  # Placeholder
            'orientation': [0],  # Placeholder
            'equivalent_diameter_area': [0],  # Placeholder
            'angle': [0],
        })

        # Update trace parameters
        trace.df_current = df
        trace.centroid_ijk = centroid_ijk
        trace.current_mask = labels_t0
        trace.found_mask = True
        trace.ts_correlations = correlations

        trace.log.info('Troubleshoot was successful')
    except:
        trace.log.info('Random error on ultrack not accounted for')
        trace.df_current = None
        trace.centroid_ijk = None
        trace.current_mask = None
        trace.found_mask = False
        trace.ts_correlations = None
    
    delete_zarr_temp_files(trace)
