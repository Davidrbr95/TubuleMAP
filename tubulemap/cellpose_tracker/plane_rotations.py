import numpy as np
from tubulemap.cellpose_tracker.vector_ops import *
from tubulemap.cellpose_tracker.geometry import *
from tubulemap.cellpose_tracker.segmentation import *
import math
import copy

import numpy as np
import math

def _unit(v):
    """Compute unit."""
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _rodrigues_rotate(v, k, angle_rad):
    """Rotate vector v about unit axis k by angle (radians)."""
    k = _unit(k)
    v_par = k * np.dot(k, v)
    v_perp = v - v_par
    return v_par + v_perp * math.cos(angle_rad) + np.cross(k, v) * math.sin(angle_rad)

def _inplane_basis_from_transform_at(trace, y_px, x_px):
    """3D in-plane basis at (y,x) using your ijk2ras mapping."""
    p0 = np.asarray(ijk2ras(y_px,   x_px,   trace.current_slice_transform), dtype=float).ravel()
    px = np.asarray(ijk2ras(y_px,   x_px+1, trace.current_slice_transform), dtype=float).ravel()
    py = np.asarray(ijk2ras(y_px+1, x_px,   trace.current_slice_transform), dtype=float).ravel()
    ex = _unit(px - p0)   # ∂RAS/∂x
    ey = _unit(py - p0)   # ∂RAS/∂y
    n  = _unit(np.cross(ex, ey))  # current slice normal
    return ex, ey, n, p0

def compute_rotation_vectors(trace, distance_ignored, prev_point_ignored):
    """
    NEW behavior:
      - Rotate the CURRENT SLICE PLANE about the ellipse MINOR axis through the centroid,
        by the fixed angles in trace.rot_angles (constructed from rotation_angle, angle_steps).
      - Produces a list of rotated normals in RAS for set_slice_view(...).

    Kept the signature for compatibility; the 'distance' and 'prev_point' are not used.
    """
    trace.log.info("Computing vectors for rotation (axis-angle about MINOR axis)")

    # 1) angles to try (deg)
    angles_deg = np.arange(-trace.rotation_angle,
                           trace.rotation_angle + trace.angle_steps,
                           trace.angle_steps, dtype=float)
    # angles_deg = np.linspace(-trace.rotation_angle,trace.rotation_angle, trace.angle_steps)
    trace.rot_angles = angles_deg

    # 2) centroid in pixels and RAS
    y0 = float(trace.df_current.iloc[-1]['centroid-0'])
    x0 = float(trace.df_current.iloc[-1]['centroid-1'])
    center_ras = np.asarray(ijk2ras(y0, x0, trace.current_slice_transform), dtype=float).ravel()

    # 3) local 3D basis at centroid from the slice transform
    ex, ey, n0, _ = _inplane_basis_from_transform_at(trace, y0, x0)

    # If you want the new normal to keep the same "hemisphere" as your last vector:
    if hasattr(trace, "vectors") and len(trace.vectors) > 0 and trace.vectors[-1] is not None:
        try:
            last_v = np.asarray(trace.vectors[-1], dtype=float).ravel()
            if np.dot(n0, last_v) < 0:
                n0 = -n0
        except Exception:
            pass

    # 4) ellipse orientation (image coords, y-down). Minor axis in image is (-sinθ, -cosθ)
    theta = float(trace.df_current.iloc[-1]['orientation'])  # radians
    u_minor_img = np.array([-math.sin(theta), -math.cos(theta)], dtype=float)

    # Project image minor axis into RAS via in-plane basis
    axis_minor_ras = _unit(u_minor_img[0] * ex + u_minor_img[1] * ey)

    # 5) rotate the slice normal around the minor axis by each angle
    vt_options = []
    for a_deg in angles_deg:
        a_rad = math.radians(a_deg)
        n_rot = _unit(_rodrigues_rotate(n0, axis_minor_ras, a_rad))
        vt_options.append(n_rot.reshape(3, 1))

    trace.rot_vectors = vt_options

    # (Optional) quick visualization in image space if you want to keep your debug plot:
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(trace.current_mask)
    # plt.plot([x0], [y0], '.g', markersize=12)
    # plt.title("Axis-angle rotation about MINOR axis")
    # plt.savefig('Can_delete.png', dpi=300)


def weighted_average(x, w):
    """Compute weighted average."""
    wa = np.dot(x, w)/np.sum(w)
    idx = np.argmin(np.abs(x-wa))
    return idx

# def compute_rotation_vectors(trace, distance, prev_point):
#     trace.log.info("Computing vectors for rotation")
#     orientation = trace.df_current.iloc[-1]['orientation']
    
    
#     angles = np.arange(-trace.rotation_angle,trace.rotation_angle+trace.angle_steps, trace.angle_steps)
#     trace.rot_angles = angles
#     radian_angles = np.radians(angles)
#     tan_values = np.tan(radian_angles)
    
#     x01 = int(trace.dim/2)
#     y01 = int(trace.dim/2) 

#     plt.figure()
#     plt.imshow(trace.current_mask)
#     x1 = x01 + math.cos(orientation) * 0.5 * 10
#     y1 = y01 - math.sin(orientation) * 0.5 * 10
#     x2 = x01 - math.sin(orientation) * 0.5 * 20
#     y2 = y01 - math.cos(orientation) * 0.5 * 20
#     plt.plot((x01, x1), (y01, y1), '-r', linewidth=2.5)
#     plt.plot((x01, x2), (y01, y2), '-r', linewidth=2.5)
#     plt.plot(x01, y01, '.g', markersize=15)

#     x_11 = x01 - math.sin(orientation) * tan_values *  distance #trace_parameters['stepsize'] #5
#     y_11 = y01 - math.cos(orientation) * tan_values *  distance #trace_parameters['stepsize']# 5
#     plt.plot(x_11, y_11, 'r-o')
#     plt.savefig('Can_delete.png', dpi=300)

#     x221 = list(x_11)
#     y221 = list(y_11)
   
#     ras_coordinates1 = [np.expand_dims(np.array(ijk2ras(y, x, trace.current_slice_transform)), -1) for x, y in zip(x221, y221)]
    
    
#     vt_options11 = [np.array(r) - prev_point for r in ras_coordinates1]
#     vt_options11 = [v/np.linalg.norm(v) for v in vt_options11]
#     vt_options = vt_options11
#     trace.rot_vectors = vt_options

def apply_rotations(trace, ras_centroid_1st_attempt_v):
    """Apply rotations."""
    trace.log.info("Applying rotations")
    centroid_coll = []
    raw_image_list = []
    slice_transforms = []
    for idx, v in enumerate(trace.rot_vectors):
        v = v.reshape(3,1)
        set_slice_view(trace, vector = v, points=ras_centroid_1st_attempt_v) 
        get_frame(trace)
        raw_image_list.append(trace.current_raw)
        slice_transforms.append(trace.current_slice_transform)

    #set the current image to the list of images obtained at different rotations
    trace.current_raw = raw_image_list 

    run_cellpose(trace)

    for idx, mask in enumerate(trace.current_mask):
        analyze_segmenation(trace, idx=idx)
        centroid_coll.append(trace.centroid_ijk)
        if not trace.found_mask:
            continue
        trace.df_current['idx'] = idx
        trace.df_current['angle'] = trace.rot_angles[idx]
        trace.rot_df = pd.concat([trace.rot_df, trace.df_current])
    trace.rot_df = trace.rot_df.reset_index(drop=True)

    
def identify_best_plane(trace, ecc_to_beat, ras_centroid_1st_attempt_v, trace_bk):
    """Identify best plane."""
    trace.log.info("Identifying the best plane after rotations")
    if len(trace.rot_df)==0:
        # trace.current_slice_transform = trace_bk.current_slice_transform
        # trace.current_raw = trace_bk.current_raw
        # trace.current_mask = trace_bk.current_mask
        # trace.df_current = trace_bk.df_current
        # trace.found_mask = trace_bk.found_mask
        # trace.centroid_ijk = trace_bk.centroid_ijk
        # trace.vectors[-1] = trace_bk.vectors[-1]
        trace.restore_rotation_backup(trace_bk)
        # set_slice_view(trace)
        # get_frame(trace)
        # run_cellpose(trace)
        # analyze_segmenation(trace)

        trace.rot_improved_ecc = False
        trace.log.info('Rotations failed to find any planes with segmentable objects')
        return
    
    if np.min(trace.rot_df['eccentricity'])<ecc_to_beat:
        data = trace.rot_df[['angle', 'eccentricity']].to_numpy()
        # TODO: add a variable that allows to select between weighted average or just minimum ecc
        idx_data = weighted_average(data[:,0], 1-data[:,1])
        idx = trace.rot_df.loc[idx_data, 'idx']
        angle = trace.rot_df.loc[idx_data, 'angle']
        v = trace.rot_vectors[idx].reshape(3,1)
        set_slice_view(trace, vector = v, points=ras_centroid_1st_attempt_v)
        trace.current_mask = trace.current_mask[idx]
        trace.current_raw = trace.current_raw[idx]
        col_keep = ['label','centroid-0','centroid-1','eccentricity','axis_major_length','axis_minor_length','orientation', 'equivalent_diameter_area', 'angle']
        new_values = trace.rot_df.iloc[[idx_data]][col_keep] # Correct new behavior
        trace.df_current = new_values
        trace.centroid_ijk = trace.df_current[['centroid-0', 'centroid-1']].to_numpy()[0]

        ## OLD behavior comment out in final version
        # new_values = trace.rot_df.loc[trace.rot_df['eccentricity']==np.min(trace.rot_df['eccentricity']), col_keep] # OLD BEHAVIOR - this would mess up everything based on this dataframe but not the centroid as it was outputed in a different way     
        # trace.df_current = new_values
        ###
        
        trace.log.info('Rotation was successful')
        trace.rot_improved_ecc = True
        trace.rot_final_angle = str(angle)
        trace.rot_angles  = list([str(j) for j in trace.rot_angles])
        trace.vectors[-1] = v 
    else:
        # trace.current_slice_transform = trace_bk.current_slice_transform
        # trace.current_raw = trace_bk.current_raw
        # trace.current_mask = trace_bk.current_mask
        # trace.df_current = trace_bk.df_current
        # trace.found_mask = trace_bk.found_mask
        # trace.centroid_ijk = trace_bk.centroid_ijk
        # trace.vectors[-1] = trace_bk.vectors[-1]

        trace.restore_rotation_backup(trace_bk) 
        # set_slice_view(trace)
        # get_frame(trace)
        # run_cellpose(trace)
        # analyze_segmenation(trace)
        trace.rot_improved_ecc = False
        trace.log.info('Rotations failed to find plane that improved the eccentricity')

def rotate_to_improve_ecc(trace):
    """Compute rotate to improve ecc."""
    ecc_to_beat = trace.df_current.iloc[-1]['eccentricity']
    trace.log.info('Applying rotations: The starting ecc is {%0.6f}', ecc_to_beat)
    #get the centroid | TODO: Is this just centroid ijk
    y0, x0 = trace.df_current.iloc[-1]['centroid-0'], trace.df_current.iloc[-1]['centroid-1']
    
    #ras_origin = centroid_found_mask
    ras_centroid_1st_attempt  = ijk2ras(y0, x0, trace.current_slice_transform) #rename ras_centroid

    ras_centroid_1st_attempt_v = np.expand_dims(np.array(ras_centroid_1st_attempt), -1)
    
    ras_previous_point = get_point_curve_ras(trace.pointIndex-1, trace.curvenode)
    V_last = (ras_centroid_1st_attempt_v-ras_previous_point)
    distance = np.linalg.norm(V_last) # length of the vector
    V_last = V_last/np.linalg.norm(V_last)
    
    

    # trace_bk = copy.deepcopy(trace)
    trace_bk = trace.make_rotation_backup()

    trace.vectors[-1] = V_last
    # df_current_bk = trace.df_current
    # centroid_ijk_bk = trace.centroid_ijk
    # found_mask_bk = trace.found_mask
    # current_mask_bk = trace.current_mask
    # current_raw_bk = trace.current_raw
    # current_slice_transform_bk = trace.current_slice_transform
    # (df_current_bk, centroid_ijk_bk, centroid_ijk_bk)

    # Set slice based of vector between previous point throuhg centroid of the mask gen in first attempt
    set_slice_view(trace, vector = V_last, points=ras_centroid_1st_attempt_v) 
    get_frame(trace)
    run_cellpose(trace)
    analyze_segmenation(trace)

    if not trace.found_mask:
        trace.log.info("After applying centroid correct, no mask was found: applying ultrack troubleshooting")
        if trace.use_ultrack:
            trouble_shooting(trace)
        if trace.found_mask:
            trace.log.info('Troubleshootin in rotation was succesful')
            analyze_segmenation(trace)
            # trace_bk = copy.deepcopy(trace)
            trace_bk = trace.make_rotation_backup()

        else: # revert back to initial configuration prior to centroid shifiting
            trace.log.info("No mask found during ultrack trouble shooting, reverting to original configuration")
            # set_slice_view(trace)
            # get_frame(trace)
            # run_cellpose(trace)
            # analyze_segmenation(trace)
            # trace = trace_bk.copy()

            # trace.current_slice_transform = trace_bk.current_slice_transform
            # trace.current_raw = trace_bk.current_raw
            # trace.current_mask = trace_bk.current_mask
            # trace.df_current = trace_bk.df_current
            # trace.found_mask = trace_bk.found_mask
            # trace.centroid_ijk = trace_bk.centroid_ijk
            # trace.vectors[-1] = trace_bk.vectors[-1]
            trace.restore_rotation_backup(trace_bk)
    
    # trace_bk = copy.deepcopy(trace)
    trace_bk = trace.make_rotation_backup()
    
    compute_rotation_vectors(trace, distance, ras_previous_point)

    apply_rotations(trace, ras_centroid_1st_attempt_v)

    identify_best_plane(trace, ecc_to_beat, ras_centroid_1st_attempt_v, trace_bk)
    