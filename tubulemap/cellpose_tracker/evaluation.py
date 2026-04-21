import os
import math
from math import sqrt
from tubulemap.cellpose_tracker.io_utils import *

def distance(p1, p2):
    """Compute distance."""
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def resample_curve_fixed_spacing(points, step):
    """
    Resample a polyline so that the spacing between consecutive points is constant.
    
    Parameters:
        points: A list of 3D points defining the curve, e.g. [[x0,y0,z0], [x1,y1,z1], ...]
        step: The desired distance between consecutive resampled points.
        
    Returns:
        A list of points (x,y,z) that are equally spaced by 'step' along the original curve.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required to define a curve.")

    # Compute cumulative arc length
    cumulative_length = [0.0]
    for i in range(1, len(points)):
        dist = distance(points[i-1], points[i])
        cumulative_length.append(cumulative_length[-1] + dist)

    total_length = cumulative_length[-1]
    if total_length == 0:
        # All points are identical
        return [points[0]]
    
    # Determine how many steps we can fit from start (0) to end (total_length)
    # We'll have a point at 0, step, 2*step, ... up to the end.
    n_samples = int(total_length / step) + 1
    
    # Generate target arc lengths
    target_lengths = [i * step for i in range(n_samples)]
    # Ensure the last point ends exactly at the end of the curve
    # if due to rounding we went beyond total_length, set it to total_length.
    if target_lengths[-1] > total_length:
        target_lengths[-1] = total_length

    resampled = []

    seg_idx = 0  # Index to track which segment of the original points we are on
    for tlen in target_lengths:
        # Advance seg_idx until we find the segment that contains 'tlen'
        while seg_idx < len(points)-1 and cumulative_length[seg_idx+1] < tlen:
            seg_idx += 1
        
        # If tlen matches a known point exactly (rare but could happen)
        if math.isclose(cumulative_length[seg_idx], tlen, rel_tol=1e-12):
            resampled.append(points[seg_idx])
            continue

        # Interpolate between points[seg_idx] and points[seg_idx+1]
        seg_length = cumulative_length[seg_idx+1] - cumulative_length[seg_idx]
        
        # Handle edge cases: if two consecutive points are identical
        if seg_length == 0:
            resampled.append(points[seg_idx])
            continue

        frac = (tlen - cumulative_length[seg_idx]) / seg_length
        p1 = points[seg_idx]
        p2 = points[seg_idx+1]

        # Linear interpolation in 3D
        x = p1[0] + frac * (p2[0] - p1[0])
        y = p1[1] + frac * (p2[1] - p1[1])
        z = p1[2] + frac * (p2[2] - p1[2])

        resampled.append([x, y, z])

    return resampled

def master_compare(trace, error=""):
    """Compute master compare."""
    pred_points = resample_curve_fixed_spacing(trace.curvenode, trace.resample_step_size)
    gt_points = resample_curve_fixed_spacing(trace.ground_truth_curvenode, trace.resample_step_size)
    trace.monotonic_index, distance, next_starting_points = compare_points(pred_points, gt_points, window_size = trace.gt_window_size, monotonic_index = trace.monotonic_index)
    distance_break, ending_break = False, False

    if distance >= trace.break_distance:
        error = "Deviation from GT"
        distance_break = True

    if trace.monotonic_index+ trace.gt_window_size >= len(gt_points):
        error = "End of GT"
        ending_break = True
    scale_zyx = getattr(trace, "run_level_scale_zyx", [1.0, 1.0, 1.0])
    save_curve_nodes(trace, reset = True, finished=ending_break)
    save_curve_nodes_gt(
        gt_points,
        os.path.join(trace.next_run_folder, 'Resampled_GT' ),
        reset = True,
        error=error,
        scale_zyx=scale_zyx,
    )
    save_curve_nodes_gt(
        next_starting_points,
        os.path.join(trace.next_run_folder, 'Next_Starting_Points' ),
        reset = True,
        error=error,
        scale_zyx=scale_zyx,
    )
    trace.ground_truth_deviation.append((trace.monotonic_index, distance))
    return distance_break, ending_break


def compare_points(pred_points, gt_points, window_size, monotonic_index):
    # print(pred_points)
    """Compare points."""
    if len(pred_points) < window_size:
        return monotonic_index, -1, gt_points[monotonic_index:]

    if len(pred_points) >= window_size:
        pred_window = pred_points[-window_size:]
        best_j, best_avg_dist = find_best_gt_segment(gt_points, pred_window, monotonic_index, window_size=window_size)
        monotonic_index = best_j
        print("New segment:", monotonic_index, "Average distance:", best_avg_dist, "Best j:", best_j+window_size, "GT len:", len(gt_points))

    new_start_points = gt_points[best_j+window_size:]
    return monotonic_index, best_avg_dist, new_start_points


def find_best_gt_segment(gt_points, pred_window, monotonic_index, window_size):
    """Find best gt segment."""
    best_avg_dist = float('inf')
    best_j = None
    
    # Make sure we don't exceed the length of GT
    max_start = len(gt_points) - window_size
    
    for j in range(monotonic_index, max_start+1):
        # Compute the average distance
        total_dist = 0
        for k in range(window_size):
            total_dist += distance(pred_window[k], gt_points[j+k])
        avg_dist = total_dist / window_size
        
        if avg_dist < best_avg_dist:
            best_avg_dist = avg_dist
            best_j = j

    return best_j, best_avg_dist
