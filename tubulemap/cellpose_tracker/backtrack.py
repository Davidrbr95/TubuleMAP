import numpy as np
from sklearn.neighbors import KDTree

def check_backtrack(trace):
    """Check backtrack."""
    trace.log.info("Starting the backtrack check")
    curve = np.vstack(trace.curvenode)
    back_flag, delete_point_index = detect_turnbacks(curve, window_size=trace.bktk_window_size, 
                                                     search_radius=trace.bktk_search_radius,
                                                     direction_threshold=trace.bktk_dir_thresh, 
                                                     min_gap=trace.bktk_min_gap)
    if back_flag:
        # Remove the last 'delete_point_index' points from curvenode if valid
        if delete_point_index > 0 and delete_point_index <= len(trace.curvenode):
            trace.log.info('Backtrack was identified. Deleting points after index: ' + str(delete_point_index))
            trace.curvenode = trace.curvenode[:-delete_point_index]
        else:
            # Log a warning if delete_point_index is invalid
            trace.log.info('Backtrack was identified. Deleting points after index: ' + str(delete_point_index))
    return back_flag


def compute_local_direction(curve, idx, window_size):
    """
    Compute a direction vector at a given index by looking at points
    'window_size' steps behind.
    """
    start_idx = max(0, idx - window_size)
    # Direction vector: from p_(idx-window) to p_idx
    direction = curve[idx] - curve[start_idx]
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    return direction / norm

def detect_turnbacks(curve, 
                     window_size=5, 
                     search_radius=0.5, 
                     direction_threshold=-0.9, 
                     min_gap=20):
    """
    Detect turnbacks in a given 3D curve.
    
    Parameters
    ----------
    curve : np.ndarray
        An Nx3 array of points (x, y, z).
    window_size : int
        Number of points to look back to compute a local average direction.
    search_radius : float
        Radius within which we consider points to be "close" to the current point.
    direction_threshold : float
        Dot product threshold to consider directions as opposite. 
    min_gap : int
        Minimum number of points between the current point and a candidate old point
        to ensure we're not just detecting trivial reversals immediately next to each other.
    
    Returns
    -------
    tuple
        A tuple (turnbacks: bool, delete_point_index: Optional[float])
    """
    N = len(curve)
    if N <= window_size:
        return False, None

    tree = KDTree(curve)
    turnbacks = []
    delete_point_index = None

    for i in range(window_size, N):
        current_direction = compute_local_direction(curve, i, window_size)
        if current_direction is None:
            continue

        # Query nearby points from older parts of the curve
        idx_candidates = tree.query_radius(curve[i].reshape(1, -1), r=search_radius)[0]
        
        # Filter out indices that are too close to i in time
        idx_candidates = idx_candidates[idx_candidates < i - min_gap]
        
        if len(idx_candidates) == 0:
            continue

        for j in idx_candidates:
            old_direction = compute_local_direction(curve, j, window_size)
            if old_direction is None:
                continue
            
            # Check if directions are nearly opposite
            dot = np.dot(current_direction, old_direction)
            if dot < direction_threshold:
                turnbacks.append((i, j))
                # Ensure delete_point_index is properly calculated
                delete_point_index = int(i-(i + j) / 2)
                print(f"Turnback detected: current index = {i}, old index = {j}, delete_point_index = {delete_point_index}")
                break
        
        if len(turnbacks) > 0:
            break
             
    if len(turnbacks) > 0:
        return True, delete_point_index

    return False, None