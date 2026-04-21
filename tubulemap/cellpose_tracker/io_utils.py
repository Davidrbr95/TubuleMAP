import re
import os
import zarr
import json
import time
import cv2
import numpy as np

from tubulemap.utils.zarr_resolution import DEFAULT_SCALE_ZYX


def get_napari_image_path(viewer, layer_name):
    """
    Retrieve the image path from a specified layer in the Napari viewer.

    This function accesses the layer with the given name in the provided Napari viewer
    and returns the file path of its data source.

    Args:
        viewer: The Napari viewer instance.
        layer_name (str): The name of the layer whose image path is desired.

    Returns:
        str: The file path of the image source for the specified layer.
    """
    layer = viewer.layers[layer_name]
    return layer.source.path

def get_max_run_number(directory_path):
    """
    Get the maximum run number from a directory.

    This function searches through the specified directory for subdirectories
    whose names match the pattern 'Run_<number>'. It extracts the numeric part
    from each matching subdirectory and returns the highest run number found.
    If the directory is empty or no subdirectories match the expected pattern,
    the function returns -1.

    Args:
        directory_path (str): The path to the directory containing run subdirectories.

    Returns:
        int: The maximum run number found, or -1 if no valid run directories exist.
    """
    max_run_number = -1
    pattern = re.compile(r'^Run_(\d+)$')
    if len(os.listdir(directory_path))==0:
        return -1
    for entry in os.listdir(directory_path):
        match = pattern.match(entry)
        if match and os.path.isdir(os.path.join(directory_path, entry)):
            run_number = int(match.group(1))
            max_run_number = max(max_run_number, run_number)
    return max_run_number

def read_zarr(file_path):
    """
    Open a Zarr file in read mode.

    Args:
        file_path (str): The path to the Zarr file.

    Returns:
        zarr.Array or zarr.Group: The opened Zarr object.
    """
    return zarr.open(file_path, mode='r')


def _extract_zyx(point, source_axes=None):
    """
    Extract Z, Y, X from a point using optional source axis metadata.

    Preferred behavior:
      1. If source_axes is provided and can be aligned to the point dimensionality,
         use it to map coordinates to z,y,x.
      2. Otherwise, fall back to legacy heuristics:
         - len >= 5 => [t, c, z, y, x]
         - else      => last three coords as [z, y, x]
    """
    coords = list(point)
    if len(coords) < 3:
        raise ValueError("Point must have at least 3 coordinates.")

    if source_axes is not None:
        axis_names = [str(axis).strip().lower() for axis in source_axes]
        axis_map = {axis_name: idx for idx, axis_name in enumerate(axis_names)}
        if {"z", "y", "x"}.issubset(axis_map):
            # Common case: point ndim matches source ndim.
            if len(coords) == len(axis_names):
                z = coords[axis_map["z"]]
                y = coords[axis_map["y"]]
                x = coords[axis_map["x"]]
                return float(z), float(y), float(x)

            # Fallback for points carrying only spatial dimensions.
            if len(coords) == 3:
                spatial_order = [name for name in axis_names if name in {"z", "y", "x"}]
                if len(spatial_order) == 3:
                    spatial_map = {axis_name: idx for idx, axis_name in enumerate(spatial_order)}
                    z = coords[spatial_map["z"]]
                    y = coords[spatial_map["y"]]
                    x = coords[spatial_map["x"]]
                    return float(z), float(y), float(x)

    if len(coords) >= 5:
        z, y, x = coords[2], coords[3], coords[4]
    else:
        z, y, x = coords[-3], coords[-2], coords[-1]
    return float(z), float(y), float(x)


def normalize_points_to_zyx(points, source_axes=None):
    """
    Normalize an iterable of points to [z, y, x] format.
    Accepts both [z, y, x] and [t, c, z, y, x].
    If source_axes is provided, axis-aware normalization is applied first.
    """
    normalized = []
    for idx, point in enumerate(points):
        try:
            z, y, x = _extract_zyx(point, source_axes=source_axes)
        except Exception as exc:
            raise ValueError(
                f"Invalid point at index {idx}. Expected [z,y,x] or [t,c,z,y,x]. "
                f"source_axes={source_axes}"
            ) from exc
        normalized.append([z, y, x])
    return normalized


def load_keypoints(file_path):
    """
    Load keypoints from a JSON file and convert them into a list of curve nodes.

    The JSON file is expected to contain a key 'points' mapping to a list of points.
    Each point is assumed to be a list-like object where the third element represents
    the first coordinate, the second element represents the second coordinate, and the
    first element represents the third coordinate. All coordinates are converted to float.

    Args:
        file_path (str): Path to the JSON file containing keypoints.

    Returns:
        list: A list of curve nodes, where each node is a list of three floats [ii, j, k].
    """
    with open(file_path, "r") as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        path = payload.get("points", [])
        point_axes = payload.get("point_axes")
    else:
        path = payload
        point_axes = None
    point_sub = normalize_points_to_zyx(path, source_axes=point_axes)
    curvenode = []
    for i in point_sub:
        ii = float(i[2])
        j = float(i[1])
        k = float(i[0])
        ijkPoint = [ii, j, k]
        curvenode.append(ijkPoint)
    return curvenode

def write_status(trace,
    status: str,
    error_msg: str = None
):
    """
    Write the status of a job to a JSON file.

    This function creates a dictionary containing information about a job's current
    state, including the job name, status, iteration count, total iterations, any error
    message, and a timestamp. The dictionary is then written to a JSON file specified by
    'status_path'. If the file already exists, it will be overwritten.

    Args:
        status_path (str): The file path where the status JSON will be saved.
        job_name (str): The name of the job.
        status (str): The current status of the job (e.g., "running", "done").
        iteration (int, optional): The current iteration number. Defaults to 0.
        total_iterations (int, optional): The total number of iterations. Defaults to 0.
        error_msg (str, optional): An error message to include if applicable. Defaults to None.
    """
    data = {
        "job_name": trace.next_run_folder,
        "status": status,
        "iteration": trace.cummulative_iterator,
        "total_iterations": trace.iterations,
        "error_msg": error_msg,
        "diameter":trace.diameter,
        "model": trace.model_name, 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write to disk (overwrites any existing file).
    with open(trace.status_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_curve_nodes(trace, reset=False, finished=False):
    """
    Save curve nodes to a JSON file. If the file already exists, the points
    can either be reset or appended based on the reset flag.

    Parameters:
    curve_node (list of lists): List of curve nodes, where each node is a list
                                [k, j, i] with coordinates.
    filename (str): The base name of the file where the nodes will be saved. 
                    A '.json' extension will be added automatically.
    reset (bool): If True, existing points in the file will be reset.
                  If False, new points will be appended to existing points.
                  Default is False.

    Returns:
    None
    """
    filename = trace.result_trace_path + '.json'
    trace.log.info(f"Saving curve nodes to file: {filename}")
    
    save_nodes = []
    scale_zyx = [float(v) for v in getattr(trace, "run_level_scale_zyx", DEFAULT_SCALE_ZYX)]
    sz, sy, sx = scale_zyx

    # Internal trace.curvenode stores xyz in the selected run level.
    # Save points in canonical level-0 zyx coordinates.
    for node in trace.curvenode:
        x_coord = float(node[0]) * sx
        y_coord = float(node[1]) * sy
        z_coord = float(node[2]) * sz
        zyx_point = [z_coord, y_coord, x_coord]
        save_nodes.append(zyx_point)
    
    if os.path.isfile(filename):
        trace.log.info(f"File {filename} exists. {'Resetting' if reset else 'Appending'} points.")
        with open(filename, 'w') as f:
            if reset:
                # If reset is True, save only the new nodes
                json_data = {'points': save_nodes, 'point_axes': ['z', 'y', 'x'], 'Complete':finished}
                trace.log.debug(f"Resetting file with new nodes: {save_nodes}")
                json.dump(json_data, f)
            else:
                # If reset is False, append new nodes to existing points
                with open(filename, 'r') as fr:
                    prev_points = json.load(fr)['points']
                # Stack the existing points with the new ones, excluding the last 4 of the previous set
                nodes = np.vstack([prev_points[:-4], save_nodes])
                trace.log.debug(f"Appending new nodes to existing points. Total nodes: {len(nodes)}")
                json_data = {'points': nodes.tolist(), 'point_axes': ['z', 'y', 'x'], 'Complete':finished}  # Convert numpy array to list before saving
                json.dump(json_data, f)
    else:
        # If the file doesn't exist, create a new file with the new nodes
        trace.log.info(f"File {filename} does not exist. Creating a new file.")
        with open(filename, 'w') as f:
            json_data = {'points': save_nodes, 'point_axes': ['z', 'y', 'x'], 'Complete':finished}
            trace.log.debug(f"Saving new nodes: {save_nodes}")
            json.dump(json_data, f)
    trace.record_current_node_params()
    trace.write_detailed_parameters()
    # write_ply(filename, vertices, faces)
    if trace.write_ply:
        write_ply(trace)
    trace.log.info(f"Finished saving curve nodes to {filename}.")


def save_curve_nodes_gt(points, filename, reset=False, error="", scale_zyx=None):
    """
    Save curve nodes to a JSON file. If the file already exists, the points
    can either be reset or appended based on the reset flag.

    Parameters:
    curve_node (list of lists): List of curve nodes, where each node is a list
                                [k, j, i] with coordinates.
    filename (str): The base name of the file where the nodes will be saved. 
                    A '.json' extension will be added automatically.
    reset (bool): If True, existing points in the file will be reset.
                  If False, new points will be appended to existing points.
                  Default is False.

    Returns:
    None
    """
    filename = filename + '.json'
    
    save_nodes = []
    if scale_zyx is None:
        scale_zyx = DEFAULT_SCALE_ZYX
    sz, sy, sx = [float(v) for v in scale_zyx]

    # Input points are xyz in run-level coordinates.
    # Save in canonical level-0 zyx coordinates.
    for node in points:
        x_coord = float(node[0]) * sx
        y_coord = float(node[1]) * sy
        z_coord = float(node[2]) * sz
        zyx_point = [z_coord, y_coord, x_coord]
        save_nodes.append(zyx_point)
    
    if os.path.isfile(filename):
        with open(filename, 'w') as f:
            if reset:
                # If reset is True, save only the new nodes
                json_data = {'points': save_nodes, 'point_axes': ['z', 'y', 'x'], 'Error':error}
                json.dump(json_data, f)
            else:
                # If reset is False, append new nodes to existing points
                with open(filename, 'r') as fr:
                    prev_points = json.load(fr)['points']
                # Stack the existing points with the new ones, excluding the last 4 of the previous set
                nodes = np.vstack([prev_points[:-4], save_nodes])
                json_data = {'points': nodes.tolist(), 'point_axes': ['z', 'y', 'x'], 'Error':error}  # Convert numpy array to list before saving
                json.dump(json_data, f)
    else:
        with open(filename, 'w') as f:
            json_data = {'points': save_nodes, 'point_axes': ['z', 'y', 'x'], 'Error':error}
            json.dump(json_data, f)
            


def add_mask_edge_loop(trace):
    """
    Compute the edge loop for a given mask and add it to the global lists.
    
    Parameters:
        traceparameters (dict): Contains settings such as 'dim'.
        mask (np.ndarray): The segmentation mask.
        transform (SimpleITK.Transform, optional): Transformation to world coordinates.
    """
    
    # Get the edge loop (ordered list of vertices) from the mask.
    edge_loop_vertices = generate_edge_loop(trace)
    if not edge_loop_vertices:
        return  # No contour found.
    
    # Record the current starting index in the global vertex list.
    start_index = len(trace.ply_vertices)
    num_new_vertices = len(edge_loop_vertices)
    
    # Append these vertices to the global list.
    trace.ply_vertices.extend(edge_loop_vertices)
    
    # Create a face using the indices of these new vertices.
    face_indices = list(range(start_index, start_index + num_new_vertices))
    trace.ply_faces.append(face_indices)

def generate_edge_loop(trace):
    """
    Extract the perimeter (edge loop) from a segmentation mask.
    
    Parameters:
        traceparameters (dict): Parameters including dimension info.
        mask (np.ndarray): 2D segmentation mask.
        transform (SimpleITK.Transform, optional): Transform from pixel to world space.
    
    Returns:
        List of vertices (each a list [x, y, z]) in order.
    """
    # Convert the mask to uint8; assume the foreground is defined by the central pixel value.
    central_value = trace.current_mask[int(trace.dim/2), int(trace.dim/2)]
    mask_uint8 = (trace.current_mask == central_value).astype(np.uint8) * 255

    # Find contours; use RETR_EXTERNAL to get the outer boundary.
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Choose the largest contour by area.
    contour = max(contours, key=cv2.contourArea)
    
    # Convert the contour (Nx1x2 array) into an ordered list of vertices.
    vertices = []
    for point in contour:
        x, y = point[0]
        pt = [float(x), float(y), 0.0]  # z=0 for 2D slice; adjust if needed.
        if trace.current_slice_transform is not None:
            pt = list(trace.current_slice_transform.TransformPoint(pt))
        vertices.append(pt)
        
    return vertices

def write_ply(trace):
    """
    Write a PLY file with multiple faces.
    
    Parameters:
        filename (str): Output filename.
        vertices (list of lists): List of vertices (each [x, y, z]).
        faces (list of lists): List of faces, where each face is a list of vertex indices.
    """
    
    job_name = str(trace.name)
    job_stem = job_name[:-5] if job_name.endswith(".json") else job_name
    filename = os.path.join(trace.next_run_folder, f"{job_stem}_cloud.ply")

    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(trace.ply_vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face {}\n".format(len(trace.ply_faces)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertex list
        for v in trace.ply_vertices:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(*v))
        
        # Write faces (each face starts with a count of vertices, then the indices)
        for face in trace.ply_faces:
            face_line = "{} {}".format(len(face), " ".join(str(idx) for idx in face))
            f.write(face_line + "\n")
