import SimpleITK as sitk
import numpy as np


def generate_center_transform(trace):
    """
    Generate an affine transform to center the XY slice.

    This function creates a SimpleITK affine transform that translates the image so that
    its center (based on the 'dim' parameter) is moved to the origin. It uses an identity
    matrix for the transformation and applies a translation computed from half of the
    'dim' value.

    Args:
        trace_parameters (dict): A dictionary containing the key 'dim' which specifies
            the dimension of the image slice.

    Returns:
        sitk.AffineTransform: The generated affine transform.
    """
    XYToSlice_matrix = [1, 0, 0,
                 0, 1, 0,
                 0, 0, 1]
    XYToSlice_transform = sitk.AffineTransform(3)
    XYToSlice_transform.SetMatrix(XYToSlice_matrix)
    XYToSlice_transform.Translate((-int(trace.dim/2), -int(trace.dim/2), 0))
    return XYToSlice_transform

def get_point_curve_ras(idx, curveNode):
    """
    Convert a curve point to a column vector.

    Args:
        idx (int): Index of the point in curveNode.
        curveNode (list): List of points.

    Returns:
        np.ndarray: A column vector of the point.
    """
    pt = np.expand_dims(np.array(curveNode[idx]), 1)
    return pt

def load_image_gt(trace, lw_bnds, up_bnds):
    """Load image gt."""
    lw_bnds = lw_bnds.astype('int')
    up_bnds = up_bnds.astype('int')
    im=trace.volume[lw_bnds[2]:up_bnds[2], lw_bnds[1]:up_bnds[1], lw_bnds[0]:up_bnds[0]].astype('uint16')
    print('H1')
    image = sitk.GetImageFromArray(im)
    print('H2')
    image.SetOrigin([float(i) for i in lw_bnds])
    print('H3')
    # im = trace.volume()

def load_image(trace, idx = None):
    """
    Load an image slice from the 3D volume based on a given point index.

    This function extracts a point from trace.curvenode, determines the bounding
    indices using check_chunk_size, and returns a SimpleITK image from the corresponding
    sub-volume of trace.volume. The image origin is set using the lower bounds.

    Args:
        pointIndex (int): Index of the point in trace.curvenode.
        trace: An object with attributes 'curvenode', 'volume', and 'chunck_size'.

    Returns:
        SimpleITK.Image: The image slice extracted from the volume.
    """
    if idx is not None:
        points = get_point_curve_ras(idx, trace.curvenode).ravel()
    else:
        points = get_point_curve_ras(trace.pointIndex, trace.curvenode).ravel()
    points_int = points.astype('int')
    lw_bnds, up_bnds = check_chunk_size(trace.volume, trace.chunk_size, points_int)
    im=trace.volume[lw_bnds[2]:up_bnds[2], lw_bnds[1]:up_bnds[1], lw_bnds[0]:up_bnds[0]].astype('uint16')
    image = sitk.GetImageFromArray(im)
    image.SetOrigin([float(i) for i in lw_bnds])
    return image

def check_chunk_size(stack, chunck_size, points_int):
    """Check chunk size."""
    pts_reorient =points_int 
    print('STACK', stack)
    print('STACK SHAPE', stack.shape)
    stack_size = [stack.shape[2], stack.shape[1], stack.shape[0]]
    lw_bnds = [i-chunck_size for i in pts_reorient]
    up_bnds = [i+chunck_size for i in pts_reorient]
    for idx, v in enumerate(lw_bnds):
        if v<0:
            lw_bnds[idx] = 0
    
    for idx, v in enumerate(up_bnds):
        if v>=stack_size[idx]:
            up_bnds[idx] = stack_size[idx]-1
    return lw_bnds, up_bnds

def set_slice_view(trace, vector = None, points=None):
    """
    Set up the slice view transformation for a given curve point.

    This function computes an affine transformation (slice_T) that aligns the image
    slice based on the provided direction vector (V_last) and the current point
    coordinates. It calculates a normalized normal vector from V_last, determines a
    translation direction by computing the cross product with the z-axis, and then
    creates an initial transform using SetSliceToRASByNTP. This transform is then combined
    with a pre-defined XY-to-slice transform from trace_parameters to yield the final
    composite transform.

    Args:
        curveNode (list or array-like): The list of points defining the curve.
        trace_parameters (dict): Dictionary containing tracking parameters; must include
            'XYToSlice_transform'.
        V_last (np.ndarray): The last computed direction vector.
        pointIndex (int): The index of the current point in the curve.
        points (np.ndarray, optional): Coordinates for the current point. If None, the
            coordinates are obtained using get_point_curve_ras().

    Returns:
        tuple: A tuple (slice_T, points) where slice_T is the composite SimpleITK transform,
            and points is the coordinate array of the current point.
    """
    if vector is None: # this would have be updated in the intial step that found the mask
        vector = trace.vectors[-1]
    normal_v = vector.ravel()
    normal_v = normal_v/np.linalg.norm(normal_v)
    trans_dir = np.cross([0.0, 0.0, 1], normal_v)
    trans_dir = trans_dir/np.linalg.norm(trans_dir)
    # get the coordinates for the current point index
    if points is None: # this would have be updated in the intial step that found the mask -- maybe?
        points = get_point_curve_ras(trace.pointIndex, trace.curvenode) ### THIS NEEDS TO BE SOMETHING ELSE
    slice_T = SetSliceToRASByNTP(normal_v[0],
                        normal_v[1],
                        normal_v[2],
                        trans_dir[0],
                        trans_dir[1],
                        trans_dir[2],
                        points[0,0],
                        points[1,0],
                        points[2,0],
                        0)
    trace.current_slice_transform = sitk.CompositeTransform([slice_T, trace.center_transform])
    
def SetSliceToRASByNTP(Nx, Ny, Nz, Tx, Ty, Tz, Px, Py, Pz, orientation = 0):
    """
    Create an affine transform from NTP (Normal, Tangent, Point) parameters to RAS coordinates.

    This function constructs a SimpleITK affine transform based on the provided normal vector (Nx, Ny, Nz)
    and tangent vector (Tx, Ty, Tz) components. It computes an orthonormal basis using cross products and
    then uses the given point (Px, Py, Pz) to set the origin of the transform. The resulting transform
    maps coordinates to the RAS (Right-Anterior-Superior) space. The optional orientation parameter is
    reserved for future adjustments and is not used in the current computation.

    Args:
        Nx (float): Normal vector component along the x-axis.
        Ny (float): Normal vector component along the y-axis.
        Nz (float): Normal vector component along the z-axis.
        Tx (float): Tangent vector component along the x-axis.
        Ty (float): Tangent vector component along the y-axis.
        Tz (float): Tangent vector component along the z-axis.
        Px (float): The x-coordinate of the point.
        Py (float): The y-coordinate of the point.
        Pz (float): The z-coordinate of the point.
        orientation (float, optional): Orientation adjustment (currently unused). Defaults to 0.

    Returns:
        sitk.AffineTransform: The affine transform mapping to RAS coordinates.
    """
    n = np.zeros((3,))
    t = np.zeros((3,))
    n[0] = Nx
    n[1] = Ny
    n[2] = Nz
    t[0] = Tx
    t[1] = Ty
    t[2] = Tz
    c = np.cross(n, t)
    t = np.cross(c, n)
    n = n/np.linalg.norm(n)
    t = t/np.linalg.norm(t)
    c = c/np.linalg.norm(c)
    matrix_T = [t[0], c[0], n[0],
                t[1], c[1], n[1],
                t[2], c[2], n[2]]
    slice_T = sitk.AffineTransform(3)
    slice_T.SetMatrix(matrix_T)
    
    slice_T.Translate((Px, Py, Pz))
    return slice_T

# def get_frame(trace):
#     resampled_image = sitk.Resample(trace.current_chunk, trace.reference_image,
#                                     trace.current_slice_transform,
#                                     trace.interpolator, trace.dim, sitk.sitkUInt16)
#     resampled_image = sitk.GetArrayFromImage(resampled_image)[0,:,:]
#     trace.current_raw = resampled_image

def get_frame(trace):
    """Get frame."""
    trace.resampler.SetTransform(trace.current_slice_transform)
    resampled = trace.resampler.Execute(trace.current_chunk)
    trace.current_raw = sitk.GetArrayFromImage(resampled)[0]


def set_slice_view_ut(trace, points=None):
    """Set slice view ut."""
    normal_v = trace.vectors[-1].ravel()
    normal_v = normal_v/np.linalg.norm(normal_v)
    trans_dir = np.cross([0.0, 0.0, 1], normal_v)
    trans_dir = trans_dir/np.linalg.norm(trans_dir)
    # get the coordinates for the current point index
    if points is None:
        points = get_point_curve_ras(trace.pointIndex, trace.curvenode)
    # Get the direction cosines and point P
    direction_cosines, P = SetSliceToRASByNTP_ut(normal_v[0],
                                              normal_v[1],
                                              normal_v[2],
                                              trans_dir[0],
                                              trans_dir[1],
                                              trans_dir[2],
                                              points[0,0],
                                              points[1,0],
                                              points[2,0],
                                              0)
    return direction_cosines, P, points

def SetSliceToRASByNTP_ut(Nx, Ny, Nz, Tx, Ty, Tz, Px, Py, Pz, orientation=0):
    """Compute SetSliceToRASByNTP ut."""
    n = np.array([Nx, Ny, Nz])
    t = np.array([Tx, Ty, Tz])
    c = np.cross(n, t)
    t = np.cross(c, n)
    n = n / np.linalg.norm(n)
    t = t / np.linalg.norm(t)
    c = c / np.linalg.norm(c)
    # Direction cosines
    direction_cosines = [t[0], c[0], n[0],
                         t[1], c[1], n[1],
                         t[2], c[2], n[2]]
    P = np.array([Px, Py, Pz])
    return direction_cosines, P

def get_volume(P, direction_cosines, image, size=[150, 150, 50], spacing=[1.0, 1.0, 1.0]):
    # Create the reference image
    """Get volume."""
    reference_image = sitk.Image(size, image.GetPixelID())
    reference_image.SetSpacing(spacing)
    reference_image.SetDirection(direction_cosines)

    # Compute the offsets along t, c, n axes
    size_np = np.array(size, dtype=float)
    spacing_np = np.array(spacing, dtype=float)
    
    # Offsets to center the volume in t and c directions, and start at P along n
    offset_t = - (size_np[0] - 1) / 2.0 * spacing_np[0]
    offset_c = - (size_np[1] - 1) / 2.0 * spacing_np[1]
    offset_n = 0.0  # Start at P along n direction

    # Form the offset vector in the volume's coordinate system
    offset_volume = np.array([offset_t, offset_c, offset_n])

    # Transform the offset into physical space
    R = np.array(direction_cosines).reshape(3,3)
    offset_physical = R @ offset_volume

    # Compute the origin so that P is at one face of the volume
    origin = np.array(P).flatten() + offset_physical
    reference_image.SetOrigin(origin.tolist())

    # Use identity transform
    transform = sitk.Transform(3, sitk.sitkIdentity)

    # Resample the image
    resampled_image = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
    resampled_array = sitk.GetArrayFromImage(resampled_image)

    return resampled_image, resampled_array
