import numpy as np
from scipy.interpolate import splev, splprep
from tubulemap.cellpose_tracker.geometry import *

def direction_vector(trace, pointIndex):
    """Compute direction vector."""
    c_pt = np.expand_dims(np.array(trace.curvenode[pointIndex]), 1)
    p_pt = np.expand_dims(np.array(trace.curvenode[pointIndex-1]), 1)
    dirr = c_pt-p_pt
    Vt = dirr/(np.sqrt(np.sum(dirr**2))) 
    return Vt, c_pt

def ijk2ras(x, y, slice_T):
    """Compute ijk2ras."""
    point_ijk = [y, x,  0]
    point_VolumeRas = slice_T.TransformPoint(point_ijk)
    return point_VolumeRas

def new_vector(trace):
    # logging.info('Predicting new vector and next point')
    """Compute new vector."""
    c_pt = np.expand_dims(np.array(trace.curvenode[trace.pointIndex]), 1)
    if trace.vector_method == 'traditional':
        # logging.info('Using traditional method to estimate vector')
        Vt, _ = direction_vector(trace, trace.pointIndex)
        Vt_prev, _ = direction_vector(trace, trace.pointIndex-1)
        V_new = trace.w*Vt+(1-trace.w)*Vt_prev #linear comb of the last dirr vector and the new dirr vector    
    
    if trace.vector_method == 'spline':
        # logging.info('Using spline method to estimate vector')
        points = np.empty(shape=(1, 3))
        len_c = len(trace.curvenode)
        if len(np.unique(trace.curvenode, axis=0)) == len(trace.curvenode):
            for a in range(len_c-1,0,-1):
                point = get_point_curve_ras(len_c-a, trace.curvenode).reshape(1,3)
                points = np.append(points, point, axis=0)
            tck, u = splprep([points[:,0], points[:,1],  points[:,2]], s = np.sqrt(len_c))
            Vspline = splev(u[-1],tck,der=1)
            V_new = Vspline / np.linalg.norm(Vspline)
        V_new =  np.expand_dims(V_new, -1) 
        
    if trace.overwite_w_rot and trace.vector_method == 'traditional' and trace.vectors[-1] is not None:
        # logging.info('Overwrite vector with rotation direction')
        V_new = trace.vectors[-1]
    trace.log.critical('Trace parameter stepsize  = '  + str(trace.stepsize))
    n_pt = c_pt + trace.stepsize*V_new  # set new point in space based on a vector direction
    return n_pt, V_new
