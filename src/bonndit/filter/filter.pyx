#%%cython --annotate
#cython: language_level=3, boundscheck=False

import numpy as np
import cython
from tqdm import tqdm
from libc.math cimport exp

@cython.cdivision(True)
cdef void intersection_finder(double[:,:,:] hast_dict, double[:]x, double[:] y) except *:

    """
    This function takes to points x and y and hash_dict. It adds one to all visted grid points.

    Parameters
    ----------
    hast_dict
    x
    y
    visited

    Returns
    -------

    """
    cdef float dx = abs(x[0] - y[0]),  dy = abs(x[1] - y[1]), dz = abs(x[2] - y[2])
    cdef int x0  = int(x[0]), y0 = int(x[1]), z0 =  int(x[2])
    cdef float dt_dx = 1/dx, dt_dy = 1/dy,  dt_dz = 1/dz
    cdef int n = 1, j,k,l,i
    cdef int x_inc, y_inc, z_inc
    cdef float t_next_x, t_next_y, t_next_z
    if dx == 0:
        x_inc = 0
        t_next_x = dt_dx
    elif y[0] > x[0]:
        x_inc = 1
        n += int(y[0] - x0)
        t_next_x = (x0 + 1 - x[0]) * dt_dx
    else:
        x_inc = -1
        n += int(x0 - y[0])
        t_next_x = (x[0] - x0) * dt_dx
    #print(n)
    if dy == 0:
        y_inc = 0
        t_next_y = dt_dy
    elif y[1] > x[1]:
        y_inc = 1
        n += int(y[1] - y0)
        t_next_y = (y0 + 1 - x[1]) * dt_dy
    else:
        y_inc = -1
        n += int(y0 - y[1])
        t_next_y = (x[1] - y0) * dt_dy
    #print(n)
    if dz == 0:
        z_inc = 0
        t_next_z = dt_dy
    elif y[2] > x[2]:
        z_inc = 1
        n += int(y[2] - z0)
        t_next_z = (z0 + 1 - x[2]) * dt_dz
    else:
        z_inc = -1
        n += int(z0 - y[2])
        t_next_z = (x[2] - z0) * dt_dz
    #print(n)
    for i in range(n):
        # to world
        for j in range(-1,2):
            for k in range(-1,2):
                for l in range(-1,2):
                    hast_dict[x0 + j, y0 + k, z0 + l] +=exp(-(j**2+k**2+l**2))
        #print(x0, y0, z0)
        if t_next_x <= t_next_y and t_next_x <= t_next_z:
            x0 += x_inc
            t_next_x += dt_dx
        elif t_next_y <= t_next_z and t_next_y <= t_next_x:
            y0 += y_inc
            t_next_y += dt_dy
        elif t_next_z <= t_next_y and t_next_z <= t_next_x:
            z0 += z_inc
            t_next_z += dt_dz


cpdef intersection_dict(double[:,:,:] hash_dict, streamlines_test, verbose):
    """
    Create intersection dict. coordinates are set as keys of dict and value is the intesection count of streamline
    with grid
    """
    for streamline in tqdm(streamlines_test, disable=not verbose):
        for i in range(1, streamline.shape[0]):
            if np.all(~np.isnan(streamline[i - 1:i+1])):
                intersection_finder(hash_dict, np.array(streamline[i - 1], dtype=np.float64),np.array(streamline[i], dtype=np.float64))
    return hash_dict



cpdef plysplitter(vertices, endindex, verbose):
    """
    split plydata into streamlines
    """
    streamlines = []
    endindex = np.insert(endindex, 0, 0)
    for i in tqdm(range(1, endindex.shape[0]), disable=not verbose):
        streamlines.append(vertices[int(endindex[i - 1]): int(endindex[i])])
    return streamlines

cdef filter_s(streamline, exclusion):
    """
    Filter along hemisphere
    """
    for task in exclusion:
        if task[0] == 'x':
            if task[1] == '<' and streamline[0] < task[2]:
                return True
            elif task[1] == '>' and streamline[0] > task[2]:
                return True
        if task[0] == 'y':
            if task[1] == '<' and streamline[1] < task[2]:
                return True
            elif task[1] == '>' and streamline[1] > task[2]:
                return True
        if task[0] == 'z':
            if task[1] == '<' and streamline[2] < task[2]:
                return True
            elif task[1] == '>' and streamline[2] > task[2]:
                return True

cdef filter_q(streamline, exclusion):
    """
    Exclude everything which intersects with a cube
    """
    for task in exclusion:
        if task[0][0] > streamline[0] or streamline[0] > task[0][-1]:
            continue
        if task[1][0] > streamline[1] or streamline[1] > task[1][-1]:
            continue
        if task[2][0] > streamline[2] or streamline[2] > task[2][-1]:
            continue
        return True
    return False


cpdef streamline_filter(streamlines, exclusionq, exclusion, trafo, verbose):
    """
    Filter out all streamlines which do not fullfill the filter rules
    """
    delete = []
    for i in tqdm(range(len(streamlines)), disable=not verbose):
        for j in range(streamlines[i].shape[0]):
            streamlines[i][j][:-1] = trafo.wtoi_p(np.array(streamlines[i][j][:-1], dtype=np.float64))
            if filter_q(streamlines[i][j][:-1], exclusionq) or filter_s(streamlines[i][j][:-1],
                                                                                 exclusion) or np.any(np.isnan(streamlines[i][j][:-1])):
                delete.append(False)
                break
        else:
            delete.append(True)
    return delete

cpdef streamline_itow(streamlines_test, trafo):
    cdef int i, j
    for i in range(len(streamlines_test)):
        for j in range(streamlines_test[i].shape[0]):
            streamlines_test[i][j][:-1] = trafo.itow_p(np.array(streamlines_test[i][j][:-1], dtype=np.float64))
    return streamlines_test

cpdef filter_mask(streamlines_test, mask, verbose):
    """
    Cut of the streamline at the first intersection with a low density area.
    """
    streamlines = []
    for stream in tqdm(streamlines_test, disable=not verbose):
        for i in range(stream.shape[0]):
            if stream[i][3] == 1:
                for j, point in enumerate(stream[i:]):
                    point = np.nan_to_num(point)
                    if int(point[0]) >= mask.shape[0] or int(point[1]) >= mask.shape[1] or int(point[2]) \
                            >=mask.shape[2]:
                        forward = j
                        break
                    if mask[int(point[0]), int(point[1]), int(point[2])] == 0:
                        forward = j
                        break
                else:
                    forward = j

                for j, point in enumerate(stream[i::-1]):
                    point = np.nan_to_num(point)
                    if int(point[0]) >= mask.shape[0] or int(point[1]) >= mask.shape[1] or int(point[2]) \
                            >=mask.shape[2]:
                        backward = j
                        break
                    if mask[int(point[0]), int(point[1]), int(point[2])] == 0:
                        backward = j
                        break
                else:
                    backward = j

                streamlines.append(stream[i - backward: i + forward])
                break
        else:
            print("boo")

    return streamlines
