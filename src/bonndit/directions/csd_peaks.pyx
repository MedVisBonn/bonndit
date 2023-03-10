#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
import numpy as np
cimport numpy as cnp
from tqdm import tqdm
from libc.math cimport fabs, cos, pi
from bonndit.utilc.hota cimport hota_8o3d_sym_eval_cons as hota_8o3d_sym_eval
from bonndit.utilc.lowrank cimport refine_rank1_3d
cimport cython
import os
dirname = os.path.dirname(__file__)

@cython.boundscheck(False)
@cython.wraparound(False)
def search_descending(cython.floating[::1] a, double relative_threshold):
    """`i` in descending array `a` so `a[i] < a[0] * relative_threshold`
    Call ``T = a[0] * relative_threshold``. Return value `i` will be the
    smallest index in the descending array `a` such that ``a[i] < T``.
    Equivalently, `i` will be the largest index such that ``all(a[:i] >= T)``.
    If all values in `a` are >= T, return the length of array `a`.
    Parameters
    ----------
    a : ndarray, ndim=1, c-contiguous
        Array to be searched.  We assume `a` is in descending order.
    relative_threshold : float
        Applied threshold will be ``T`` with ``T = a[0] * relative_threshold``.
    Returns
    -------
    i : np.intp
        If ``T = a[0] * relative_threshold`` then `i` will be the largest index
        such that ``all(a[:i] >= T)``.  If all values in `a` are >= T then
        `i` will be `len(a)`.
    Examples
    --------
    >>> a = np.arange(10, 0, -1, dtype=float)
    >>> a
    array([ 10.,   9.,   8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])
    >>> search_descending(a, 0.5)
    6
    >>> a < 10 * 0.5
    array([False, False, False, False, False, False,  True,  True,  True,  True], dtype=bool)
    >>> search_descending(a, 1)
    1
    >>> search_descending(a, 2)
    0
    >>> search_descending(a, 0)
    10
    """
    if a.shape[0] == 0:
        return 0

    cdef:
        cnp.npy_intp left = 0
        cnp.npy_intp right = a.shape[0]
        cnp.npy_intp mid
        double threshold = relative_threshold * a[0]

    while left != right:
        mid = (left + right) // 2
        if a[mid] >= threshold:
            left = mid + 1
        else:
            right = mid
    return left


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(True)
cdef local_maxima(double[:] odf, cnp.uint16_t[:, :] edges):
    """Local maxima of a function evaluated on a discrete set of points.
    If a function is evaluated on some set of points where each pair of
    neighboring points is an edge in edges, find the local maxima.
    Parameters
    ----------
    odf : array, 1d, dtype=double
        The function evaluated on a set of discrete points.
    edges : array (N, 2)
        The set of neighbor relations between the points. Every edge, ie
        `edges[i, :]`, is a pair of neighboring points.
    Returns
    -------
    peak_values : ndarray
        Value of odf at a maximum point. Peak values is sorted in descending
        order.
    peak_indices : ndarray
        Indices of maximum points. Sorted in the same order as `peak_values` so
        `odf[peak_indices[i]] == peak_values[i]`.
    Notes
    -----
    A point is a local maximum if it is > at least one neighbor and >= all
    neighbors. If no points meet the above criteria, 1 maximum is returned such
    that `odf[maximum] == max(odf)`.
    See Also
    --------
    dipy.core.sphere
    """
    cdef cnp.ndarray[cnp.npy_intp] wpeak = np.zeros((odf.shape[0],), dtype=np.intp)
    count = _compare_neighbors(odf, edges, &wpeak[0])
    if count == -1:
        raise IndexError("Values in edges must be < len(odf)")
    elif count == -2:
        raise ValueError("odf can not have nans")
    indices = wpeak[:count].copy()
    # Get peak values return
    values = np.take(odf, indices)
    # Sort both values and indices
    _cosort(values, indices)
    return values, indices


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cosort(double[::1] A, cnp.npy_intp[::1] B) nogil:
    """Sorts `A` in-place and applies the same reordering to `B`"""
    cdef:
        cnp.npy_intp n = A.shape[0]
        cnp.npy_intp hole
        double insert_A
        long insert_B

    for i in range(1, n):
        insert_A = A[i]
        insert_B = B[i]
        hole = i
        while hole > 0 and insert_A > A[hole -1]:
            A[hole] = A[hole - 1]
            B[hole] = B[hole - 1]
            hole -= 1
        A[hole] = insert_A
        B[hole] = insert_B


@cython.wraparound(False)
@cython.boundscheck(False)
cdef long _compare_neighbors(double[:] odf, cnp.uint16_t[:, :] edges,
                             cnp.npy_intp *wpeak_ptr) nogil:
    """Compares every pair of points in edges
    Parameters
    ----------
    odf : array of double
        values of points on sphere.
    edges : array of uint16
        neighbor relationships on sphere. Every set of neighbors on the sphere
        should be an edge.
    wpeak_ptr : pointer
        pointer to a block of memory which will be updated with the result of
        the comparisons. This block of memory must be large enough to hold
        len(odf) longs. The first `count` elements of wpeak will be updated
        with the indices of the peaks.
    Returns
    -------
    count : long
        Number of maxima in odf. A value < 0 indicates an error:
            -1 : value in edges too large, >= than len(odf)
            -2 : odf contains nans
    """
    cdef:
        cnp.npy_intp lenedges = edges.shape[0]
        cnp.npy_intp lenodf = odf.shape[0]
        cnp.npy_intp i
        cnp.uint16_t find0, find1
        double odf0, odf1
        long count = 0

    for i in range(lenedges):

        find0 = edges[i, 0]
        find1 = edges[i, 1]
        if find0 >= lenodf or find1 >= lenodf:
            count = -1
            break
        odf0 = odf[find0]
        odf1 = odf[find1]

        """
        Here `wpeak_ptr` is used as an indicator array that can take one of
        three values.  If `wpeak_ptr[i]` is:
        * -1 : point i of the sphere is smaller than at least one neighbor.
        *  0 : point i is equal to all its neighbors.
        *  1 : point i is > at least one neighbor and >= all its neighbors.
        Each iteration of the loop is a comparison between neighboring points
        (the two point of an edge). At each iteration we update wpeak_ptr in the
        following way::
            wpeak_ptr[smaller_point] = -1
            if wpeak_ptr[larger_point] == 0:
                wpeak_ptr[larger_point] = 1
        If the two points are equal, wpeak is left unchanged.
        """
        if odf0 < odf1:
            wpeak_ptr[find0] = -1
            wpeak_ptr[find1] |= 1
        elif odf0 > odf1:
            wpeak_ptr[find0] |= 1
            wpeak_ptr[find1] = -1
        elif (odf0 != odf0) or (odf1 != odf1):
            count = -2
            break

    if count < 0:
        return count

    # Count the number of peaks and use first count elements of wpeak_ptr to
    # hold indices of those peaks
    for i in range(lenodf):
        if wpeak_ptr[i] > 0:
            wpeak_ptr[count] = i
            count += 1

    return count

cdef maxima_finder(double[:] ls, double[:,:] vs, tens, ten_eval, edges, vertices, der, testv, iso, aniso,
                   relative_peak_threshold, min_separation_angle, int max_num):
    #find local maxima:
    ls_heap, indices = local_maxima(ten_eval, edges)
    # If there is only one peak return
    n = len(ls_heap)
    if n == 0 or ls_heap[0] < 0:
        return
    elif n == 1:
        ls[0] = ls_heap[0]
        for k in range(3):
            vs[0, k] = vertices[indices, k]
        if ls[0] > 0 and sum(vs[0]) != 0:
            #print(*der, *testv, *iso, *aniso)
            refine_rank1_3d(ls[0:1], vs[0], tens, der, testv, iso, aniso)
        return
    #print(2)
    odf_min = np.min(ten_eval)
    odf_min = odf_min if (odf_min >= 0.) else 0.
    #print(3)
    # because of the relative threshold this algorithm will give the same peaks
    # as if we divide (values - odf_min) with (odf_max - odf_min) or not so
    # here we skip the division to increase speed
    values_norm = (ls_heap - odf_min)
    # Remove small peaks
    n = search_descending(values_norm, relative_peak_threshold)
    #print(4)
    indices = indices[:n]
    vs_heap = vertices[indices]
    #print(5)
    #  print(len(vs_heap))
    # Remove peaks too close together
    vs_heap, uniq = remove_similar_vertices(vs_heap, min_separation_angle)
    values = ls_heap[uniq]
    for j in range(min(len(uniq), max_num)):
        ls[j] = values[j]
        # print(7)
        for k in range(3):
            vs[j, k] = vs_heap[j, k]
        # print(*ls[i], *vs[i,j], *tens[1:, i])
        # refine as in the teem library
        if ls[k] > 0 and sum(vs[k]) != 0:
            refine_rank1_3d(ls[j:j + 1], vs[j], tens, der, testv, iso, aniso)

cpdef csd_peaks(double[:,:] tens, int max_num, float relative_peak_threshold, float min_separation_angle):
    """
    Given a set of fodfs we extract max_num highest peaks.
    """
    sphere = np.load(dirname + '/sphere.npz')
    tens_eval = hota_8o3d_sym_eval(tens[1:].T, sphere['vertices'].T)
    cdef int NUM = tens.shape[1], i, size = tens.shape[0] -1
    cdef double[:,:,:] vs = np.zeros((NUM, max_num, 3))
    cdef double[:,:,:] ls = np.zeros((NUM, max_num, 1))
    cdef double[:] testv = np.zeros((3,),dtype=np.float64), der = np.zeros((3,),dtype=np.float64), \
        iso = np.zeros(size,dtype=np.float64),  aniso = np.zeros((size,),dtype=np.float64)
    for i in tqdm(range(NUM)):
        if tens[0, i] == 0:
            continue
        maxima_finder(ls[i, :, 0], vs[i], tens[1:,i], tens_eval[i], sphere['edges'], sphere['vertices'], der, testv, iso,
                      aniso, relative_peak_threshold, min_separation_angle, max_num)


    return np.concatenate((np.asarray(ls), np.asarray(vs)), axis=-1).transpose(2,1,0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef remove_similar_vertices(double[:, :] vertices, double theta, bint return_mapping=False, bint return_index=True):
    """Remove vertices that are less than `theta` degrees from any other
    Returns vertices that are at least theta degrees from any other vertex.
    Vertex v and -v are considered the same so if v and -v are both in
    `vertices` only one is kept. Also if v and w are both in vertices, w must
    be separated by theta degrees from both v and -v to be unique.
    Parameters
    ----------
    vertices : (N, 3) ndarray
        N unit vectors.
    theta : float
        The minimum separation between vertices in degrees.
    return_mapping : {False, True}, optional
        If True, return `mapping` as well as `vertices` and maybe `indices`
        (see below).
    return_indices : {False, True}, optional
        If True, return `indices` as well as `vertices` and maybe `mapping`
        (see below).
    Returns
    -------
    unique_vertices : (M, 3) ndarray
        Vertices sufficiently separated from one another.
    mapping : (N,) ndarray
        For each element ``vertices[i]`` ($i \in 0..N-1$), the index $j$ to a
        vertex in `unique_vertices` that is less than `theta` degrees from
        ``vertices[i]``.  Only returned if `return_mapping` is True.
    indices : (N,) ndarray
        `indices` gives the reverse of `mapping`.  For each element
        ``unique_vertices[j]`` ($j \in 0..M-1$), the index $i$ to a vertex in
        `vertices` that is less than `theta` degrees from
        ``unique_vertices[j]``.  If there is more than one element of
        `vertices` that is less than theta degrees from `unique_vertices[j]`,
        return the first (lowest index) matching value.  Only return if
        `return_indices` is True.
    """
    if vertices.shape[1] != 3:
        raise ValueError('Vertices should be 2D with second dim length 3')
    cdef:
        cnp.ndarray[cnp.float_t, ndim=2, mode='c'] unique_vertices
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] mapping
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] index
        char pass_all
        # Variable has to be large enough for all valid sizes of vertices
        cnp.npy_int32 i, j
        cnp.npy_int32 n_unique = 0
        # Large enough for all possible sizes of vertices
        cnp.npy_intp n = vertices.shape[0]
        double a, b, c, sim
        double cos_similarity = cos(pi/180 * theta)
    if n >= 2**16:  # constrained by input data type
        raise ValueError("too many vertices")
    unique_vertices = np.empty((n, 3), dtype=float)
    if return_mapping:
        mapping = np.empty(n, dtype=np.uint16)
    if return_index:
        index = np.empty(n, dtype=np.uint16)

    for i in range(n):
        pass_all = 1
        a = vertices[i, 0]
        b = vertices[i, 1]
        c = vertices[i, 2]
        # Check all other accepted vertices for similarity to this one
        for j in range(n_unique):
            sim = fabs(a * unique_vertices[j, 0] +
                       b * unique_vertices[j, 1] +
                       c * unique_vertices[j, 2])
            if sim > cos_similarity:  # too similar, drop
                pass_all = 0
                if return_mapping:
                    mapping[i] = j
                # This point unique_vertices[j] already has an entry in index,
                # so we do not need to update.
                break
        if pass_all:  # none similar, keep
            unique_vertices[n_unique, 0] = a
            unique_vertices[n_unique, 1] = b
            unique_vertices[n_unique, 2] = c
            if return_mapping:
                mapping[i] = n_unique
            if return_index:
                index[n_unique] = i
            n_unique += 1

    verts = unique_vertices[:n_unique].copy()
    if not return_mapping and not return_index:
        return verts
    out = [verts]
    if return_mapping:
        out.append(mapping)
    if return_index:
        out.append(index[:n_unique].copy())
    return out


