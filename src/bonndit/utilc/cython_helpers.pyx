#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
cimport cython
from libc.math cimport pow, floor, acos, pi
from libc.stdio cimport printf

cdef double min_c(double a, double b) nogil:
    if a < b:
        return a
    else:
        return b

cdef double max_c(double a, double b) nogil:
    if a < b:
        return b
    else:
        return a

cdef double dist(double[:] v, double[:]  w) nogil:
    cdef int i, n =v.shape[0]
    cdef double res = 0
    for i in range(n):
        res += pow(v[i] - w[i], 2)
    return pow(res, 0.5)

cdef double angle_deg(double[:] vec1, double[:] vec2) nogil:
    """ Calculate the angle between two vectors

    Parameters
    ----------
    vec1 :
    vec2 : two vectors of same dimensions.

    Returns
    -------

    """
    if vec1.shape[0] != vec2.shape[0]:
        printf("Wrong dimensions \n")
    if acos(clip(scalar(vec1, vec2) / (norm(vec1) * (norm(vec2))), -1,1)) * 180/ pi < 0:
        printf("Something is wrong \n")
    return acos(clip(scalar(vec1, vec2) / (norm(vec1) * (norm(vec2))), -1,1)) * 180 / pi

cdef void matrix_mult(double[:] res, double[:,:] A, double[:] v) nogil:
    """Easy matrix multiplication.
    :param res: Result vector size m
    :param A: Matrix size mxn
    :param v: vector size n
    :return: res size n
    """

    cdef int i, j, n = A.shape[0], m=A.shape[1]
    if n != res.shape[0] or m != v.shape[0]:
        printf("Wrong dimensions\n")

    for i in range(n):
        for j in range(m):
            res[i] += A[i,j] * v[j]

cdef int argmax(double[:] a) nogil:
    cdef int index = 0, n = a.shape[0], i = 0
    cdef double m = 0
    for i in range(n):
        if m < a[i]:
            index = i
            m = a[i]
    return index


cdef double clip(double a, double min_value, double max_value) nogil:
    return min_c(max_c(a, min_value), max_value)

cdef double sum_c(double[:] v) nogil:
    cdef int i, n = v.shape[0]
    cdef double res = 0
    for i in range(n):
        res += v[i]
    return res

cdef int sum_c_int(int[:] v) nogil:
    cdef int i, n = v.shape[0]
    cdef int res = 0
    for i in range(n):
        res += v[i]
    return res

cdef void floor_pointwise_matrix(double[:,:] res, double[:,:] v) nogil:
    cdef int i, j, n = v.shape[0], m = v.shape[1]
    for i in range(n):
        for j in range(m):
            res[i, j] = floor(v[i, j])

cdef void add_pointwise(double[:,:] res, double[:,:] v, double[:] w) nogil:
    cdef int i, j, n = v.shape[0], m = v.shape[1]
    for i in range(n):
        for j in range(m):
            res[i,j] = v[i,j] + w[j]

cdef double scalar(double[:] v, double[:] w) nogil:
    cdef int i, k = v.shape[0]
    cdef double res = 0
    for i in range(k):
        res += v[i] * w[i]
    return res

cdef double norm(double[:] v) nogil:
    return pow(scalar(v,v), 1/2)


cdef double scalar_minus(double[:] v, double[:] w) nogil:
    cdef int i, k = v.shape[0]
    cdef double res = 0
    for i in range(k):
        res = -v[i] * w[i]
    return res

cdef void mult_with_scalar(double[:] res, double s, double[:] v) nogil:
    cdef int i, k = v.shape[0]
    for i in range(k):
        res[i] = s*v[i]


cdef void add_vectors(double[:] res, double[:] v, double[:] w) nogil:
    cdef int i, k = v.shape[0]
    for i in range(k):
        res[i] = v[i] + w[i]

cdef void sub_vectors(double[:] res, double[:] v, double[:] w) nogil:
    cdef int i, k = v.shape[0]
    for i in range(k):
        res[i] = v[i] - w[i]

cdef double fak(int l) nogil:
    cdef int i
    cdef int x = 1
    for i in range(1,l+1):
        x *= i
    return x

@cython.cdivision(True)
cdef double binom(int n,int k) nogil:
    return fak(n)/(fak(k)*fak(n-k))

cdef void set_zero_matrix(double[:,:] v) nogil:
    cdef int i, j, m = v.shape[0], n = v.shape[1]
    for i in range(m):
        for j in range(n):
            v[i, j] = 0

cdef void set_zero_matrix_int(int[:,:] v) nogil:
    cdef int i, j, m = v.shape[0], n = v.shape[1]
    for i in range(m):
        for j in range(n):
            v[i, j] = 0

cdef void set_zero_vector(double[:] v) nogil:
    cdef int i, k = v.shape[0]
    for i in range(k):
        v[i] = 0

cdef void set_zero_vector_int(int[:] v) nogil:
    cdef int i, k = v.shape[0]
    for i in range(k):
        v[i] = 0

cdef void set_zero_3d(double[:,:,:] v) nogil:
    cdef int i, j, k, m = v.shape[0], n = v.shape[1], o = v.shape[2]
    for i in range(m):
        for j in range(n):
            for k in range(o):
                v[i,j,k] = 0

