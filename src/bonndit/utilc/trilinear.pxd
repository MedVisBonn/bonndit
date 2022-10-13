#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cpdef double linear_p(double[:], double[:], double[:, :, :]) #nogil except *
cdef double linear(double[:], double[:], double[:, :, :]) nogil except *
