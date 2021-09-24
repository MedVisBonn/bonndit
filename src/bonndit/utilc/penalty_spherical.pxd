#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cdef double calc_norm_spherical(double[:], double[:,:], double[:], double[:,:], double, double[:]) nogil except *
cdef double refine_average_spherical(double[:], double[:,:], double[:,:], double[:], double[:,:], double,
                                          double, double[:], double[:], double[:], double[:,:], double[:],
                                          double[:], double[:],  double[:],  double[:],  double[:], double[:],
                                          double[:]) nogil except *

