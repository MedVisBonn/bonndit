#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True

cdef double approx_initial(double[:], double[:,:], double[:,:], double[:], int ,double[:], double[:], double[:],
                                double[:],double[:], double[:]) # nogil
cdef void init_max_3d(double[:], double[:], double[:]) nogil
cdef int refine_rank1_3d(double[:], double[:], double[:], double[:], double[:],double[:], double[:]) #nogil except *
