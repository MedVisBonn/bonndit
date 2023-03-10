#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True
# warn.unused_results=True

cdef class Trafo:
	cdef double[:] point_wtoi, point_itow, three_vector
	cdef double[:,:] ItoW, ItoW_inv
	cdef double[:] origin
	cdef void itow(self, double[:]) # nogil
	cdef void wtoi(self, double[:]) # nogil
	cpdef wtoi_p(self, double[:] point)
	cpdef itow_p(self, double[:])
