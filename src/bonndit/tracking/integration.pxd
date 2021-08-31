#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from .ItoW cimport Trafo

cdef class Integration:
	cdef double stepsize, width
	cdef double[:] next_point, three_vector
	cdef double[:,:] ItoW
	cdef double[:] origin
	cdef Trafo trafo
	cdef void integrate(self, double[:], double[:]) nogil

cdef class Euler(Integration):
	cdef void integrate(self, double[:], double[:]) nogil