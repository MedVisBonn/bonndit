#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True
# warn.unused_results=True

from .ItoW cimport Trafo
from .interpolation cimport Interpolation

cdef class Integration:
	cdef double stepsize, width
	cdef double[:] next_point, three_vector
	cdef double[:,:] ItoW
	cdef double[:] origin
	cdef Trafo trafo
	cdef double[:] old_dir

	cdef int integrate(self, double[:], double[:]) # nogil except *

cdef class Euler(Integration):
	cdef int integrate(self, double[:], double[:]) # nogil except *

cdef class EulerUKF(Integration):
	cdef int integrate(self, double[:], double[:]) # nogil except *

cdef class RungeKutta(Integration):
	cdef Interpolation interpolate
	cdef double[:] k1
	cdef double[:] k2
	cdef double[:] k2_x

	cdef int integrate(self, double[:], double[:]) # nogil except *


