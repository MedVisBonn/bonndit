#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from .ItoW cimport Trafo
from .alignedDirection cimport Probabilities
cdef class Interpolation:
	cdef double[:,:,:,:,:] vector_field
	cdef double[:,:,:] cuboid
	cdef double[:,:]  floor_point
	cdef double[:,:] best_dir
	cdef double chosen_angle
	cdef Trafo trafo
	cdef Probabilities prob
	cdef int best_ind
	cdef double[:] next_dir, vector
	cdef void main_dir(self, double[:]) nogil
	cdef void calc_cube(self, double[:]) nogil
	cdef void nearest_neigh(self, double[:]) nogil
	cdef void set_vector(self, int, int) nogil
	cdef void interpolate(self, double[:], double[:]) nogil except *

cdef class FACT(Interpolation):
	cdef void interpolate(self, double[:], double[:]) nogil except *


cdef class Trilinear(Interpolation):
	cdef double[:,:] array, x_array, new_best_dir
	cdef int[:,:] not_check
	cdef double[:] point
	cdef double[:,:,:] dir
	cdef int[:,:,:,:,:] cache
	cdef int[:] floor
	cdef int[:] permutation
	cdef void set_array(self, int, int, int) nogil
	cdef void interpolate(self, double[:], double[:]) nogil except *
	cdef int kmeans(self, double[:]) nogil except *
	cdef void permute(self, double[:]) nogil except *
