#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=True, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from bonndit.utilc.cython_helpers cimport matrix_mult, add_vectors, sub_vectors, set_zero_vector
import numpy as np
DTYPE = np.float64

cdef class Trafo:

	def __cinit__(self, double[:,:] ItoW, double[:] origin):
		self.ItoW = ItoW
		self.origin = origin
		self.ItoW_inv = np.linalg.inv(ItoW)
		self.point_itow = np.zeros((3,))
		self.point_wtoi = np.zeros((3,))
		self.three_vector = np.zeros((3,))



	cdef void itow(self, double[:] point) : # nogil:
		"""
		Converts a point from index space to world space
		@param point: vector (3,)
			Input point
		@return: Nothing. Result is saved in point_itow
		"""
		set_zero_vector(self.point_itow)
		matrix_mult(self.point_itow, self.ItoW, point)
		add_vectors(self.point_itow, self.origin, self.point_itow)


	cdef void wtoi(self, double[:] point) : # nogil:

		"""
    		Converts a point from world space to index space
    		@param point: vector (3,)
    		    Input point
    		@return: Nothing. Result is saved in point_wtoi
		"""
		set_zero_vector(self.point_wtoi)
		sub_vectors(self.three_vector, point, self.origin)
		matrix_mult(self.point_wtoi, self.ItoW_inv, self.three_vector)

	##Python wrapper
	cpdef wtoi_p(self, double[:] point):
		self.wtoi(point)
		return np.asarray(self.point_wtoi)

	cpdef itow_p(self, double[:] point):
		self.itow(point)
		return np.asarray(self.point_itow)

