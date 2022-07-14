#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True

import numpy as np
from bonndit.utilc.cython_helpers cimport norm, add_vectors, mult_with_scalar, sum_c
from .ItoW cimport Trafo
###
# Given a direction and a Coordinate compute the next point

cdef class Integration:

	def __cinit__(self, double[:,:] ItoWMatrix, double[:] origin, Trafo trafo, double stepsize, **kwargs):
		self.stepsize = stepsize
		self.trafo = trafo
		self.ItoW = ItoWMatrix
		self.origin = origin
		self.next_point = np.zeros((3,))
		self.three_vector = np.zeros((3,))
		self.old_dir = np.ndarray((3,))

	cdef int integrate(self, double[:] direction, double[:] coordinate) nogil except *:
		pass



"""
###
# calc next intersection with a border via solving
# ItoW * act_coor  + act_dir * x  = x + origin
# x = ( act_dir - id )^-1 * ItoW^-1 * origin - coor
###
cdef class FACT(Integration):
	cdef void integrate(self, direction, coordinate) nogil:
		direction_inv = np.linalg.inv(np.dot(direction, np.identity(3)) - np.identity(3))
		np.dot(direction_inv, np.dot(np.linalg.inv(self.ItoW), self.origin)) - coordinate, np.linalg.norm(
			self.stepsize)
"""

# Euler Integration. Transform to world coordinates before integrating. Transform back afterwards.
cdef class Euler(Integration):
	cdef int integrate(self, double[:] direction, double[:] coordinate) nogil except *:
		""" Euler Integration

		Converts itow and adds the current direction to the current position

		Parameters
		----------
		direction: current direction
		coordinate: current coordinate


		"""
		#print("ssd", *coordinate)
		self.trafo.itow(coordinate)
		self.old_dir = direction
		mult_with_scalar(self.three_vector, self.stepsize/norm(direction), direction)
		add_vectors(self.three_vector, self.trafo.point_itow, self.three_vector)
		self.trafo.wtoi(self.three_vector)
		self.next_point = self.trafo.point_wtoi#self.three_vector #coordinate #self.trafo.point_wtoi
		return 0

cdef class EulerUKF(Integration):
	cdef int integrate(self, double[:] direction, double[:] coordinate) nogil except *:
		""" Euler Integration

		Converts itow and adds the current direction to the current position

		Parameters
		----------
		direction: current direction
		coordinate: current coordinate


		"""
		mult_with_scalar(self.three_vector, self.stepsize/norm(direction), direction)
		self.old_dir = direction
		add_vectors(self.next_point, coordinate, self.three_vector)
		return 0


# Calculate next steps according to Wikipedia https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren for constant time?
# best fit to current coordinate. Branching prohibited
cdef class RungeKutta(Integration):
	def __cinit__(self, double[:,:] ItoWMatrix, double[:] origin, Trafo trafo, double stepsize, **kwargs):
		super().__init__(ItoWMatrix, origin, trafo, stepsize)
		self.interpolate = kwargs['interpolate']
		self.k1 =np.zeros((3,))
		self.k2 =np.zeros((3,))
		self.k2_x =np.zeros((3,))



	cdef int integrate(self, double[:] direction, double[:] coordinate) nogil except *:
		mult_with_scalar(self.three_vector, self.stepsize/(2*norm(direction)), direction)
		self.trafo.itow(coordinate)
		add_vectors(self.three_vector, self.trafo.point_itow, self.three_vector)
		self.trafo.wtoi(self.three_vector)
		self.k2_x = self.trafo.point_wtoi
		if self.interpolate.interpolate(self.k2_x, self.old_dir, 1) != 0:
			return 1
		self.k2 = self.interpolate.next_dir
		self.old_dir = self.interpolate.next_dir
		if sum_c(self.k2) == 0 or sum_c(self.k2) != sum_c(self.k2):
			return 1
		mult_with_scalar(self.k1, self.stepsize/norm(self.k2), self.k2)
		self.trafo.itow(coordinate)
		add_vectors(self.k2, self.trafo.point_itow, self.k1)
		self.trafo.wtoi(self.k2)
		self.next_point = self.trafo.point_wtoi
		return 0


