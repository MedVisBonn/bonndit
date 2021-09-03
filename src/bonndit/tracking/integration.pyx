#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True

import numpy as np
from helper_functions.cython_helpers cimport norm, add_vectors, mult_with_scalar, sum_c
from .ItoW cimport Trafo
###
# Given a direction and a Coordinate compute the next point

cdef class Integration:

	def __cinit__(self, double[:,:] ItoWMatrix, double[:] origin, Trafo trafo, double stepsize):
		self.stepsize = stepsize
		self.trafo = trafo
		self.ItoW = ItoWMatrix
		self.origin = origin
		self.next_point = np.zeros((3,))
		self.three_vector = np.zeros((3,))
		self.old_dir = np.ndarray((3,))

	cdef void integrate(self, double[:] direction, double[:] coordinate) nogil:
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
	cdef void integrate(self, double[:] direction, double[:] coordinate) nogil:
		""" Euler Integration

		Converts itow and adds the current direction to the current position

		Parameters
		----------
		direction: current direction
		coordinate: current coordinate


		"""
		#print("ssd", *coordinate)
		self.trafo.itow(coordinate)
		mult_with_scalar(self.three_vector, self.stepsize/norm(direction), direction)
		add_vectors(self.three_vector, self.trafo.point_itow, self.three_vector)
		#print("threee ", *self.three_vector)
		self.trafo.wtoi(self.three_vector)
		self.next_point = self.trafo.point_wtoi



"""
# Calculate next steps according to Wikipedia https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren for constant time?
# best fit to current coordinate. Branching prohibited
class RungeKutta(Integration):
	def __init__(self, ItoWMatrix, origin, trafo, stepsize, interpolation):
		super().__init__(ItoWMatrix, origin, trafo, stepsize)
		self.interpolate = interpolation

	def integrate(self, direction, coordinate):
		k1 = direction
		k2_x = self.trafo.wtoi(self.trafo.itow(coordinate) + self.stepsize / 2 * k1)
		self.interpolate.interpolate(k2_x, k1, 1000, 1000)
		k2 = self.interpolate.best_dir[0]
		k3_x = self.trafo.wtoi(self.trafo.itow(coordinate) - self.stepsize * k1 + self.stepsize * 2 * k2)
		self.interpolate.interpolate(k3_x, k1, 1000, 1000)
		k3 = self.interpolate.best_dir[0]
		return self.trafo.wtoi(
			self.trafo.itow(coordinate) + self.stepsize * (1 / 6 * k1 + 4 / 6 * k2 + 1 / 6 * k3)), np.linalg.norm(
			self.stepsize * (1 / 6 * k1 + 4 / 6 * k2 + 1 / 6 * k3))
"""
