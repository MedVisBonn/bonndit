#%%cython --annotate
#cython: language_level=3, boundscheck=False,
import Cython
from libc.math cimport acos, pi, exp, abs, cos, pow
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from bonndit.helper_functions.cython_helpers cimport scalar, clip, mult_with_scalar, sum_c, norm
import numpy as np

cdef:
	struct Probabilities:
		double sigma
		double expectation
		double *probability
		double *angles
		double *best_fit
		double *test_vectors

	void aligned_direction(Probabilities *self, double* vectors, double* direction) nogil:
			"""

			"""
			cdef int i, n = vectors.shape[0]
			cdef double test_angle, min_angle = 180

			for i in range(n):
				#with gil:
				#	print(*vectors[i])
				#	if sum(direction) == 0 or sum(vectors[i]) == 0:
				#		print(*direction,*vectors[i])
				if norm(vectors[i]) != 0:
					test_angle = acos(clip(scalar(direction, vectors[i*3:(i+1)*3])/(norm(direction)*(norm(vectors[
						                                                                                      i*3:(i+1)*3]))), -1, 1)) *180/pi
					if test_angle < 90:
						self.angles[i] = test_angle
						self.test_vectors[i*3:(i+1)*3] = &vectors[i*3:(i+1)*3]
					else:
						self.angles[i] = 180 - test_angle
						mult_with_scalar(self.test_vectors[i*3:(i+1)*3], -1 , vectors[i*3:(i+1)*3])
				else:
					mult_with_scalar(self.test_vectors[i], 0, vectors[i])

	void random_choice(Probabilities *self, double* direction) nogil:
		cdef double best_choice = rand() / RAND_MAX
		if sum_c(self.probability) != 0:
			mult_with_scalar(self.probability, 1/sum_c(self.probability), self.probability)
			if best_choice < self.probability[0]:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[0])
			elif best_choice < self.probability[0] + self.probability[1]:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[1])
			else:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[2])
		else:
			with gil:
				print(*direction, *self.test_vectors[0], *self.test_vectors[1], *self.test_vectors[2])
			mult_with_scalar(self.best_fit, 0, self.test_vectors[2])

	void gaussian(Probabilities *self, double* vectors, double* direction) nogil:
			cdef int i
			self.aligned_direction(self, vectors, direction)
			for i in range(3):
				self.probability[i] = exp(-1/2*((self.angles[i] - self.expectation)/self.sigma)**2)
			self.random_choice(self, direction)

	void laplacian(Probabilities *self, double* vectors, double[:] direction) nogil:
			cdef int i
			self.aligned_direction(self, vectors, direction)
			for i in range(3):
				self.probability[i] = 1/2 * exp(- (abs(self.angles[i] - self.expectation) / self.sigma))
			self.random_choice(self, direction)

	void scalar(Probabilities *self, double[:,:] vectors, double[:] direction) nogil:
			cdef int i
			cdef double s
			self.aligned_direction(self, vectors, direction)
			for i in range(3):
				self.probability[i] = pow(norm(vectors[i]),4) * pow(cos(self.angles[i]/180*pi), self.sigma)
			self.random_choice(self, direction)
