#%%cython --annotate
#cython: language_level=3, boundscheck=False,
import cython
from libc.math cimport acos, pi, exp, fabs, cos, pow, tanh
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from bonndit.utilc.cython_helpers cimport scalar, clip, mult_with_scalar, sum_c, norm
import numpy as np


###
# Given a set of vectors should be in shape [n,3] and a vector returns most aligned vector shape [3]

cdef class Probabilities:
	srand(time(NULL))



	def __cinit__(self, double expectation=0, double sigma=9):
		self.sigma = sigma
		self.expectation = expectation
		self.chosen_prob = 0
		self.probability = np.zeros((3,))
		self.angles = np.zeros((3,))
		self.best_fit = np.zeros((3))
		self.test_vectors = np.zeros((3, 3))
		self.chosen_angle = 0
		self.old_fa = 1


	cdef void aligned_direction(self, double[:,:] vectors, double[:] direction) nogil  except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		#calculate angle between direction and possibilities. If angle is bigger than 90 use the opposite direction.
		cdef int i, n = vectors.shape[0]
		cdef double test_angle, min_angle = 180

		for i in range(n):
			#with gil:
			#	print(*vectors[i])
			#	if sum(direction) == 0 or sum(vectors[i]) == 0:
			#		print(*direction,*vectors[i])
			if sum_c(direction) == 0:
				break
			if norm(vectors[i]) != 0 and norm(vectors[i]) == norm(vectors[i]):
				#with gil:
				#	print(norm(direction), norm(vectors[i]))
				test_angle = acos(clip(scalar(direction, vectors[i])/(norm(direction)*(norm(vectors[i]))), -1,
				                       1)) *180/pi
		#		with gil:
		#			print(test_angle)
				if test_angle < 90:
					self.angles[i] = test_angle
					self.test_vectors[i] = vectors[i]
				elif test_angle <= 180:
					self.angles[i] = 180 - test_angle
					mult_with_scalar(self.test_vectors[i], -1, vectors[i])
			else:
				self.angles[i] = 180
				mult_with_scalar(self.test_vectors[i], 0, vectors[i])

	cdef void random_choice(self, double[:] direction) nogil  except *:
		"""

		@param direction:
		@return:
		"""
		cdef double best_choice = rand() / RAND_MAX
		with gil:
			print(*self.probability)
		if sum_c(self.probability) != 0:
			mult_with_scalar(self.probability, 1/sum_c(self.probability), self.probability)


			if best_choice < self.probability[0]:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[0])
				self.chosen_prob = self.probability[0]
				self.chosen_angle = self.angles[0]
			elif best_choice < self.probability[0] + self.probability[1]:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[1])
				self.chosen_prob = self.probability[1]
				self.chosen_angle = self.angles[1]
			else:
				mult_with_scalar(self.best_fit, 1, self.test_vectors[2])
				self.chosen_prob = self.probability[2]
				self.chosen_angle = self.angles[2]
			self.old_fa = norm(self.best_fit)
		else:

		#	with gil:
		#		print(*direction, *self.test_vectors[0], *self.test_vectors[1],
		#		      *self.test_vectors[2])
			mult_with_scalar(self.best_fit, 0, self.test_vectors[2])
			self.chosen_angle = 0
			self.chosen_prob = 0
		with gil:
			print(*self.probability, ' and I chose ', self.chosen_prob, ' where the angle are ', *self.angles, ' I chose ', self.chosen_angle)



	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil except *:
		pass


cdef class Gaussian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i
		self.aligned_direction(vectors, direction)
		for i in range(3):
			self.probability[i] = exp(-1/2*((self.angles[i] - self.expectation)/self.sigma)**2)
		self.random_choice(direction)


cdef class Laplacian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i
		self.aligned_direction(vectors, direction)
		for i in range(3):
			self.probability[i] = 1/2 * exp(- (fabs(self.angles[i] - self.expectation) /
			                                             self.sigma))
		self.random_choice(direction)


cdef class ScalarOld(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i
		cdef double s
		self.aligned_direction(vectors, direction)
		#with gil:
		#	print(*self.angles)
		for i in range(3):
			if sum_c(self.test_vectors[i]) == sum_c(self.test_vectors[i])  and pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2) <= 1/2*pi:
				with gil:
					print('First angle ' , self.angles[i], pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2))
				self.probability[i]=pow(cos(pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2)),self.sigma)*norm(self.test_vectors[i])
			else:
				self.probability[i] = 0
		self.random_choice(direction)


cdef class ScalarNew(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil  except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i
		cdef double s
		self.aligned_direction(vectors, direction)
		#with gil:
		#	print(*self.angles)
		for i in range(3):
			if sum_c(vectors[i]) == sum_c(vectors[i]):
				self.probability[i] = pow(cos(self.angles[i]/180*pi),self.sigma)*norm(self.test_vectors[i])
			else:
				self.probability[i] = 0

		self.random_choice(direction)


cdef class Deterministic2(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil  except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i, min_index=0
		cdef double s, min_angle=0
		self.aligned_direction(vectors, direction)
		for i in range(3):
			if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
				if self.angles[i] < min_angle or i==0:
					min_angle=self.angles[i]
					min_index=i
		for i in range(3):
			if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
				if self.angles[i] < min_angle + self.sigma and min_angle < 30:
					self.probability[i] = 1
				else:
					self.probability[i] = 0
		self.probability[min_index] = 1
		self.random_choice(direction)
#		mult_with_scalar(self.best_fit, 1, self.test_vectors[min_index])
#		self.chosen_prob = 0
#		self.chosen_angle = self.angles[min_index]

cdef class Deterministic(Probabilities):
	cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) nogil  except *:
		"""

		@param vectors:
		@param direction:
		@return:
		"""
		cdef int i, min_index=0
		cdef double s, min_angle=0
		self.aligned_direction(vectors, direction)
		for i in range(3):
			if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
				if self.angles[i] < min_angle or i==0:
					min_angle=self.angles[i]
					min_index=i

		mult_with_scalar(self.best_fit, 1, self.test_vectors[min_index])
		self.chosen_prob = 0
		self.chosen_angle = self.angles[min_index]

