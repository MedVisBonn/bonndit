#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import Cython
from helper_functions.cython_helpers cimport add_pointwise, floor_pointwise_matrix, norm, mult_with_scalar,\
	add_vectors, sub_vectors, scalar, clip, set_zero_vector, set_zero_matrix, sum_c, sum_c_int, set_zero_vector_int, \
	angle_deg, set_zero_matrix_int
import numpy as np
from .ItoW cimport Trafo

from .alignedDirection cimport Probabilities
from libc.math cimport pow, pi, acos, floor, fabs
from libc.stdio cimport printf
DTYPE = np.float64
###
# Given the ItoW trafo matrix and a cuboid of 8 points shape [2, 2, 2, 3] with a vector for each edge compute the
# trilinear
# interpolation

cdef int[:] permutation = np.array([0,1,2]*8, dtype=np.int32)
cdef double[:,:] neigh = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=DTYPE)

cdef int[:,:] neigh_int = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=np.int32)

cdef class Interpolation:

	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Trafo trafo, Probabilities prob):
		self.vector_field = vector_field
		self.vector = np.zeros((3,), dtype=DTYPE)
		self.cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
		self.floor_point = np.zeros((8, 3), dtype=DTYPE)
		self.best_dir  = np.zeros((3,3), dtype=DTYPE)
		self.next_dir = np.zeros((3,), dtype=DTYPE)
		self.best_ind = 0
		self.trafo = trafo
		self.prob = prob



	cdef void calc_cube(self,double[:] point) nogil:
		""" This function calculates the cube around a point.
		
		Parameters
		----------
		point 3 dimensional point


		Returns
		------- 

		"""
		add_pointwise(self.floor_point, neigh, point)
		floor_pointwise_matrix(self.floor_point, self.floor_point)

	cdef void nearest_neigh(self,double[:] point) nogil:
		""" Return the nearest neighbour to a given point by l2 norm. Therefore uses the cube around a point and calc 
		the distance to all other points.
		
		Parameters
		----------
		point

		Returns
		-------

		"""
		cdef double dist, act_dist
		cdef int index, best_ind
		self.calc_cube(point)
		# since linearity this works. Otherwise we first have to shift and the calc distances. For linear functions
		# it doesnt matter.
		sub_vectors(self.vector, self.floor_point[0], point)
		dist = norm(self.vector)
		# find nearest neighbor
		for index in range(8):
			sub_vectors(self.vector, self.floor_point[index], point)
			act_dist = norm(self.vector)
			if act_dist <= dist:
				dist = act_dist
				best_ind = index
		self.best_ind = best_ind

	#somehow slicing is not possible
	cdef void set_vector(self, int index, int j) nogil:
		cdef int i
		for i in range(3):
			self.vector[i] = self.vector_field[i + 1, j, int(self.floor_point[index, 0]),
		                  int(self.floor_point[index, 1]), int(self.floor_point[index, 2])]


	#@Cython.cdivision(True)
	cdef void main_dir(self, double[:] point) nogil:
			self.nearest_neigh(point)
			self.set_vector(self.best_ind, 0)
			mult_with_scalar(self.next_dir, pow(self.vector_field[0, 0, int(self.floor_point[			                                                                                            self.best_ind, 0]),
			                                                   int(self.floor_point[ self.best_ind, 1]),
			                                                  int(self.floor_point[self.best_ind, 2])], 0.25), self.vector)

	cdef void interpolate(self,double[:] point, double[:] old_dir) nogil except *:
		pass


cdef class FACT(Interpolation):
	cdef void interpolate(self, double[:] point, double[:] old_dir) nogil except *:
		cdef int i
		cdef double l
		self.nearest_neigh(point)
		for i in range(3):
			if self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[self.best_ind, 1]),int(self.floor_point[self.best_ind, 2])] != 0:
				l = pow(fabs(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),
				        int(self.floor_point[self.best_ind, 1]),int(self.floor_point[self.best_ind,
				                                                    2])]), 1/4)
			else:
				l = 0
			self.set_vector(self.best_ind, i)
			mult_with_scalar(self.best_dir[i], l, self.vector)
		#printf('%i \n', thread_id)
		self.prob.calculate_probabilities(self.best_dir, old_dir)
		mult_with_scalar(self.next_dir, 1, self.prob.best_fit)



cdef class Trilinear(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Trafo trafo, Probabilities prob):
		super(Trilinear, self).__init__(vector_field, grid, trafo, prob)
		self.array = np.zeros((2,3), dtype=DTYPE)
		self.x_array = np.zeros((4,3), dtype=DTYPE)
		self.point = np.zeros((3,), dtype=DTYPE)
		self.dir = np.zeros((8, 3, 3), dtype=DTYPE)
		self.new_best_dir = np.zeros((3, 3), dtype=DTYPE)
		self.cache = np.zeros((grid[0], grid[1], grid[2], 24,2), dtype=np.int32)
		self.permutation = np.zeros((16,), dtype=np.int32)
		self.not_check = np.zeros((3,2), dtype=np.int32)
		self.floor = np.zeros((3,), dtype=np.int32)



	cdef void set_array(self, int array, int index, int i) nogil:
		self.set_vector(index, i)
		if self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
		                     int(self.floor_point[index, 2])] != 0 and self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
		                     int(self.floor_point[index, 2])] == self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
		                     int(self.floor_point[index, 2])]:
			mult_with_scalar(self.array[array], pow(self.vector_field[0, i, int(self.floor_point[index, 0]),
			                                                          int(self.floor_point[index, 1]),
		                                                          int(self.floor_point[index, 2])], 1 / 4),
			                 self.vector)
		else:
			mult_with_scalar(self.array[array], 0, self.vector)



	cdef void interpolate(self, double[:] point, double[:] old_dir) nogil except *:
		""" This function calculates the interpolation based on https://en.wikipedia.org/wiki/Trilinear_interpolation 
		for each vectorfield. Afterwards the we chose randomly from the 3 vectors.
		
		Parameters
		----------
		point   Point in plane
		old_dir direction in point

		Returns
		-------

		"""
		cdef int i, j
		cdef int con = 1

		self.calc_cube(point)
		for i in range(3):
			self.floor[i] = int(point[i])


		#if sum_c_int(self.cache[self.floor[0], self.floor[1], self.floor[2],:,0]) == 0:
		con = self.kmeans(point)
		#else:
		#	self.permute(point)
		if con:

			for i in range(3):
				self.point[i] = point[i]%1

			for i in range(3):
				# interpolate in x direction
				for j in range(4):
					mult_with_scalar(self.array[0], 1 - self.point[0], self.cuboid[4+j, i])
					mult_with_scalar(self.array[1], self.point[0], self.cuboid[j, i])
					add_vectors(self.x_array[j], self.array[0], self.array[1])

				# then in y direction
				mult_with_scalar(self.x_array[0], 1-self.point[1], self.x_array[0])
				mult_with_scalar(self.x_array[1], self.point[1], self.x_array[1])
				add_vectors(self.array[0], self.x_array[0], self.x_array[1])
				mult_with_scalar(self.x_array[2], 1 - self.point[1], self.x_array[2])
				mult_with_scalar(self.x_array[3], self.point[1], self.x_array[3])
				add_vectors(self.array[1], self.x_array[2], self.x_array[3])

				# then z direction
				mult_with_scalar(self.array[0], 1 - self.point[2], self.array[0])
				mult_with_scalar(self.array[1], self.point[2], self.array[1])
				add_vectors(self.best_dir[i], self.array[0], self.array[1])


			self.prob.calculate_probabilities(self.best_dir, old_dir)
			self.next_dir = self.prob.best_fit

		else:
			mult_with_scalar(self.next_dir, 0, self.prob.best_fit)

#	### TODO is here a better way?
	cdef void permute(self, double[:] point) nogil except *:
		""" Little anoying... If a cube was already cached, this function uses the cache to set the dir parameter
		accordingly. We loop through all indices and for each index through the 2 saved vectors. The two saved
		vectors are set to the first two directions and the remaining is set to the last direction. To get the last
		one it should be enough to calculate 3-a-b where a and b are the indices of the other two directions.

		Since only the first two are saved

		Parameters
		----------
		mapping Vector dimension 16 with entries from 0 to two which define the permutation

		Returns
		-------

		"""
		cdef int i, index, k, l
		for index in range(8):
			for i in range(3):
				mult_with_scalar(self.dir[index,i], self.cache[int(point[0]), int(point[1]),int(point[2]), index*3 +
				                                               i, 1], self.cuboid[index,  self.cache[int(point[0]), int(point[1]),int(point[2]), index*3 + i, 0
				]])
		for index in range(8):
			for i in range(3):
				mult_with_scalar(self.cuboid[index,i], 1, self.dir[index, i])


#
	cdef int kmeans(self, double[:] point) nogil except *:
		cdef int i, j, k, l, max_try=0, minus, best_min=0, best=0, con=0
		cdef double exponent = 0, best_angle=0, min_angle=0
		set_zero_matrix_int(self.not_check)
		for i in range(24):
			self.cache[int(point[0]), int(point[1]), int(point[2]), i, 0] = permutation[i]
		for i in range(8):
			set_zero_matrix(self.cuboid[i])

		for i in range(8):
			for j in range(3):
				if self.vector_field[0, j, int(self.floor_point[i, 0]),
				                                       int(self.floor_point[i, 1]),
				                                       int(self.floor_point[i, 2])] != 0 and self.vector_field[0, j, int(self.floor_point[i, 0]),
				                                       int(self.floor_point[i, 1]),
				                                       int(self.floor_point[i, 2])] == self.vector_field[0, j, int(self.floor_point[i, 0]),
				                                       int(self.floor_point[i, 1]),
				                                       int(self.floor_point[i, 2])]:
					exponent = pow(fabs(self.vector_field[0, j, int(self.floor_point[i, 0]),
				                                       int(self.floor_point[i, 1]),
				                                       int(self.floor_point[i, 2])]), 1/4)
				else:
					exponent = 0
				# Does not work with mult_with_scalar dont understand :(
				for k in range(3):
					self.cuboid[i,j,k] = exponent * self.vector_field[1+k, j, int(self.floor_point[i, 0]),int(self.floor_point[i, 1]),int(self.floor_point[i, 2])]
				if sum_c(self.best_dir[j])!=0 and sum_c(self.cuboid[i,k])!=0:
					test_angle = angle_deg(self.best_dir[j], self.cuboid[i,j])
					if test_angle > 90:
						mult_with_scalar(self.cuboid[i,j], -1, self.cuboid[i,j])
				add_vectors(self.best_dir[j], self.best_dir[j], self.cuboid[i,j])
		for i in range(3):
			mult_with_scalar(self.best_dir[i],1/8, self.best_dir[i])
		while True:
			max_try += 1
			set_zero_matrix(self.new_best_dir)
			# each corner
			for i in range(8):
				for l in range(3):
					self.not_check[l, 0] = -1
				# each avg direction
				for j in range(3):
					# each possibility
					# Fit each direction recursively to the edge with the smallest angle. Keep them in groups of three.
					for l in range(3):
						min_angle = 0
						for k in range(3):
							if self.not_check[0,0]==k or self.not_check[1,0]==k:
								continue
							# angle between avg direction and corner direction:
							if sum_c(self.cuboid[i,k])!=0:
								test_angle = angle_deg(self.best_dir[j], self.cuboid[i,k])
								minus = 1
								if test_angle > 90:
									test_angle = 180 - test_angle
									minus = -1
								if min_angle == 0 or test_angle<min_angle:
									min_angle=test_angle
									best=k
									best_min = minus
						# No proper assignment found. Because angle where zero
						if min_angle!=0:
							self.not_check[l, 0]=best
							self.not_check[l, 1]=best_min
				# reoder cuboid according to order from above. And take the correct direction.

				for k in range(3):
					mult_with_scalar(self.dir[i, k], self.not_check[k, 1],self.cuboid[i, self.not_check[k,0]])

					#self.array_pl[k] = self.cache[int(point[0]),int(point[1]),int(point[2]),i * 3 + self.not_check[k,
					 #                                                                                            0], 0]
					#self.cache[int(point[0]), int(point[1]), int(point[2]), i * 3 + k, 1] *= self.not_check[k, 1]
			#	for k in range(3):
			#		self.cache[int(point[0]),int(point[1]),int(point[2]),i * 3 + k,0] = self.array_pl[k]


				for k in range(3):
					mult_with_scalar(self.cuboid[i, k], 1, self.dir[i, k])
					#build new avg from cuboids:
					add_vectors(self.new_best_dir[k], self.new_best_dir[k], self.cuboid[i, k])

			con = 1
			for i in range(3):
				mult_with_scalar(self.new_best_dir[i], 1/8, self.new_best_dir[i])
				sub_vectors(self.point, self.best_dir[i], self.new_best_dir[i])
				if norm(self.point) == 0:
					con = 0
				mult_with_scalar(self.best_dir[i], 1, self.new_best_dir[i])
			if con:
				return 1
			if max_try==100:
				with gil:
					print('NOOO')
				for i in range(24):
					self.cache[int(point[0]), int(point[1]), int(point[2]),i, 0]=permutation[i]
				return 0











