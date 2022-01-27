#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import Cython
from bonndit.utilc.cython_helpers cimport add_pointwise, floor_pointwise_matrix, norm, mult_with_scalar,\
	add_vectors, sub_vectors, scalar, clip, set_zero_vector, set_zero_matrix, sum_c, sum_c_int, set_zero_vector_int, \
	angle_deg, set_zero_matrix_int, point_validator
import numpy as np
from .ItoW cimport Trafo
cdef int[:,:] permute_poss = np.array([[0,1,2],[0,2,1], [1,0,2], [1,2,0], [2,1,0], [2,0,1]], dtype=np.int32)

from .alignedDirection cimport Probabilities
from libc.math cimport pow, pi, acos, floor, fabs,fmax
from libc.stdio cimport printf
DTYPE = np.float64
###
# Given the ItoW trafo matrix and a cuboid of 8 points shape [2, 2, 2, 3] with a vector for each edge compute the
# trilinear
# interpolation

cdef int[:] permutation = np.array([0,1,2]*8, dtype=np.int32)
cdef double[:,:] neigh = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=DTYPE)
cdef int[:] minus = np.array((3,), dtype=np.int32)
cdef int[:,:] neigh_int = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=np.int32)
cdef int[:] best = np.zeros((4*8,), dtype=np.int32),  old_best = np.zeros((8,), dtype=np.int32)
cdef double[:,:,:] test_cuboid = np.zeros((8, 3, 3), dtype=DTYPE)

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
		cdef double l, max_value
		self.nearest_neigh(point)
		max_value = fmax(fmax(self.vector_field[0, 0, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[
			                                                                                   self.best_ind,1]),
		                                   int(self.floor_point[self.best_ind, 2])], self.vector_field[0, 1,
		                                                                                               int(self.floor_point[self.best_ind, 0]), int(self.floor_point[
			                                                                                   self.best_ind,1]),
		                                                                                               int(
			                                                                                               self.floor_point[self.best_ind, 2])]), self.vector_field[0, 2, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[
			                                                                                   self.best_ind,1]),
		                                                                                                                                                            int(self.floor_point[self.best_ind, 2])])

		for i in range(3):
			if self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[self.best_ind,
			                                                                                         1]),
			                     int(self.floor_point[self.best_ind, 2])] > max_value/10:
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
		self.cache = np.zeros((grid[0], grid[1], grid[2], 4*8), dtype=np.int32)
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
		cdef double test=0
		self.calc_cube(point)
		for i in range(3):
			self.floor[i] = int(point[i])
		# Check if the best dir is initialized. If no initizialize first with the nearest neighbor. Then fit neighbors.
		for i in range(3):
			test+=norm(self.best_dir[i])
		if test==0:
			self.nearest_neigh(point)
			for i in range(3):
				if point_validator(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[self.best_ind,1]),int(self.floor_point[self.best_ind, 2])], 1):
					l = pow(fabs(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),int(self.floor_point[self.best_ind, 1]), int(self.floor_point[self.best_ind,2])]), 1 / 4)
				else:
					l = 0
				self.set_vector(self.best_ind, i)
				mult_with_scalar(self.best_dir[i], l, self.vector)

		if sum_c_int(self.cache[self.floor[0], self.floor[1], self.floor[2],:]) == 0:
			self.permute(point)
		else:
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
				mult_with_scalar(self.dir[index,i], self.cache[int(point[0]), int(point[1]),int(point[2]), index*4 + 1 +i], self.cuboid[index,  permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i]])
		for index in range(8):
			for i in range(3):
				mult_with_scalar(self.cuboid[index,i], 1, self.dir[index, i])

	cdef void set_new_poss(self) nogil except *:
		cdef int i,j,k
		for i in range(8):
			set_zero_matrix(self.cuboid[i])
		for i in range(8):
			for j in range(3):
				for k in range(3):
					self.cuboid[i, j, k] = best[4*i + j + 1] * test_cuboid[i, permute_poss[best[4*i], j], k]

	cdef int kmeans(self, double[:] point) nogil except *:
		cdef int i, j, k, l, max_try=0, best_min=0
		cdef double exponent = 0, best_angle=0, min_angle=0, con=0
		for i in range(8):
			set_zero_matrix(test_cuboid[i])

		for i in range(8):
			for j in range(3):
				if point_validator(self.vector_field[0, j, int(self.floor_point[i, 0]), int(self.floor_point[i, 1]), int(self.floor_point[i, 2])], 1):
					exponent = pow(fabs(self.vector_field[0, j, int(self.floor_point[i, 0]),
				                                       int(self.floor_point[i, 1]),
				                                       int(self.floor_point[i, 2])]), 1/4)
				else:
					exponent = 0
				# Does not work with mult_with_scalar dont understand :(
				for k in range(3):
					test_cuboid[i,j,k] = exponent *  self.vector_field[1 + k, j, int(self.floor_point[i, 0]),int(self.floor_point[i, 1]),int(self.floor_point[i, 2])]

				if norm(self.best_dir[j])!=0 and norm(test_cuboid[i,k])!=0:
					test_angle = angle_deg(self.best_dir[j], test_cuboid[i,j])
					if test_angle > 90:
						mult_with_scalar(test_cuboid[i,j], -1, test_cuboid[i,j])
#				add_vectors(self.best_dir[j], self.best_dir[j], test_cuboid[i,j])
		while True:
			con = 0
			max_try += 1
			# each corner
			for i in range(8):
				min_angle = 0
				for j in range(6):
					set_zero_vector_int(minus)
					for k in range(3):
						if norm(test_cuboid[i,permute_poss[j,k]]) != 0 and norm(self.best_dir[k]) != 0:
							ang = angle_deg(self.best_dir[k], test_cuboid[i, permute_poss[j,k]])
							if ang > 90:
								test_angle += 180 - ang
								minus[k] = -1
							else:
								test_angle = ang
								minus[k] = 1
					if min_angle == 0 or test_angle < min_angle:
						min_angle = test_angle
						best[4*i] = j
						for k in range(3):
							best[4*i + k + 1] = minus[k]

				for j in range(3):
					if norm(self.best_dir[j]) == 0:
						add_vectors(self.best_dir[j], self.best_dir[j], test_cuboid[i, permute_poss[best[4*i], j]])
			for i in range(8):
				con += fabs(best[4*i] - old_best[i])
				old_best[i] = best[4*i]
			if con == 0:
				con = 1
				break
			set_zero_matrix(self.best_dir)
#			for i in range(8):
#				for j in range(3):
#					mult_with_scalar(test_cuboid[i, permute_poss[best[4 * i], j]], best[4 * i + k + 1], test_cuboid[i, permute_poss[best[4 * i], j]])
#					add_vectors(self.best_dir[j], self.best_dir[j], test_cuboid[i, permute_poss[best[4*i],j]])

			if max_try == 100:
				con = 0
				with gil: print('I do not converge')
				break

		for i in range(32):
			self.cache[self.floor[0], self.floor[1], self.floor[2],i] = int(best[i])
		self.set_new_poss()
		return int(con)










