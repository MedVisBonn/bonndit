#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, profile=False,
# warn.unused_results=True

import Cython
import cython
from tqdm import tqdm
from bonndit.utilc.cython_helpers cimport add_pointwise, floor_pointwise_matrix, norm, mult_with_scalar,\
	add_vectors, sub_vectors, scalar, clip, set_zero_vector, set_zero_matrix, sum_c, sum_c_int, set_zero_vector_int, \
	angle_deg, set_zero_matrix_int, point_validator
import numpy as np
import time
from bonndit.utilc.cython_helpers cimport dm2toc
from bonndit.utilc.hota cimport hota_4o3d_sym_norm, hota_4o3d_sym_eval
from bonndit.utilc.lowrank cimport approx_initial
from .ItoW cimport Trafo
cdef int[:,:] permute_poss = np.array([[0,1,2],[0,2,1], [1,0,2], [1,2,0], [2,1,0], [2,0,1]], dtype=np.int32)
from .kalman.model cimport AbstractModel, fODFModel, MultiTensorModel
from .kalman.kalman cimport Kalman
from .alignedDirection cimport Probabilities
from libc.math cimport pow, pi, acos, floor, fabs,fmax, exp
from libc.stdio cimport printf
from bonndit.utilc.cython_helpers cimport fa, dctov, sphere2cart, r_z_r_y_r_z
from bonndit.utilc.blas_lapack cimport *
DTYPE = np.float64
cdef double _lambda_min = 0.1
###
# Given the ItoW trafo matrix and a cuboid of 8 points shape [2, 2, 2, 3] with a vector for each edge compute the
# trilinear
# interpolation

cdef int[:] permutation = np.array([0,1,2]*8, dtype=np.int32)
cdef double[:,:] neigh = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=DTYPE)
cdef int[:] minus = np.zeros((3,), dtype=np.int32)
cdef int[:,:] neigh_int = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=np.int32)
cdef int[:] best = np.zeros((4*8,), dtype=np.int32),  old_best = np.zeros((8,), dtype=np.int32)
cdef double[:,:,:] test_cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
cdef double[:] placeholder = np.zeros((3,), dtype=DTYPE)


cdef class Interpolation:

	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
		self.vector_field = vector_field
		self.vector = np.zeros((3,), dtype=DTYPE)
		self.cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
		self.floor_point = np.zeros((8, 3), dtype=DTYPE)
		self.inv_trafo = np.linalg.inv(kwargs['trafo_data'])
		self.point_index = np.zeros((4,), dtype=DTYPE)
		self.point_world = np.zeros((4,), dtype=DTYPE)
		self.best_dir  = np.zeros((3,3), dtype=DTYPE)
		self.next_dir = np.zeros((3,), dtype=DTYPE)
		self.cache = np.zeros((grid[0], grid[1], grid[2], 4 * 8), dtype=np.int32)
		self.best_ind = 0
		self.prob = probClass



	cdef void calc_cube(self,double[:] point) : # : # : # nogil:
		""" This function calculates the cube around a point.

		Parameters
		----------
		point 3 dimensional point


		Returns
		-------

		"""
		add_pointwise(self.floor_point, neigh, point)
		floor_pointwise_matrix(self.floor_point, self.floor_point)

	cdef void nearest_neigh(self,double[:] point) : # : # : # nogil:
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
	cdef void set_vector(self, int index, int j) : # : # : # nogil:
		cdef int i
		for i in range(3):
			self.vector[i] = self.vector_field[i + 1, j, int(self.floor_point[index, 0]),
		                  int(self.floor_point[index, 1]), int(self.floor_point[index, 2])]


	#@Cython.cdivision(True)
	cdef void main_dir(self, double[:] point) : # : # : # nogil:
			cdef double zero = 0
			self.point_world[:3] = point
			self.point_world[3] = 1
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,
						&self.point_index[0], 1)

			self.nearest_neigh(self.point_index[:3])
			self.set_vector(self.best_ind, 0)
			mult_with_scalar(self.next_dir, pow(self.vector_field[0, 0, int(self.floor_point[			                                                                                            self.best_ind, 0]),
			                                                   int(self.floor_point[ self.best_ind, 1]),
			                                                  int(self.floor_point[self.best_ind, 2])], 0.25), self.vector)

	cpdef int interpolate(self,double[:] point, double[:] old_dir, int r) : # : # : # nogil except *:
		pass


cdef class FACT(Interpolation):
	cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) : # : # : # nogil except *:
		cdef int i
		cdef double l, max_value
		self.point_world[:3] = point
		self.point_world[3] = 1
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)

		self.nearest_neigh(self.point_index[:3])
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
		#printf('%i \n', thread_id)<
		self.prob.calculate_probabilities(self.best_dir, old_dir)
		mult_with_scalar(self.next_dir, 1, self.prob.best_fit)
		return 0


cdef double[:] valsec = np.empty([1, ], dtype=DTYPE), \
				val= np.empty([1,], dtype=DTYPE), \
				testv = np.empty([3,], dtype=DTYPE), \
				anisoten = np.empty([15,],  dtype=DTYPE), \
				isoten = np.empty([15,],dtype=DTYPE), \
				der = np.empty([3,],  dtype=DTYPE)

cdef double[:, :] tens = np.zeros([3, 15], dtype=DTYPE)

cdef class TrilinearFODF(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
		super(TrilinearFODF, self).__init__(vector_field, grid, probClass, **kwargs)
		self.data = kwargs['data']
		self.fodf = np.zeros((16,))
		self.fodf1 = np.zeros((16,))
		self.length = np.zeros((3,))
		self.inc= np.int32(np.prod(kwargs['data'].shape[1:]))
		self.empty = np.zeros((15,))
		if kwargs['r'] == -1:
			kwargs['r'] = 2
			while True:
				kwargs['r'] += 1
				if len(np.array(
					[[i, j, k] for i in range(-int(kwargs['r']), int(kwargs['r']) + 1) for j in range(-int(kwargs['r']), int(kwargs['r']) + 1) for k in
					 range(-int(kwargs['r']), int(kwargs['r']) + 1) if np.linalg.norm(np.dot(kwargs['trafo'], np.array([i, j, k]))) <= kwargs['r']],
					dtype=np.intc)) >= 90:
					break
			self.auto = True
			self.r = kwargs['r']
		else:
			self.auto = False
			self.r = kwargs['r']


		if kwargs['sigma_2'] == 0 and kwargs['r'] != 0:
			self.sigma_2 = ((np.linalg.norm(kwargs['trafo'] @ np.array((1, 0, 0))) + np.linalg.norm(kwargs['trafo'] @ np.array((0, 1, 0))) + np.linalg.norm(kwargs['trafo'] @ np.array((0, 0, 1)))) / 3)
		else:
			self.sigma_2 = kwargs['sigma_2']
		self.neighbors = np.array(sorted([[i, j, k] for i in range(-int(kwargs['r']), 1 + int(kwargs['r'])) for j in
										  range(-int(kwargs['r']), 1 + int(kwargs['r'])) for k in
										  range(-int(kwargs['r']), 1 + int(kwargs['r']))],
										 key=lambda x: np.linalg.norm(kwargs['trafo'] @ x)), dtype=np.int32)
		if kwargs['sigma_1'] == 0 and kwargs['r'] != 0:
			skip = np.zeros(kwargs['data'].shape[1:])
			neighbors = np.array([x for x in self.neighbors if np.linalg.norm(kwargs['trafo'] @ x) <= kwargs['r']])
			var = np.zeros((len(neighbors), ) + kwargs['data'].shape[1:])

			for i,j,k in tqdm(np.ndindex(kwargs['data'].shape[1:]), total=np.prod(kwargs['data'].shape[1:])):
				if kwargs['data'][0, i,j,k] == 0.0:
					skip[i,j,k] = 1
					continue

				for index in range(neighbors.shape[0]):
					if kwargs['data'].shape[1] > i + neighbors[index, 0] >= 0 \
						and kwargs['data'].shape[2] > j + neighbors[index, 1] >= 0 \
						and kwargs['data'].shape[3] > k + neighbors[index, 2] >= 0:
						sub_vectors(isoten, kwargs['data'][1:, i,j,k], kwargs['data'][1:, i + neighbors[index, 0], j + neighbors[index, 1], k + neighbors[index, 2]])
						var[index, i,j,k] = hota_4o3d_sym_norm(isoten)
					else:
						skip[i,j,k] = 1
			#nu = np.mean(var[:, skip == 0])
			self.sigma_1 = np.median(var[:, skip == 0])**2
		else:
			self.sigma_1 = kwargs['sigma_1']
		self.best_dir_approx = np.zeros((3,3))

		self.point_diff = np.zeros((3,), dtype=DTYPE)
		self.vlinear = np.zeros((8, kwargs['data'].shape[0]))
		self.trafo = kwargs['trafo']
		self.dist = np.zeros((3,), dtype=DTYPE)
		self.r = kwargs['r']
		self.rank = kwargs['rank']
		print(self.sigma_2, self.sigma_1, self.r)


	cdef void trilinear(self, double[:] point) : # : # : # nogil except *:
		cdef int i, j, k, m,n,o
		for i in range(8):
			j = <int> floor(i / 2) % 2
			k = <int> floor(i / 4) % 2
			m = <int> point[0] + i%2
			n = <int> point[1] + j
			o = <int> point[2] + k

			dm2toc(&self.vlinear[i, 0], self.data[:, m,n,o],  self.vlinear.shape[1])
		for i in range(4):
			cblas_dscal(self.vlinear.shape[1], (1 + floor(point[2]) - point[2]), &self.vlinear[i, 0], 1)
			cblas_daxpy(self.vlinear.shape[1], (point[2] - floor(point[2])), &self.vlinear[4+i, 0], 1, &self.vlinear[i,0], 1)
		for i in range(2):
			cblas_dscal(self.vlinear.shape[1], (1 + floor(point[1]) - point[1]), &self.vlinear[i, 0], 1)
			cblas_daxpy(self.vlinear.shape[1], (point[1] - floor(point[1])), &self.vlinear[2 + i, 0], 1, &self.vlinear[i, 0], 1)
		cblas_dscal(self.vlinear.shape[1], (1 + floor(point[0]) - point[0]), &self.vlinear[0, 0], 1)
		cblas_daxpy(self.vlinear.shape[1], (point[0] - floor(point[0])), &self.vlinear[1,0], 1, &self.vlinear[0,0], 1)
		cblas_dcopy(self.vlinear.shape[1], &self.vlinear[0,0], 1, &self.fodf[0], 1)

	cdef void neigh(self, double[:] point) : # : # : # nogil except *:
		cdef double x, scale = 0, dis=0, distance =0
		cdef int i, index, p_0 = <int> point[0], p_1 = <int> point[1], p_2 = <int> point[2], pw_0, pw_1, pw_2
		self.trilinear(point)
		cblas_dscal(16,0, &self.fodf1[0],1)
		for index in range(<int> self.neighbors.shape[0]):
			pw_0 = p_0 + self.neighbors[index, 0]
			pw_1 = p_1 + self.neighbors[index, 1]
			pw_2 = p_2 + self.neighbors[index, 2]
			for i in range(3):
				self.point_diff[i] = self.neighbors[index, i] - point[i]%1
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 3,3,1, &self.trafo[0,0], 3, &self.point_diff[0], 1, 0, &self.dist[0], 1)
			distance = cblas_dnrm2(3, &self.dist[0], 1)
			if distance > self.r or (index>27 and self.auto):
				break
			if self.data.shape[1] > pw_0 >= 0 and self.data.shape[2] > pw_1 >= 0 and self.data.shape[3] > pw_2 >= 0:
				sub_vectors(self.empty, self.fodf[1:], self.data[1:, pw_0, pw_1, pw_2])
				x = hota_4o3d_sym_norm(self.empty)
				dis = exp(-(x*x)/self.sigma_1 - distance**2/self.sigma_2)
				scale += dis
				for i in range(16):
					self.fodf1[i] += dis*self.data[i, pw_0, pw_1, pw_2]
				#cblas_daxpy(16, dis, &self.data[0, pw_0, pw_1, pw_2], 1, &self.fodf1[0], 1)
		if scale > 0:
			mult_with_scalar(self.fodf, 1/scale, self.fodf1)

	cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) : # : # : # nogil except *:
	#	with gil: print(np.array(old_dir))
		# Initialize with last step. Except we are starting again.
		self.point_world[:3] = point
		self.point_world[3] = 1
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
		if r==0:
			cblas_dscal(9,0, &self.best_dir[0,0],1)
			cblas_dscal(3,0, &self.length[0],1)
		# If self.r==0: Interpolate trilinear else: calculate average over neighborhood.

		cdef int i
		if self.r==0:
			self.trilinear(self.point_index[:3])
		else:
			self.neigh(self.point_index[:3])
		if self.fodf[0] == 0:
			return -1
		#set_zero_matrix(tens)
		for i in range(3):
			hota_4o3d_sym_eval(tens[i, :], self.length[i], self.best_dir_approx[:, i])
			cblas_daxpy(15, -1, &tens[i,0], 1, &self.fodf[1], 1)
#		tijk_approx_rankk_3d_f(*self.length[0], *self.best_dir_approx[0,0], )
		approx_initial(self.length, self.best_dir_approx, tens, self.fodf[1:], self.rank, valsec, val,der, testv, anisoten, isoten)

		for i in range(3):
			if self.length[i] > 0.1:
				mult_with_scalar(self.best_dir[i], pow(self.length[i], 1/4), self.best_dir_approx[:,i])
			else:
				mult_with_scalar(self.best_dir[i], 0, self.best_dir_approx[:,i])
		self.prob.calculate_probabilities(self.best_dir, old_dir)
		cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0],1)
		return 0




cdef class Trilinear(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
		super(Trilinear, self).__init__(vector_field, grid, probClass, **kwargs)
		self.array = np.zeros((2,3), dtype=DTYPE)
		self.x_array = np.zeros((4,3), dtype=DTYPE)
		self.point = np.zeros((3,), dtype=DTYPE)
		self.dir = np.zeros((8, 3, 3), dtype=DTYPE)
		self.new_best_dir = np.zeros((3, 3), dtype=DTYPE)
		#self.cache = np.zeros((grid[0], grid[1], grid[2], 4*8), dtype=np.int32)
		self.permutation = np.zeros((16,), dtype=np.int32)
		self.not_check = np.zeros((3,2), dtype=np.int32)
		self.floor = np.zeros((3,), dtype=np.int32)



	cdef void set_array(self, int array, int index, int i) : # : # : # nogil:
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



	cpdef best_dirp(self):
		return np.array(self.best_dir)

	cpdef set_best_dirp(self, double[:,:] best_dir):
		self.best_dir = best_dir

	cpdef get_cache(self):
		return np.array(self.cache)

	cpdef get_next_dir(self):
		return np.array(self.next_dir[0:3])

	cpdef set_cache(self, int[:,:,:,:] cache):
		self.cache = cache



	cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) : # : # : # nogil except *:
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
		self.point_world[:3] = point
		self.point_world[3] = 1
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
		self.calc_cube(self.point_index[:3])
		for i in range(3):
			self.floor[i] = int(self.point_index[i])
		# Check if the best dir is initialized. If no initizialize first with the nearest neighbor. Then fit neighbors.
		for i in range(3):
			test+=norm(self.best_dir[i])
		if test==0:
			self.nearest_neigh(self.point_index[:3])
			for i in range(3):
				if point_validator(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[self.best_ind,1]),int(self.floor_point[self.best_ind, 2])], 1):
					l = pow(fabs(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),int(self.floor_point[self.best_ind, 1]), int(self.floor_point[self.best_ind,2])]), 1 / 4)
				else:
					l = 0
				self.set_vector(self.best_ind, i)
				mult_with_scalar(self.best_dir[i], l, self.vector)

		if sum_c_int(self.cache[self.floor[0], self.floor[1], self.floor[2],:]) != 0:
			#print('hier wird permutiert')
			self.permute(self.point_index[:3])
		else:
			con = self.kmeans(self.point_index[:3])
			self.permute(self.point_index[:3])
		#else:
		#	self.permute(point)
		if con:

			for i in range(3):
				self.point[i] = self.point_index[i]%1

			for i in range(3):
				# interpolate in x direction
				for j in range(4):
					mult_with_scalar(self.array[0], self.point[0], self.cuboid[4+j, i])
					mult_with_scalar(self.array[1], 1-self.point[0], self.cuboid[j, i])
					add_vectors(self.x_array[j], self.array[0], self.array[1])

				# then in y direction
				mult_with_scalar(self.x_array[0], 1-self.point[1], self.x_array[0])
				mult_with_scalar(self.x_array[2], self.point[1], self.x_array[2])
				add_vectors(self.array[0], self.x_array[0], self.x_array[2])
				mult_with_scalar(self.x_array[1], 1-self.point[1], self.x_array[1])
				mult_with_scalar(self.x_array[3], self.point[1], self.x_array[3])
				add_vectors(self.array[1], self.x_array[1], self.x_array[3])

				# then z direction
				mult_with_scalar(self.array[0], 1-self.point[2], self.array[0])
				mult_with_scalar(self.array[1], self.point[2], self.array[1])
				add_vectors(self.best_dir[i], self.array[0], self.array[1])


			self.prob.calculate_probabilities(self.best_dir, old_dir)
			self.next_dir = self.prob.best_fit

		else:
			mult_with_scalar(self.next_dir, 0, self.prob.best_fit)
		#TODO change back
		return con

#	### TODO is here a better way?
	cdef void permute(self, double[:] point) : # : # : # nogil except *:
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
		cdef int i, index, k, l, z = 1
		cdef double exponent = 0
		for index in range(8):
			for i in range(3):
				if point_validator(self.vector_field[0, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]), int(self.floor_point[index, 1]), int(self.floor_point[index, 2])], 1):
					exponent = pow(fabs(self.vector_field[0,  permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]), int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]), 1/4)
				else:
					exponent = 0
				for k in range(3):
					placeholder[k] = self.vector_field[1 + k, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]
				for k in range(3):
					self.cuboid[index,i,k] = exponent * self.cache[int(point[0]), int(point[1]),int(point[2]), index*4 + i + 1] * self.vector_field[1 +k, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]


	cdef void set_new_poss(self) : # : # : # nogil except *:
		cdef int i,j,k
		for i in range(8):
			set_zero_matrix(self.cuboid[i])
		for i in range(8):
			for j in range(3):
				for k in range(3):
					self.cuboid[i, j, k] = best[4*i + j + 1] * test_cuboid[i, permute_poss[best[4*i], j], k]

	cdef int kmeans(self, double[:] point) : # : # : # nogil except *:
		cdef int i, j, k, l, max_try=0, best_min=0
		cdef double exponent = 0, best_angle=0, min_angle=0, con=0, test_angle=0
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

				for k in range(3):
					test_cuboid[i,j,k] = exponent *  self.vector_field[1 + k, j, int(self.floor_point[i, 0]),int(self.floor_point[i, 1]),int(self.floor_point[i, 2])]
		#for i in range(8):
		# Init best dir. Since it won't work good otherwise....
	#	for j in range(3):
	#		add_vectors(self.best_dir[j], self.best_dir[j], test_cuboid[7, j])
		while True:
			con = 0
			max_try += 1
			# each corner
			for i in range(8):
				min_angle = 0
				for j in range(6):
					test_angle=0
					set_zero_vector_int(minus)
					for k in range(3):
						ang = angle_deg(self.best_dir[k], test_cuboid[i, permute_poss[j,k]])
						if ang > 90:
							mult_with_scalar(placeholder, -1, test_cuboid[i, permute_poss[j, k]])
							minus[k] = -1
						else:
							mult_with_scalar(placeholder, 1, test_cuboid[i, permute_poss[j, k]])
							minus[k] = 1
						sub_vectors(placeholder, placeholder, self.best_dir[k])
						test_angle += norm(placeholder)
					if min_angle == 0 or test_angle < min_angle:
						min_angle = test_angle
					#	with gil: print(test_angle, min_angle)
						best[4*i] = j
						for u in range(3):
							best[4*i + u + 1] = minus[u]

		#	set_zero_matrix(self.best_dir)
		#	for i in range(8):
					for u in range(3):
						if norm(self.best_dir[u]) == 0:
							mult_with_scalar(placeholder, best[4*i+ 1 +u]/8, test_cuboid[i, permute_poss[best[4*i], u]])
							add_vectors(self.best_dir[u], self.best_dir[u], placeholder)
			for i in range(8):
				con += fabs(best[4*i] - old_best[i])
				old_best[i] = best[4*i]
			if con == 0:
				con = 1
				for i in range(32):
					self.cache[self.floor[0], self.floor[1], self.floor[2], i] = int(best[i])
				#self.set_new_poss()
				break

			if max_try == 1000:
				con = 0
				#with gil: print('I do not converge')
				break
		set_zero_matrix(self.best_dir)
		return int(con)

cdef class UKF(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
		super(UKF, self).__init__(vector_field, grid, probClass, **kwargs)
		self.mean = np.zeros((kwargs['dim_model'],), dtype=np.float64)
		self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
		self.P = np.zeros((kwargs['dim_model'],kwargs['dim_model']), dtype=np.float64)
		self.y = np.zeros((kwargs['data'].shape[3],), dtype=np.float64)
		if kwargs['baseline'] != "" and kwargs['model'] != 'fodf':
			self.data = kwargs['data']/kwargs['baseline'][:,:,:,np.newaxis]
		else:
			self.data = kwargs['data']
		if kwargs['model'] == 'fodf':
			self._model = fODFModel(vector_field=vector_field, **kwargs)
		else:
			self._model = MultiTensorModel(**kwargs)
		self._kalman = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model)

	cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) : # : # : # nogil except *:
		self.point_world[:3] = point
		self.point_world[3] = 1
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
		cdef int z, i, info = 0
		# Interpolate current point
		self._kalman.linear(self.point_index[:3], self.y, self.mlinear, self.data)
		# If we are at the seed. Initialize the Kalmanfilter
		if restart == 0:
			self._model.kinit(self.mean, self.point_index[:3], old_dir, self.P, self.y)
		# Run Kalmannfilter
		info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
		return self.select_next_dir(info, old_dir)

	cdef int select_next_dir(self, int info, double[:] old_dir):
		return info

cdef class UKFFodf(UKF):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		super(UKFFodf, self).__init__(vector_field, grid, prob, **kwargs)

#	cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) : # : # : # nogil except *:
#		self.point_world[:3] = point
#		self.point_world[3] = 1
#		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
#		cdef int i, info = 0
#		# Interpolate current point
#		self._kalman.linear(self.point_index[:3], self.y, self.mlinear, self.data)
#		# If we are at the seed. Initialize the Kalmanfilter
#		if restart == 0:
#			#with gil:
#			self._model.kinit(self.mean, self.point_index[:3], old_dir, self.P, self.y)
#		#	for i in range(10):
#			#	print(np.array(self.y))
#		#		info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
#			#	print(np.array(self.P))
#		# Run Kalmannfilter
#
#		info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
#		# Order directions by length an
	cdef int select_next_dir(self, int info, double[:] old_dir):
		if info != 0:
			return info


		for i in range(self._model.num_tensors):
			if cblas_dnrm2(3, &self.mean[4*i], 1) != 0:
				cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[4*i], 1), &self.mean[4*i], 1)
				if cblas_ddot(3, &self.mean[4*i], 1, &old_dir[0],1) < 0:
					cblas_dscal(3, -1, &self.mean[4*i], 1)
				self.mean[4*i+3] = max(self.mean[4*i+3],_lambda_min)
			else:
				cblas_dscal(3, cblas_dnrm2(3, &self.mean[4 * i], 1), &self.mean[4 * i], 1)
				self.mean[4 * i + 3] = max(self.mean[4 * i + 3], _lambda_min)


		for i in range(self._model.num_tensors):
			dctov(&self.mean[4*i], self.best_dir[i])
			if self.mean[4 * i + 3] > 0.1:
				#print(self.mean[4 * i + 3])
				cblas_dscal(3,pow(self.mean[4 * i + 3], 0.25), &self.best_dir[i,0],1)
			else:
				cblas_dscal(3,0, &self.best_dir[i,0],1)

		self.prob.calculate_probabilities(self.best_dir, old_dir)
		self.next_dir = self.prob.best_fit

		return info


cdef class UKFMultiTensor(UKF):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
		super(UKFMultiTensor, self).__init__(vector_field, grid, probClass, **kwargs)


		#cblas_dcopy(self.mean.shape[0], &self.mean[0], 1, &self.tmpmean[0], 1)
	cdef int select_next_dir(self, int info, double[:] old_dir):
		for i in range(self._model.num_tensors):
			if cblas_dnrm2(3, &self.mean[5*i], 1) != 0:
				cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[5*i], 1), &self.mean[5*i], 1)
			if cblas_ddot(3, &self.mean[5*i], 1, &old_dir[0],1) < 0:
				cblas_dscal(3, -1, &self.mean[5*i], 1)
			self.mean[5*i+3] = max(self.mean[5*i+3],_lambda_min)
			self.mean[5*i+4] = max(self.mean[5*i+4],_lambda_min)

		# Use alw
		for i in range(self._model.num_tensors):
			if fa(self.mean[5*i + 3],self.mean[5*i + 4],self.mean[5*i + 4]) > 0.15:
				self.best_dir[i] = self.mean[5*i: 5*i + 3]
			else:
				self.mean[5 * i + 4] = min(self.mean[5*i + 3], self.mean[5*i + 4])
				set_zero_vector(self.best_dir[i])
		self.prob.calculate_probabilities(self.best_dir, old_dir)
		self.next_dir = self.prob.best_fit
		return info









