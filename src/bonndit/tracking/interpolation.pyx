#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=True, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import Cython
from bonndit.utilc.cython_helpers cimport add_pointwise, floor_pointwise_matrix, norm, mult_with_scalar,\
	add_vectors, sub_vectors, scalar, clip, set_zero_vector, set_zero_matrix, sum_c, sum_c_int, set_zero_vector_int, \
	angle_deg, set_zero_matrix_int, point_validator
import numpy as np
import time
from .ItoW cimport Trafo
cdef int[:,:] permute_poss = np.array([[0,1,2],[0,2,1], [1,0,2], [1,2,0], [2,1,0], [2,0,1]], dtype=np.int32)
from .kalman.model cimport AbstractModel, fODFModel, MultiTensorModel
from .kalman.kalman cimport Kalman
from .alignedDirection cimport Probabilities
from libc.math cimport pow, pi, acos, floor, fabs,fmax
from libc.stdio cimport printf
from bonndit.utilc.cython_helpers cimport fa, dctov, dinit
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

	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		self.vector_field = vector_field
		self.vector = np.zeros((3,), dtype=DTYPE)
		self.cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
		self.floor_point = np.zeros((8, 3), dtype=DTYPE)
		self.best_dir  = np.zeros((3,3), dtype=DTYPE)
		self.next_dir = np.zeros((3,), dtype=DTYPE)
		self.cache = np.zeros((grid[0], grid[1], grid[2], 4 * 8), dtype=np.int32)
		self.best_ind = 0
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
			cdef double zero = 0
			self.nearest_neigh(point)
			self.set_vector(self.best_ind, 0)
			mult_with_scalar(self.next_dir, pow(self.vector_field[0, 0, int(self.floor_point[			                                                                                            self.best_ind, 0]),
			                                                   int(self.floor_point[ self.best_ind, 1]),
			                                                  int(self.floor_point[self.best_ind, 2])], 0.25), self.vector)

	cdef int interpolate(self,double[:] point, double[:] old_dir, int r) nogil except *:
		pass


cdef class FACT(Interpolation):
	cdef int interpolate(self, double[:] point, double[:] old_dir, int r) nogil except *:
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
		return 0



cdef class Trilinear(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		super(Trilinear, self).__init__(vector_field, grid, prob, **kwargs)
		self.array = np.zeros((2,3), dtype=DTYPE)
		self.x_array = np.zeros((4,3), dtype=DTYPE)
		self.point = np.zeros((3,), dtype=DTYPE)
		self.dir = np.zeros((8, 3, 3), dtype=DTYPE)
		self.new_best_dir = np.zeros((3, 3), dtype=DTYPE)
		#self.cache = np.zeros((grid[0], grid[1], grid[2], 4*8), dtype=np.int32)
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



	cdef int interpolate(self, double[:] point, double[:] old_dir, int r) nogil except *:
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

		if sum_c_int(self.cache[self.floor[0], self.floor[1], self.floor[2],:]) != 0:
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
		return 0

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
				if index > 0:
					ang = angle_deg(self.cuboid[0,i] , placeholder)
					if ang > 90:
						z=-1
					else:
						z=1
				for k in range(3):
					self.cuboid[index,i,k] = exponent * z * self.vector_field[1 +k, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]


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
						test_angle += pow(norm(placeholder),4)
					if min_angle == 0 or test_angle < min_angle:
						min_angle = test_angle
					#	with gil: print(test_angle, min_angle)
						best[4*i] = j
						for k in range(3):
							best[4*i + k + 1] = minus[k]

		#	set_zero_matrix(self.best_dir)
		#	for i in range(8):
					for j in range(3):
						if norm(self.best_dir[j]) == 0:
							mult_with_scalar(placeholder, best[4*i+ 1 +j]/8, test_cuboid[i, permute_poss[best[4*i], j]])
							add_vectors(self.best_dir[j], self.best_dir[j], placeholder)
			for i in range(8):
				con += fabs(best[4*i] - old_best[i])
				old_best[i] = best[4*i]
			if con == 0:
				con = 1
				for i in range(32):
					self.cache[self.floor[0], self.floor[1], self.floor[2], i] = int(best[i])
				self.set_new_poss()
				break

			if max_try == 1000:
				con = 0
				with gil: print('I do not converge')
				break
		set_zero_matrix(self.best_dir)
		return int(con)

cdef class UKF(Interpolation):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		super(UKF, self).__init__(vector_field, grid, prob, **kwargs)
		self.mean = np.zeros((kwargs['dim_model'],), dtype=np.float64)
		self.mlinear  = np.zeros((kwargs['dim_model'],kwargs['data'].shape[-1]), dtype=np.float64)
		self.P = np.zeros((kwargs['dim_model'],kwargs['dim_model']), dtype=np.float64)
		self.y = np.zeros((kwargs['data'].shape[-1],), dtype=np.float64)
		if kwargs['baseline'] != "" and kwargs['model'] != 'fodf':
			self.data = kwargs['data']/kwargs['baseline'][:,:,:,np.newaxis]
		else:
			self.data = kwargs['data']
		if kwargs['model'] == 'fodf':
			self._model = fODFModel(vector_field=vector_field, **kwargs)
		else:
			self._model = MultiTensorModel(**kwargs)
		self._kalman = Kalman(kwargs['data'].shape[-1], kwargs['dim_model'], self._model)

cdef class UKFFodf(UKF):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		super(UKFFodf, self).__init__(vector_field, grid, prob, **kwargs)

	cdef int interpolate(self, double[:] point, double[:] old_dir, int restart) nogil except *:
		cdef int i, info = 0
		# Interpolate current point
		self._kalman.linear(point, self.y, self.mlinear, self.data)
		# If we are at the seed. Initialize the Kalmanfilter
		if restart == 0:
			with gil:
				self._model.kinit(self.mean, point, old_dir, self.P, self.y)
		# Run Kalmannfilter

		info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
		# Order directions by length an
		if info != 0:
			return info
		for i in range(self._model.num_tensors):
			cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[4*i], 1), &self.mean[4*i], 1)
			if cblas_ddot(3, &self.mean[4*i], 1, &old_dir[0],1) < 0:
				cblas_dscal(3, -1, &self.mean[4*i], 1)
			self.mean[4*i+3] = max(self.mean[4*i+3],_lambda_min)


		if cblas_ddot(3, &self.mean[0], 1, &old_dir[0],1) < cblas_ddot(3, &self.mean[4], 1, &old_dir[0],1):
			cblas_dswap(4, &self.mean[0], 1, &self.mean[4], 1)
			for i in range(4):
				cblas_dswap(4, &self.P[i,0], 1, &self.P[i+4,4], 1)
				cblas_dswap(4, &self.P[i,4], 1, &self.P[i+4,0], 1)
		dctov(&self.mean[0], self.next_dir)
	#with gil: print('dir', np.array(self.next_dir))
		return info




cdef class UKFMultiTensor(UKF):
	def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
		super(UKFMultiTensor, self).__init__(vector_field, grid, prob, **kwargs)

	cdef int interpolate(self, double[:] point, double[:] old_dir, int restart) nogil except *:
		cdef int z, i, info = 0
		# Interpolate current point
		self._kalman.linear(point, self.y, self.mlinear, self.data)
		# If we are at the seed. Initialize the Kalmanfilter
		if restart == 0:
			with gil:
				##print(np.array(self.y))
				self._model.kinit(self.mean, point, old_dir, self.P, self.y)
		# Run Kalmannfilter
		info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
		#cblas_dcopy(self.mean.shape[0], &self.mean[0], 1, &self.tmpmean[0], 1)
		for i in range(self._model.num_tensors):
			cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[5*i], 1), &self.mean[5*i], 1)
			if cblas_ddot(3, &self.mean[5*i], 1, &old_dir[0],1) < 0:
				cblas_dscal(3, -1, &self.mean[5*i], 1)
			self.mean[5*i+3] = max(self.mean[5*i+3],_lambda_min)
			self.mean[5*i+4] = max(self.mean[5*i+4],_lambda_min)

		# Use alw
#		for i in range(self._model.num_tensors):
#			self.best_dir[3*i: 3*(i+1)] = self.mean[5*i: 5*i + 3]
#		self.prob.calculate_probabilities(self.best_dir, old_dir)
#		self.next_dir = self.prob.best_fit
	#	if self._model.num_tensors == 1:
	#		dctov(&self.mean[0], self.next_dir)
	#	if self._model.num_tensors == 2:
	#		if cblas_ddot(3, &self.mean[0], 1, &old_dir[0],1) < cblas_ddot(3, &self.mean[5], 1, &old_dir[0],1):
	#			cblas_dswap(5, &self.mean[0], 1, &self.mean[5], 1)
	#			for i in range(5):
	#				cblas_dswap(5, &self.P[i,0], 1, &self.P[i+5,5], 1)
	#				cblas_dswap(5, &self.P[i,5], 1, &self.P[i+5,0], 1)
	#		dctov(&self.mean[0], self.next_dir)
	#	if self._model.num_tensors == 3:
	#		dot1 = cblas_ddot(3, &self.mean[0], 1, &old_dir[0], 1)
	#		dot2 = cblas_ddot(3, &self.mean[5], 1, &old_dir[0], 1)
	#		dot3 = cblas_ddot(3, &self.mean[10], 1, &old_dir[0], 1)
	#		if dot1 < dot3:
	#			if dot2 > dot3:
	#				#turn second and third direction
	#			elif dot2 < dot1:
	#				# turn first and second direction
	#		else:
	#			# turn first and third direction
	#			if dot2 > dot3:
	#				# turn second and third direction
	#				pass
	#			elif dot2 < dot1:
	#				# turn first and second direction


		#		#with gil: print('dir', np.array(self.next_dir))
		z = 0
		for i in range(self._model.num_tensors):
			if fa(self.mean[5*i + 3],self.mean[5*i + 4],self.mean[5*i + 4]) < 1/2:
				z += 1
		if z == self._model.num_tensors:
			return 1
		return info









