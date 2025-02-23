#cython: language_level=3, boundscheck=True, wraparound=True, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True, profile=True
from bonndit.directions.fodfapprox import approx_all_spherical
from bonndit.utilc.blas_lapack cimport *

from bonndit.utilc.hota cimport hota_4o3d_sym_eval, hota_6o3d_sym_eval, hota_6o3d_hessian_sh
from bonndit.utilc.quaternions cimport *
from bonndit.utilc.cython_helpers cimport special_mat_mul, rot2zyz, orthonormal_from_sphere, dinit, sphere2world, ddiagonal, world2sphere, sphere2cart, cart2sphere

from bonndit.utilc.watsonfitwrapper cimport *
from bonndit.utilc.trilinear cimport bilinear

from scipy.optimize import least_squares
import numpy as np
import os
from libc.math cimport pow, log, sqrt, pi

from bonndit.utilc.hota cimport hota_6o3d_sym_norm

DTYPE=np.float64
dirname = os.path.dirname(__file__)



cdef class AbstractModel:
	"""
	Abstract Model class for a Kalman filter: Needed for prediction of new observation from a given state. As well as
	initial setting of the state
	"""
	def __cinit__(self, **kwargs):
		self.MEASUREMENT_NOISE =  np.zeros((kwargs['data'].shape[3],kwargs['data'].shape[3]), dtype=np.float64)
		self.PROCESS_NOISE =  np.zeros((kwargs['dim_model'],kwargs['dim_model']), dtype=np.float64)
		self._lambda_min = 0.01
		self.num_tensors = 0
		self.GLOBAL_TENSOR_UNPACK_VALUE = 0.000001
		if type(kwargs['process noise']).__module__ == np.__name__:
			ddiagonal(&self.PROCESS_NOISE[0, 0], kwargs['process noise'], self.PROCESS_NOISE.shape[0],
					  self.PROCESS_NOISE.shape[1])
		if type(kwargs['measurement noise']).__module__ == np.__name__:
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], kwargs['measurement noise'], self.MEASUREMENT_NOISE.shape[0],
					  self.MEASUREMENT_NOISE.shape[1])

	cdef void normalize(self, double[:] m, double[:] v, int incr): #nogil except *:
		pass

	cdef void single_predicton(self, double[:,:] observations, double[:,:] sigma_points, int i, int j) except *:
		pass

	cdef void predict_new_observation(self, double[:,:] observations, double[:,:] sigma_points) except *: #nogil except *:
		cdef int i, j
		cblas_dscal(observations.shape[1] * observations.shape[0], 0, &observations[0, 0], 1)
		for i in range(self.num_tensors):
			for j in range(sigma_points.shape[1]):
				self.single_predicton(observations, sigma_points, i ,j)

	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y) except *:
		pass
	cdef void constrain(self, double[:,:] X) except *: #nogil except *:
		pass




cdef class fODFModel(AbstractModel):
	"""
	Classical Low-rank model, describes a fODF as
	\[ \mathcal{T} = sum_{i=0}^r \lambda_i v_i^{\otimes l} , \]
	where r describes the number of fibers and l the tensor order. Only order 4 and 6 is available - could be
	easily extended.
	TODO: Should be changed to $\lambda * v$ to remove the constrained - I assume this violates the Gaussian assumptions...
	State space is
	\[ X = \left[ \lambda_1 , v_{1_1}, v_{1_2}, v_{1_3}, \cdots ,  \lambda_r , v_{r_1}, v_{r_2}, v_{r_3} \right] , \]
	where $\lambda$ is length and $v \in \mathbb{S}^2$.
	"""
	def __cinit__(self, **kwargs):
		super(fODFModel, self).__init__(**kwargs)
		self.m = np.zeros((3,))
		self.vector_field = kwargs['vector_field']
		self.res = np.zeros((15 if kwargs['order'] == 4 else 28,))
		self.order = kwargs['order']
		if len(kwargs['process noise']) > 1:
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.005,0.005,0.005,0.1]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if len(kwargs['measurement noise']):
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.006]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])
		self.num_tensors = <int> (kwargs['dim_model'] / 4)


	cdef void normalize(self, double[:] m, double[:] v, int inc): #nogil except *:

		if cblas_dnrm2(3, &v[0], inc) != 0:
			cblas_dcopy(3, &v[0], inc, &m[0], 1)
			cblas_dscal(3, 1/cblas_dnrm2(3, &v[0],inc), &m[0], 1)

	cdef void single_predicton(self, double[:,:] observations, double[:,:] sigma_points, int i, int j) except *: #nogil except *:
		cdef double lam
		self.normalize(self.m, sigma_points[i * 4: i * 4 + 3, j], sigma_points.shape[1])
		lam = max(sigma_points[i*4 + 3, j], 0.01)
		if self.order == 4:
			hota_4o3d_sym_eval(self.res, lam, self.m)
		else:
			hota_6o3d_sym_eval(self.res, lam, self.m)
		cblas_daxpy(observations.shape[0],1,&self.res[0], 1, &observations[0,j], observations.shape[1])



	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y) except *:
		"""
		Calculates angle between all possible directions and
		"""
		cdef int i
		cdef double[:] Pv = np.array([0.01])
		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])
		for i in range(self.num_tensors):
			mean[i*4: i*4 +3] = self.vector_field[1:,i, <int> point[0], <int> point[1], <int> point[2]]
			mean[i*4 + 3] = self.vector_field[0,i, <int> point[0], <int> point[1], <int> point[2]]


	cdef void constrain(self, double[:,:] X) except *: #nogil except *:
		cdef int i, j, n = X.shape[0]//4
		for i in range(X.shape[1]):
			for j in range(n):
				if cblas_dnrm2(3,&X[4*j, i],X.shape[1]) != 0:
					cblas_dscal(3, 1/cblas_dnrm2(3,&X[4*j, i],X.shape[1]),&X[4*j, i],X.shape[1])
				X[j * 4 + 3, i] = max(X[j * 4 + 3, i], self._lambda_min)


cdef class WatsonModel(AbstractModel):
	"""
	Uses Wason model, describes a fODF as
	\[ \mathcal{T} = sum_{i=0}^r \lambda_i W \left( x_i, \kappa_i \right) \star k \]
	where r describes the number of fibers, $W$ a Watson distribution, with parameters $x_i$ direction and $\kappa_i$
	concentration parameter. Additionally, we fold with $k$ to account for the CSD model.
	Only order 4 and 6 is available - could be easily extended.
	State space is
	\[ X = \left[ \kappa_1, \lambda_1 , v_{1_1}, v_{1_2}, v_{1_3}, \cdots , \kappa_r, \lambda_r , v_{r_1}, v_{r_2}, v_{r_3} \right] , \]
	where $\lambda$ is length and $v \in \mathbb{S}^2$.
	"""
	def __cinit__(self, **kwargs):
		super(WatsonModel, self).__init__(**kwargs)
		self.m = np.zeros((3,))
		self.vector_field = kwargs['vector_field']
		self.res = np.zeros((15 if kwargs['order'] == 4 else 28,))
		self.order = kwargs['order']
		if kwargs['order'] == 4:
			self.rank_1_rh_o4 = np.array([2.51327412, 1.43615664, 0.31914592], dtype=DTYPE)
		else:
			self.rank_1_rh_o4 = np.array([1.7951958 , 1.1967972 , 0.43519898, 0.06695369], dtype=DTYPE)

		self.angles = np.zeros((3,), dtype=DTYPE)
		self.rot_pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
		self.pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
		self.dipy_v = np.zeros((15 if kwargs['order'] == 4 else 28,), dtype=DTYPE)
		if type(kwargs['process noise']).__module__ != np.__name__:
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.5,0.1,0.01, 0.01, 0.01]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if type(kwargs['measurement noise']).__module__ != np.__name__:
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.06]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])
		self.num_tensors = <int> (kwargs['dim_model'] / 5)


	cdef void normalize(self, double[:] m, double[:] v, int inc): #nogil except *:
		#no need to normalize since euler angle


		if cblas_dnrm2(3, &v[2], inc) != 0:
			cblas_dcopy(3, &v[2], inc, &m[0], 1)
			cblas_dscal(3, 1/cblas_dnrm2(3, &v[2],inc), &m[0], 1)
			self.m[0] *= -1
			self.m[2] *= -1


	cdef double sh_norm(self, double[:] v):
		return v[0]*sqrt(1/(4*pi)) * self.rank_1_rh_o4[0] + \
				v[3] * 1/2 * sqrt(5/pi) * self.rank_1_rh_o4[1] + \
				v[10] * 3/16 * sqrt(1/pi) * (35-30+3) * self.rank_1_rh_o4[2]


	cdef void single_predicton(self, double[:,:] observations, double[:,:] sigma_points, int i, int j) except *: # nogil except *:
		cdef double lam, kappa, div
		self.normalize(self.m, sigma_points[i * 5: i * 5 + 5, j], sigma_points.shape[1])
		lam = max(sigma_points[i*5 + 1, j], 0.01)
		kappa = exp(sigma_points[i*5, j])
		cart2sphere(self.angles[1:], self.m)
		cblas_dscal(self.dipy_v.shape[0], 0, &self.dipy_v[0], 1)
		c_sh_watson_coeffs(kappa, &self.dipy_v[0], self.order)
		self.dipy_v[0] *= self.rank_1_rh_o4[0]
		self.dipy_v[3] *= self.rank_1_rh_o4[1]
		self.dipy_v[10] *= self.rank_1_rh_o4[2]
		if self.order == 6:
			self.dipy_v[21] *= self.rank_1_rh_o4[3]

		c_sh_rotate_real_coef_fast(&observations[0,j], observations.shape[1], &self.dipy_v[0],
								   1, self.order, &self.angles[0])
		cblas_dscal(observations.shape[0], lam , &observations[0,j], observations.shape[1])


	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y) except *:
		"""
		Calculates angle between all possible directions and
		"""
		cdef int i, j
		cdef double[:] Pv = np.array([0.01], dtype=DTYPE)
		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])
		for i in range(self.vector_field.shape[1]):

			mean[i*5: i*5+5]= self.vector_field[:,i, <int> point[0], <int> point[1], <int> point[2]]

			mean[i*5] = log(mean[i*5])



	cdef void constrain(self, double[:,:] X) except *: # nogil except *:
		cdef int i, j, n = X.shape[0]//5
		for i in range(X.shape[1]):
			for j in range(n):
				X[j * 5, i] = min(max(exp(X[j * 5 , i]), log(self._lambda_min)), log(80))
				X[j * 5 + 1, i] = max(X[j * 5 + 1, i], self._lambda_min)



cdef class BinghamModel(WatsonModel):
	"""
	Uses Bingham model, describes a fODF as
	\[ \mathcal{T} = sum_{i=0}^r \lambda_i B \left( x_i, \kappa_i \right) \star k \]
	where r describes the number of fibers, $B$ a Bingham distribution, with parameters $r_i$ rotation in angle and $\kappa_i$
	concentration parameter. Additionally, we fold with $k$ to account for the CSD model.
	Only order 4 and 6 is available - could be easily extended.
	State space is
	\[ X = \left[ \kappa_1, \lambda_1 , \omega_{1_1}, v_{1_2}, v_{1_3}, \cdots , \kappa_r, \lambda_r , v_{r_1}, v_{r_2}, v_{r_3} \right] , \]
	where $\lambda$ is length and $v \in \mathbb{S}^2$.
	"""
	def __cinit__(self, **kwargs):
		super(BinghamModel, self).__init__(**kwargs)
		self.lookup_kappa_beta_table = np.load(dirname + '/kappa_beta_lookup.npy')

		lookup_table1 =  np.load(dirname + '/bingham_normalized_o6_new.npz')
		self.sh = np.zeros((29,))
		self.lookup_table1 = lookup_table1['arr_0']

		self.num_tensors = <int> (kwargs['dim_model'] / 6)
		if type(kwargs['process noise']).__module__ != np.__name__:
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.01, 0.01,0.01,0.001, 0.001, 0.001]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if type(kwargs['measurement noise']).__module__ != np.__name__:
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.04]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])

	cdef void set_angle_for_prediction(self, double[:] params, int params_cols):
		cblas_dcopy(3, &params[0], params_cols, &self.angles[0], 1)


	cdef void single_predicton(self, double[:,:] observations, double[:,:] sigma_points, int i, int j): # nogil except *:
		cdef double lam, kappa, beta
		lam = max(sigma_points[i*self.num_parameter, j], 0.01)
		kappa = max(min(exp(sigma_points[i*self.num_parameter + 1, j]), 89), 0.1)
		beta = max(min(exp(sigma_points[i*self.num_parameter + 2, j]), kappa), 0.1)
		self.set_angle_for_prediction(sigma_points[i*self.num_parameter+3:(i+1)*self.num_parameter,j], sigma_points.shape[1])
		bilinear(self.sh, self.lookup_table1[<int> (kappa * 10) : <int> (kappa * 10) + 2, <int> (beta * 10): <int> (beta * 10) + 2, :], 10*kappa, 10*beta)
		c_sh_rotate_real_coef_fast(&observations[0,j], observations.shape[1], &self.sh[0], 1, self.order, &self.angles[0])
		cblas_dscal(observations.shape[0], lam , &observations[0,j], observations.shape[1])


	cdef void lookup_kappa_beta(self, double[:] ret, double e1, double e2):
		cdef int i, min_idx = -1
		cdef double min_dist = 0, dist=0
		for i in range(self.lookup_kappa_beta_table.shape[0]):
			dist = (self.lookup_kappa_beta_table[i, 0] - e1)**2 + (self.lookup_kappa_beta_table[i, 1] - e2)**2
			if min_idx == -1 or dist < min_dist:
				min_idx = i
				min_dist = dist
		ret[0] = log(min(self.lookup_kappa_beta_table[min_idx, 2], 20))
		ret[1] = log(self.lookup_kappa_beta_table[min_idx, 3])


	cdef void set_mean(self, double[:] mean, double[:,:] basis, double[:] ev, int i, int z):
		rot2zyz(mean[i*6+3:(i+1)*6], basis)
		self.lookup_kappa_beta(mean[1: 3], ev[z], ev[(z+1) % 2])



	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y):
		"""
		Calculates angle between all possible directions and
		"""
		cdef int i
		cdef double[:] Pv = np.array([0.01,], dtype=DTYPE)
		cdef double[:] dir = np.zeros((3,))
		cdef double[:, : ,: ] out = np.zeros((4, self.vector_field.shape[1], 1))

		cdef double[:,:] hessian = np.zeros((2,2)), orth = np.zeros((3,2))
		cdef double[:,:] ortho_sys = np.zeros((3,3))
		dir1 = np.zeros((3, ))

		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])

		for i in range(self.vector_field.shape[1]):
			# calculate hessian:
			hota_6o3d_hessian_sh(hessian, orth,  y[1:], init_dir)
			print(np.array(orth))

			t, eig =  np.linalg.eig(hessian)
			#sort according eigenvalues:
			z =  0
			cblas_dscal(9, 0, &ortho_sys[0,0], 1)
			ortho_sys[:, 2] = init_dir

			cblas_daxpy(3, eig[0,0], &orth[0,0], 2, &ortho_sys[0,z], 3)
			cblas_daxpy(3, eig[1,0], &orth[0,1], 2, &ortho_sys[0,z], 3)
			cblas_daxpy(3, eig[0,1], &orth[0,0], 2, &ortho_sys[0,(z+1)%2], 3)
			cblas_daxpy(3, eig[1,1], &orth[0,1], 2, &ortho_sys[0,(z+1)%2], 3)
			print('ortho', np.array(eig))
			print('ortho', np.array(orth))
			print('ortho sys', np.array(ortho_sys))
			for j in range(3):
				scale = cblas_dnrm2(3, &ortho_sys[0,j], 3)
				cblas_dscal(3, 1/scale, &ortho_sys[0,j], 3)
			cblas_dscal(3, -1, &ortho_sys[0,0], 1)
			cblas_dscal(3, -1, &ortho_sys[2,0], 1)
			self.set_mean(mean, ortho_sys, t, i, z)

	cdef void constrain(self, double[:,:] X) except *: # nogil except *:
		cdef int i, j, n = X.shape[0]//6
		for i in range(X.shape[1]):
			for j in range(n):
				X[j * 6 + 1, i] = min(max(X[j * 6 + 1, i], log(0.2)), log(89))
				X[j * 6 + 2, i] = min(max(X[j * 6 + 2, i], log(0.1)), X[j * 6 + 1, i])
				X[j * 6 + 0, i] = max(X[j * 6 + 0, i], self._lambda_min)

cdef class BinghamQuatModel(BinghamModel):
	"""
	Uses Bingham model, describes a fODF as
	\[ \mathcal{T} = sum_{i=0}^r \lambda_i B \left( x_i, \kappa_i \right) \star k \]
	where r describes the number of fibers, $B$ a Bingham distribution, with parameters $r_i$ rotation in angle and $\kappa_i$
	concentration parameter. Additionally, we fold with $k$ to account for the CSD model.
	Only order 4 and 6 is available - could be easily extended.
	State space is
	\[ X = \left[ \kappa_1, \lambda_1 , \omega_{1_1}, v_{1_2}, v_{1_3}, \cdots , \kappa_r, \lambda_r , v_{r_1}, v_{r_2}, v_{r_3} \right] , \]
	where $\lambda$ is length and $v \in \mathbb{S}^2$.
	"""

	def __cinit__(self, **kwargs):
		self.num_parameter = 6
		super(BinghamQuatModel, self).__init__(**kwargs)
		self.order= kwargs["order"]

		if type(kwargs['process noise']).__module__ != np.__name__:
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.01, 0.01,0.01,0.001, 0.001, 0.001]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if type(kwargs['measurement noise']).__module__ != np.__name__:
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.04]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])


	cdef void set_mean(self, double[:] mean, double[:,:] basis, double[:] ev, int i, int z):

		cdef double[:] dir = np.zeros((3,))
		rot2zyz(dir, basis)
		ZYZ2quat(mean[i * self.num_parameter + 3:(i + 1) * self.num_parameter], dir)
		self.lookup_kappa_beta(mean[1: 3], ev[z], ev[(z + 1) % 2])

	cdef void set_angle_for_prediction(self, double[:] params, int params_cols):
		quat2ZYZ(self.angles, params)




cdef class MultiTensorModel(AbstractModel):
	def __cinit__(self, **kwargs):
		super(MultiTensorModel, self).__init__(**kwargs)
		self.m = np.zeros((3,))
		self.lam = np.zeros((3,))
		self.q = np.zeros((3,))
		self.D = np.zeros((3,3))
		self.M = np.zeros((3,3))
		self.num_tensors = <int> (kwargs['dim_model'] / 5)
		self.gradients = kwargs['gradients']
		self.c = np.zeros((kwargs['dim_model'],))
		self.baseline_signal = kwargs['b']
		self.acq_spec_const = kwargs['b0']
		self._lambda_min = 100
		if type(kwargs['process noise']).__module__ != np.__name__:
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.003,  0.003,0.003,25,25]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if type(kwargs['measurement noise']).__module__ != np.__name__:
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.02]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])

	cdef void normalize(self, double[:] m, double[:] v, int inc): # nogil except *:
		# Calculates m = v/||v||, by doing matrix operation 1/||v||*v*1 + 0*m
		#with gil: print(np.array(v))
		if cblas_dnrm2(3, &v[0],inc) != 0:
			cblas_dcopy(3, &v[0], inc, &m[0], 1)
			cblas_dscal(3, 1/cblas_dnrm2(3, &v[0],inc), &m[0], 1)



	cdef void diffusion(self, double[:,:] R, double[:] m, double[:] lambdas, double[:,:] M):
		#nogil except *:
		"""
		Calculates the diffusion Matrix for a given main diffusion direction m.

		@param R: Return Matrix
		@param m: 3 Matrix with main direction
		@param lambdas: 3 Matrix with lambda values
		@param M: 3x3 Matrix placeholder
		@return:

		"""
		M[0,0] = m[0]
		M[0,1] = m[1]
		M[0,2] = m[2]
		M[1,0] = m[1]
		M[1,1] = m[1] * m[1] / (1 + m[0]) - 1
		M[1,2] = m[1] * m[2] / (1 + m[0])
		M[2,0] = m[2]
		M[2,1] = m[1] * m[2] / (1 + m[0])
		M[2,2] = m[2] * m[2] / (1 + m[0]) - 1
		special_mat_mul(R, M, lambdas, M, self.GLOBAL_TENSOR_UNPACK_VALUE)

	cdef void single_predicton(self, double[:,:] observations, double[:,:] sigma_points, int i, int j): #1  nogil except *:
		self.normalize(self.m, sigma_points[i * 5: i * 5 + 3, j], sigma_points.shape[1])
		self.lam[0] = max(sigma_points[i * 5 + 3, j], self._lambda_min)
		self.lam[1] = max(sigma_points[i * 5 + 4, j], self._lambda_min)
		self.lam[2] = self.lam[1]
		if self.m[0] < 0:
			cblas_dscal(3, -1, &self.m[0], 1)
		self.diffusion(self.D, self.m, self.lam, self.M)
		#with gil:
		#	print(np.array(D)
		for k in range(self.gradients.shape[0]):
			cblas_dgemv(CblasRowMajor, CblasNoTrans, self.D.shape[0], self.D.shape[1], 1, &self.D[0,0], self.D.shape[1], &self.gradients[k, 0], self.gradients.shape[0], 0, &self.q[0], 1)
			observations[k,j] += 1/self.num_tensors * exp(- self.baseline_signal[k] * cblas_ddot(self.D.shape[0], &self.q[0], 1, &self.gradients[k,0],  self.gradients.shape[0]))



	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y ):
		cdef double[:] Pv = np.array([0.01])

		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])
		_, sigma, phi = world2sphere(init_dir[0], init_dir[1], init_dir[2])

		#self.linear(point, self.BASELINE_SIGNAL, self.slinear, self.basel/ine)
		x = np.array([sigma,phi,1000,sigma + np.pi/2,phi,600])
		res = least_squares(mti, x, method='lm', args=(np.array(y), self.gradients, self.num_tensors, self.GLOBAL_TENSOR_UNPACK_VALUE, self.baseline_signal),max_nfev=100)
		b = np.zeros(10)
		#print('init')
		for i in range(self.num_tensors):
			b[5*i:5*i + 3] = sphere2world(1, res.x[3*i], res.x[3*i + 1])
			b[5*i + 3:5*(i+1)] = res.x[3*i+2], res.x[3*i+2]/8
		self.c = b

		for i in range(self.num_tensors):
			dinit(5, &mean[5 * i], &self.c[5 * i], 10)
		#print(res.cost/np.linalg.norm(y))
		return True

	cdef void constrain(self, double[:,:] X): # nogil except *:
		cdef int i, j, n = X.shape[0]//5
		for i in range(X.shape[1]):
			for j in range(n):
				if cblas_dnrm2(3,&X[5*j, i],X.shape[1]) > 0:
					cblas_dscal(3, 1/cblas_dnrm2(3,&X[5*j, i],X.shape[1]),&X[5*j, i],X.shape[1])
				X[j * 5 + 3, i] = max(X[j * 5 + 3, i], self._lambda_min)
				X[j * 5 + 4, i] = max(X[j * 5 + 4, i], self._lambda_min)

cdef mti(x, y, gradients, tensor_num, GLOBAL_TENSOR_UNPACK_VALUE, b):
	z = np.copy(y)
	for i in range(tensor_num):
		orth = orthonormal_from_sphere(x[i * 3], x[i * 3 + 1])
		D = x[i * 3 + 2] * (np.outer(orth[0], orth[0])) + x[i * 3 + 2] / 8 * (np.outer(orth[1], orth[1]) + np.outer(orth[2], orth[2]))
		for j in range(gradients.shape[0]):
			z[j] -= 1 / tensor_num * np.exp(- b[j] * (gradients[j] @ D @ gradients[j].T) * GLOBAL_TENSOR_UNPACK_VALUE)
	return z




