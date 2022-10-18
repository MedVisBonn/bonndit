#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
from bonndit.utilc.blas_lapack cimport *
from bonndit.utilc.structures cimport order8_mult, order4_mult
from bonndit.utilc.hota cimport hota_4o3d_sym_eval, hota_8o3d_sym_eval
from bonndit.utilc.cython_helpers cimport special_mat_mul, orthonormal_from_sphere, dinit, sphere2world, ddiagonal, world2sphere
from scipy.optimize import least_squares
import numpy as np
from libc.math cimport pow


cdef class AbstractModel:
	def __cinit__(self, **kwargs):
		self.MEASUREMENT_NOISE =  np.zeros((kwargs['data'].shape[3],kwargs['data'].shape[3]), dtype=np.float64)
		self.PROCESS_NOISE =  np.zeros((kwargs['dim_model'],kwargs['dim_model']), dtype=np.float64)
		self._lambda_min = 0
		self.num_tensors = 0
		self.GLOBAL_TENSOR_UNPACK_VALUE = 0.000001
		if kwargs['process noise'] != "":
			ddiagonal(&self.PROCESS_NOISE[0, 0], kwargs['process noise'], self.PROCESS_NOISE.shape[0],
					  self.PROCESS_NOISE.shape[1])
		if kwargs['measurement noise'] != "":
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], kwargs['measurement noise'], self.MEASUREMENT_NOISE.shape[0],
					  self.MEASUREMENT_NOISE.shape[1])

	cdef void normalize(self, double[:] m, double[:] v, int incr) nogil except *:
		pass
	cdef void predict_new_observation(self, double[:,:] observations, double[:,:] sigma_points) nogil except *:
		pass
	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y):
		pass
	cdef void constrain(self, double[:,:] X) nogil except *:
		pass




cdef class fODFModel(AbstractModel):
	def __cinit__(self, **kwargs):
		super(fODFModel, self).__init__(**kwargs)
		self.m = np.zeros((3,))
		vector_field = kwargs['vector_field']
		self.res = np.zeros((15 if kwargs['order'] == 4 else 45,))
		self.order = kwargs['order']
		if kwargs['process noise'] == "":
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.005,0.005,0.005,0.1]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if kwargs['measurement noise'] == "":
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], 0.006*np.array(order8_mult if self.order ==8 else order4_mult), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])
	#	else:
	#		ddiagonal(&self.MEASUREMENT_NOISE[0, 0], float(kwargs['measurement noise'])*np.array(order8_mult if kwargs['order'] == 8 else order4_mult), self.MEASUREMENT_NOISE.shape[0],
	#			  self.MEASUREMENT_NOISE.shape[1])
		self.num_tensors = <int> (kwargs['dim_model'] / 4)
		self.vector_field = kwargs['vector_field']


	cdef void normalize(self, double[:] m, double[:] v, int inc) nogil except *:
		# Calculates m = v/||v||, by doing matrix operation 1/||v||*v*1 + 0*m
		#with gil: print(np.array(v))
		#WHAT TODO else?
		if cblas_dnrm2(3, &v[0], inc) != 0:
			cblas_dcopy(3, &v[0], inc, &m[0], 1)
			cblas_dscal(3, 1/cblas_dnrm2(3, &v[0],inc), &m[0], 1)

	cdef void predict_new_observation(self, double[:,:] observations, double[:,:] sigma_points) nogil except *:
		cdef int number_of_tensors = int(sigma_points.shape[0]/4)
		cdef int i, j
		cdef double lam
		cblas_dscal(observations.shape[1] * observations.shape[0], 0, &observations[0, 0], 1)
		for i in range(number_of_tensors):
			for j in range(sigma_points.shape[1]):
				self.normalize(self.m, sigma_points[i * 4: i * 4 + 3, j], sigma_points.shape[1])
				lam = max(sigma_points[i*4 + 3, j], 0.01)
				if self.order == 4:
					hota_4o3d_sym_eval(self.res, lam, self.m)
				else:
					hota_8o3d_sym_eval(self.res, lam, self.m)
				cblas_daxpy(observations.shape[0],1,&self.res[0], 1, &observations[0,j], observations.shape[1])
		#with gil:
		#	print(np.array(observations))


	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y):
		"""
		Calculates angle between all possible directions and
		"""
		cdef int i
		cdef double[:] dot = np.zeros(self.vector_field.shape[1])
		cdef double[:] Pv = np.array([0.01])
		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])
		for i in range(self.vector_field.shape[1]):
			dot[i] = cblas_ddot(3, &self.vector_field[1,i, <int> point[0], <int> point[1], <int> point[2]], 1, &init_dir[0],1)
		dot1 = sorted(dot, key=lambda x: abs(x))
		for i in range(self.vector_field.shape[1]):
			for j in range(self.vector_field.shape[1]):
				if dot[i] == dot1[j]:
					mean[j*4: j*4 +3] = self.vector_field[1:,i, <int> point[0], <int> point[1], <int> point[2]]
					if dot[i] < 0:
						cblas_dscal(3, -1, &mean[4*j], 1)
					mean[j*4 + 3] = self.vector_field[0,i, <int> point[0], <int> point[1], <int> point[2]]

	cdef void constrain(self, double[:,:] X) nogil except *:
		cdef int i, j, n = X.shape[0]//4
		for i in range(X.shape[1]):
			for j in range(n):
				if cblas_dnrm2(3,&X[4*j, i],X.shape[1]) != 0:
					cblas_dscal(3, 1/cblas_dnrm2(3,&X[4*j, i],X.shape[1]),&X[4*j, i],X.shape[1])

				X[j * 4 + 3, i] = max(X[j * 4 + 3, i], self._lambda_min)


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
		if kwargs['process noise'] == "":
			ddiagonal(&self.PROCESS_NOISE[0, 0], np.array([0.003,  0.003,0.003,25,25]), self.PROCESS_NOISE.shape[0],
				  self.PROCESS_NOISE.shape[1])
		if kwargs['measurement noise'] == "":
			ddiagonal(&self.MEASUREMENT_NOISE[0, 0], np.array([0.02]), self.MEASUREMENT_NOISE.shape[0],
				  self.MEASUREMENT_NOISE.shape[1])

	cdef void normalize(self, double[:] m, double[:] v, int inc) nogil except *:
		# Calculates m = v/||v||, by doing matrix operation 1/||v||*v*1 + 0*m
		#with gil: print(np.array(v))
		if cblas_dnrm2(3, &v[0],inc) != 0:
			cblas_dcopy(3, &v[0], inc, &m[0], 1)
			cblas_dscal(3, 1/cblas_dnrm2(3, &v[0],inc), &m[0], 1)



	cdef void diffusion(self, double[:,:] R, double[:] m, double[:] lambdas, double[:,:] M) nogil except *:
		"""
		Calculates the diffusion Matrix for a given main diffusion direction m.

		@param R: Return Matrix
		@param m: 3 Matrix with main direction
		@param lambdas: 3 Matrix with lambda values
		@param M: 3x3 Matrix placeholder
		@return:
asdfas
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

	cdef void predict_new_observation(self, double[:,:] observations, double[:,:] sigma_points, )  nogil except *:
		r"""
		Predicts new observation for a given set of sigma points according to the signal model
		.. math::
			 s_i \coloneqq


		Parameters
		----------
		p
		q

		outer_q
		outer_p
		outer_m
		observations
		sigma_points
		u
		baseline_signal
		acq_spec_const

		Returns
		-------

		"""
		cdef int number_of_tensors = int(sigma_points.shape[0]/5)
		cblas_dscal(observations.shape[1]*observations.shape[0], 0, &observations[0,0], 1)
		cdef int  i, j, k
		for i in range(number_of_tensors):
			for j in range(sigma_points.shape[1]):
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
					observations[k,j] += 1/number_of_tensors * exp(- self.baseline_signal[k] * cblas_ddot(self.D.shape[0], &self.q[0], 1, &self.gradients[k,0],  self.gradients.shape[0]))



	cdef bint kinit(self, double[:] mean, double[:] point, double[:] init_dir, double[:,:] P, double[:] y ):
		cdef double[:] Pv = np.array([0.01])

		ddiagonal(&P[0,0], Pv, P.shape[0], P.shape[1])
		_, sigma, phi = world2sphere(init_dir[0], init_dir[1], init_dir[2])

		#self.linear(point, self.BASELINE_SIGNAL, self.slinear, self.basel/ine)

		x = np.array([sigma,phi,1000,sigma + np.pi/2,phi,600])

		res = least_squares(mti, x, method='lm', args=(np.array(y)[np.array(self.baseline_signal) < 1300], self.gradients, self.num_tensors, self.GLOBAL_TENSOR_UNPACK_VALUE, self.baseline_signal),max_nfev=100)

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

	cdef void constrain(self, double[:,:] X) nogil except *:
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
		l = 0

		for j in range(gradients.shape[0]):
			if b[j] <1300:
				z[l] -= 1 / tensor_num * np.exp(- b[j] * (gradients[j] @ D @ gradients[j].T) * GLOBAL_TENSOR_UNPACK_VALUE)
				l += 1
	return z




