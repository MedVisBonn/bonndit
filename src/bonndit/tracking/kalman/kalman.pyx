#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import ctypes
import time

from bonndit.utilc.cython_helpers cimport ddiagonal,  dm2toc, dinit, sub_pointwise, special_mat_mul,inverse
from bonndit.utilc.blas_lapack cimport *
from bonndit.utilc.quaternions cimport *
import numpy as np

from libc.math cimport fabs, floor, pow

cdef class Kalman:
	def __cinit__(self, int dim_data, int dim_model, model):
		self.X = np.zeros((dim_model, 2*dim_model+1), dtype=np.float64)
		self.X2 = np.zeros((dim_model,2*dim_model+1), dtype=np.float64)
		self._model = model
		self.weights = np.zeros((2*dim_model+1,), dtype=np.float64)
		self.weights_diag = np.zeros((2*dim_model+1, 2*dim_model+1), dtype=np.float64)
		self.pred_X_mean = np.zeros((dim_model, ), dtype=np.float64)
		self.pred_Y_mean =np.zeros((dim_data, ), dtype=np.float64)
		self.y_diff = np.zeros((dim_data, ), dtype=np.float64)
		self.P_xx = np.zeros((dim_model,dim_model), dtype=np.float64)
		self.P_yy = np.zeros((dim_data,dim_data), dtype=np.float64)
		self.P_yy_copy = np.zeros((dim_data, dim_data), dtype=np.float64)
		self.P_yy_copy_worker = np.zeros((dim_data* dim_data), dtype=np.float64)
		self.P_yy_copy_IPIV = np.zeros((dim_data), dtype=np.int32)
		self.K = np.zeros((dim_model,dim_data), dtype=np.float64)
		self.P_xy = np.zeros((dim_model, dim_data), dtype=np.float64)
		self.P_M = np.zeros((dim_model,dim_model), dtype=np.float64)
		self.gamma =  np.zeros((dim_data,2*dim_model+1), dtype=np.float64)
		self.gamma2 =  np.zeros((dim_data,2*dim_model+1), dtype=np.float64)
		self.KAPPA = 0.03 # 3
		self.D =  np.zeros((dim_model,dim_model), dtype=np.float64)
		self.C =  np.zeros((dim_data, dim_model), dtype=np.float64)
		self.scale = np.array([self.KAPPA / (dim_model + self.KAPPA), 1/(2 *(dim_model+ self.KAPPA))])
		self.compute_convex_weights(self.weights, dim_model, self.KAPPA)  # weights used in the following equations




	### WAS MACHT DAS HIER?


	cdef void covariance(self, double[:,:] X, double[:,:] N, double[:,:] ret):
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, X.shape[0] , X.shape[0], X.shape[1] - 1,
					self.scale[1], &X[0,1], X.shape[1], &X[0,1], X.shape[1], 0, &ret[0,0], ret.shape[1])
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, X.shape[0] , X.shape[0], 1,
					self.scale[0], &X[0,0], X.shape[1], &X[0,0], X.shape[1], 1, &ret[0,0], ret.shape[1])
		cblas_daxpy(ret.shape[0] * ret.shape[1], 1, &N[0,0], 1, &ret[0, 0], 1)

	cdef void crosscorr(self, double[:,:] X, double[:,:] Y, double[:,:] ret):
		"""
		Calculates $\sum_i w_i (x_i - \bar{x}) (x_i - \bar{x})^T$
		"""
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,  X.shape[0] , Y.shape[0], X.shape[1] - 1,
					self.scale[1], &X[0,1], X.shape[1], &Y[0,1], Y.shape[1], 0, &ret[0,0], ret.shape[1])
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, X.shape[0] , Y.shape[0], 1,
					self.scale[0], &X[0,0], X.shape[1], &Y[0,0], Y.shape[1], 1, &ret[0,0], ret.shape[1])

	cdef void mean(self, double[:,:] X, double[:] ret):
		"""
		Calculates sum_i w_i X_i 
		"""
		cblas_dgemv(CblasRowMajor, CblasNoTrans, X.shape[0], X.shape[1], 1, &X[0, 0], X.shape[1],
					&self.weights[0], 1, 0, &ret[0], 1)

	cdef void mean_deviation(self, double[:] mean,  double[:,:] X, double[:,:] ret):
		for i in range(ret.shape[1]):
			cblas_dcopy(mean.shape[0], &mean[0], 1, &ret[0,i], ret.shape[1])
		sub_pointwise(&ret[0,0], &X[0,0], &ret[0,0], X.shape[0]*X.shape[1])

	cdef void kalman_gain(self):
		"""
		 Calculates K = P_{xy} P_{yy}^{-1}
		"""
		cblas_dcopy(self.P_yy.shape[0]*self.P_yy.shape[1], &self.P_yy[0,0], 1, &self.P_yy_copy[0,0], 1)
		inverse(self.P_yy_copy, self.P_yy_copy_worker, self.P_yy_copy_IPIV)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.P_xy.shape[0], self.P_yy_copy.shape[1], self.P_xy.shape[1], 1, &self.P_xy[0,0], self.P_xy.shape[1], &self.P_yy_copy[0,0], self.P_yy_copy.shape[1], 0, &self.K[0,0], self.P_yy_copy.shape[1])

	cdef void update_mean(self, double[:] mean, double[:] y):
		sub_pointwise(&self.y_diff[0], &y[0], &self.pred_Y_mean[0], self.pred_Y_mean.shape[0])
		cblas_dgemv(CblasRowMajor, CblasNoTrans, self.K.shape[0], self.K.shape[1], 1, &self.K[0,0], self.K.shape[1], &self.y_diff[0], 1, 1, &self.pred_X_mean[0], 1)
		cblas_dcopy(self.pred_X_mean.shape[0], &self.pred_X_mean[0],1, &mean[0], 1)

	cdef void update_covariance(self, double[:,:] P):

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, self.P_xy.shape[0], self.K.shape[0], self.P_xy.shape[1], 1, &self.P_xy[0,0], self.P_xy.shape[1], &self.K[0,0], self.P_xy.shape[1], 0, &self.D[0,0], self.C.shape[1])
		sub_pointwise(&P[0,0], &self.P_xx[0,0], &self.D[0,0], P.shape[0]*P.shape[1])


	cdef void compute_convex_weights(self, double[:] w, double n, double kappa): #nogil except *:
		""" EQ &

		Parameters
		----------
		w
		n
		kappa

		Returns
		-------

		"""
		cdef double v = 1 / (2 * (n + kappa))
		dinit(w.shape[0] -1, &w[1], &v, 1)
		w[0] = kappa / (n + kappa)
		ddiagonal(&self.weights_diag[0,0], np.array([v]), self.weights_diag.shape[0], self.weights_diag.shape[1])
		self.weights_diag[0,0] = kappa/(n + kappa)

	#
	# # Verglichen passt
	cdef int compute_sigma_points(self, double[:,:] X, double[:,:] P_M, double[:] mean, double[:,:] P, double kappa) nogil except *:
		"""
		EQ 6
		Parameters
		----------
		P_M
		X
		mean
		P
		kappa

		Returns
		-------

		"""
		cdef char TRANSA = 	0x4C
		cdef int i, info
		cblas_dcopy(P.shape[0]*P.shape[1], &P[0,0], 1, &P_M[0,0], 1)
		info = LAPACKE_dpotrf(CblasRowMajor,TRANSA, P_M.shape[0], &P_M[0,0], P_M.shape[0])
		for i in range(P_M.shape[0]-1):
			for j in range(i+1, P_M.shape[0]):
				P_M[i,j] = 0
		cblas_dscal(P.shape[0]*P.shape[1], pow(<double> P.shape[0] + kappa, 0.5), &P_M[0,0], 1)

		if info != 0:
			return info
		cblas_dcopy(mean.shape[0], &mean[0], 1, &X[0,0], X.shape[1])
		for i in range(P_M.shape[1]):
			cblas_dcopy(mean.shape[0], &mean[0], 1, &X[0, 1 + i], X.shape[1])
			cblas_dcopy(mean.shape[0], &mean[0], 1, &X[0, P.shape[1] + 1 + i], X.shape[1])
		for i in range(P_M.shape[1]):
			cblas_daxpy(P_M.shape[0], 1, &P_M[i, 0],1, &X[i, 1], 1)
			cblas_daxpy(P_M.shape[0], -1, &P_M[i, 0], 1, &X[i, P.shape[1] + 1], 1)

		return 0



	cdef int update_kalman_parameters(self, double[:] mean, double[:,:] P, double[:] y) except *: # nogil except *:
		cdef int info, i
		info = self.compute_sigma_points(self.X, self.P_M, mean, P, self.KAPPA) # eq. 17

		if info != 0:
			return info
		self._model.constrain(self.X)
		self.mean(self.X, self.pred_X_mean)
		self.mean_deviation(self.pred_X_mean, self.X, self.X2)
		self.covariance(self.X2, self._model.PROCESS_NOISE, self.P_xx)
		self._model.predict_new_observation(self.gamma, self.X) # eq. 23
		self.mean(self.gamma, self.pred_Y_mean)
		self.mean_deviation(self.pred_Y_mean, self.gamma, self.gamma2)
		self.covariance(self.gamma2, self._model.MEASUREMENT_NOISE, self.P_yy)
		self.crosscorr(self.X2, self.gamma2, self.P_xy)
		# compute Kalman GAIN
		self.kalman_gain()
		self.update_mean(mean, y)
		self.update_covariance(P)
		return 0

cdef class KalmanQuat(Kalman):
	def __cinit__(self, int dim_data, int dim_model, model):
		super(KalmanQuat, self).__init__(dim_data, dim_model, model)
		self.dim_model_mean = dim_model + 1
		self.c_mean = np.zeros((6,), dtype=np.float64)
		self.X_s = np.zeros((7, 2*dim_model+1), dtype=np.float64)
		self.pred_X_mean_q = np.zeros((7,))
		self.c_quat = np.zeros((4,), dtype=np.float64)
		self.c_quat[0] = 1
		self.c_quat1 = np.zeros((4,), dtype=np.float64)
		#print(self.c_quat)


	cdef void chart_update(self):

		for i in range(self.X2.shape[1]):
			MPR_R2H_q(self.X_s[3:,i], self.X[3:,i], self.c_quat, self.X.shape[1], 1)
			cblas_dcopy(3, &self.X[0,i], self.X.shape[1], &self.X_s[0,i], self.X_s.shape[1])
		MPR_R2H_q(self.c_quat1, self.pred_X_mean[3:], self.c_quat, 1,1)
		cblas_dcopy(4, &self.c_quat1[0], 1, &self.c_quat[0], 1)
		# mapping back would lead to zero
		cblas_dscal(3, 0, &self.pred_X_mean[3], 1)

	cdef void chart_transition(self):
		for i in range(self.X2.shape[1]):
			cblas_dcopy(6, &self.pred_X_mean[0], 1, &self.X2[0,i], self.X2.shape[1])
			MPR_H2R_q(self.X[3:, i], self.X_s[3:,i], self.c_quat, self.X_s.shape[1], 1)
		sub_pointwise(&self.X2[0,0], &self.X[0,0], &self.X2[0,0], self.X.shape[0]* self.X.shape[1])


	cdef void update_mean(self, double[:] mean, double[:] y):
		sub_pointwise(&self.y_diff[0], &y[0], &self.pred_Y_mean[0], self.pred_Y_mean.shape[0])
		cblas_dgemv(CblasRowMajor, CblasNoTrans, self.K.shape[0], self.K.shape[1], 1, &self.K[0,0], self.K.shape[1], &self.y_diff[0], 1, 1, &self.pred_X_mean[0], 1)
		cblas_dcopy(3, &self.pred_X_mean[0],1, &mean[0], 1)
		cblas_dcopy(6, &self.pred_X_mean[0],1, &self.c_mean[0], 1)
		cblas_dcopy(4, &self.c_quat[0], 1,  &mean[3], 1)



	cdef int update_kalman_parameters(self, double[:] mean, double[:,:] P, double[:] y) except *: # nogil except *:
		cdef int info, i
		info = self.compute_sigma_points(self.X, self.P_M, self.c_mean, P, self.KAPPA) # eq. 17

		if info != 0:
			return info
		self._model.constrain(self.X)
		self.mean(self.X, self.pred_X_mean)
		self.chart_update()
		self.chart_transition()
		self.covariance(self.X2, self._model.PROCESS_NOISE, self.P_xx)
		self._model.predict_new_observation(self.gamma, self.X_s) #
		self.mean(self.gamma, self.pred_Y_mean)
		self.mean_deviation(self.pred_Y_mean, self.gamma, self.gamma2)
		self.covariance(self.gamma2, self._model.MEASUREMENT_NOISE, self.P_yy)
		self.crosscorr(self.X2, self.gamma2, self.P_xy)
		self.kalman_gain()
		self.update_mean(mean, y)
		self.update_covariance(P)


		return 0
