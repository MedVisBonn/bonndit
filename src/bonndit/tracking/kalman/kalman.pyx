#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import ctypes
from bonndit.utilc.cython_helpers cimport ddiagonal,  dm2toc, dinit, sub_pointwise, special_mat_mul,inverse
from bonndit.utilc.blas_lapack cimport *
import numpy as np

from libc.math cimport fabs, floor, pow






cdef class Kalman:

	def __cinit__(self, int dim_data, int dim_model, model):
		self.X = np.zeros((dim_model, 2*dim_model+1), dtype=np.float64)
		self.X2 = np.zeros((dim_model,2*dim_model+1), dtype=np.float64)
		self._model = model
		self.weights = np.zeros((2*dim_model+1,), dtype=np.float64)
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
		self.KAPPA = 0.3
		self.D =  np.zeros((dim_model,dim_model), dtype=np.float64)
		#self.c = np.zeros((10,), dtype=np.float64)
		self.C =  np.zeros((dim_data, dim_model), dtype=np.float64)

		self.compute_convex_weights(self.weights, dim_model, self.KAPPA)  # weights used in the following equations





	cdef void linear(self, double[:] point, double[:] y, double[:,:] vlinear, double[:, :, :, :] data) nogil except *:
		cdef int i, j, k, m,n,o
		for i in range(8):
			j = <int> floor(i / 2) % 2
			k = <int> floor(i / 4) % 2
			m = <int> point[0] + i%2
			n = <int> point[1] + j
			o = <int> point[2] + k

			dm2toc(&vlinear[i, 0], data[m,n,o,:],  vlinear.shape[1])


		for i in range(4):
			cblas_dscal(vlinear.shape[1], (1 + floor(point[2]) - point[2]), &vlinear[i, 0], 1)
			cblas_daxpy(vlinear.shape[1], (point[2] - floor(point[2])), &vlinear[4+i, 0], 1, &vlinear[i,0], 1)
		for i in range(2):
			cblas_dscal(vlinear.shape[1], (1 + floor(point[1]) - point[1]), &vlinear[i, 0], 1)
			cblas_daxpy(vlinear.shape[1], (point[1] - floor(point[1])), &vlinear[2 + i, 0], 1, &vlinear[i, 0], 1)
		cblas_dscal(vlinear.shape[1], (1 + floor(point[0]) - point[0]), &vlinear[0, 0], 1)
		cblas_daxpy(vlinear.shape[1], (point[0] - floor(point[0])), &vlinear[1,0], 1, &vlinear[0,0], 1)
		cblas_dcopy(vlinear.shape[1], &vlinear[0,0], 1, &y[0], 1)


	cdef void compute_convex_weights(self, double[:] w, double n, double kappa) nogil except *:
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



	cdef int update_kalman_parameters(self, double[:] mean, double[:,:] P, double[:] y) nogil except *:
		cdef int info, i
		info = self.compute_sigma_points(self.X, self.P_M, mean, P, self.KAPPA) # eq. 17
		if info != 0:
			return info
		#check
		self._model.constrain(self.X)
		# pred X mean check
		cblas_dgemv(CblasRowMajor, CblasNoTrans, self.X.shape[0], self.X.shape[1], 1, &self.X[0, 0], self.X.shape[1], &self.weights[0], 1, 0, &self.pred_X_mean[0], 1)

		# Line 174-183
		for i in range(self.X2.shape[1]):
			cblas_dcopy(self.pred_X_mean.shape[0], &self.pred_X_mean[0], 1, &self.X2[0,i], self.X2.shape[1])
		sub_pointwise(&self.X2[0,0], &self.X[0,0], &self.X2[0,0], self.X.shape[0]* self.X.shape[1])


		special_mat_mul(self.P_xx, self.X2, self.weights, self.X2, 1)
		cblas_daxpy(self.P_xx.shape[0] * self.P_xx.shape[1], 1, &self._model.PROCESS_NOISE[0,0], 1, &self.P_xx[0, 0], 1)
		self._model.predict_new_observation(self.gamma, self.X) # eq. 23

		cblas_dgemv(CblasRowMajor, CblasNoTrans, self.gamma.shape[0], self.gamma.shape[1], 1, &self.gamma[0, 0],self.gamma.shape[1], &self.weights[0], 1, 0, &self.pred_Y_mean[0], 1)
		for i in range(self.gamma2.shape[1]):
			cblas_dcopy(self.pred_Y_mean.shape[0], &self.pred_Y_mean[0], 1, &self.gamma2[0,i], self.gamma2.shape[1])
		sub_pointwise(&self.gamma2[0,0],&self.gamma[0,0], &self.gamma2[0,0], self.gamma2.shape[0]*self.gamma2.shape[1])
		special_mat_mul(self.P_yy, self.gamma2, self.weights, self.gamma2, 1)
		cblas_daxpy(self.P_yy.shape[0] * self.P_yy.shape[1], 1, &self._model.MEASUREMENT_NOISE[0, 0], 1, &self.P_yy[0, 0], 1)
		special_mat_mul(self.P_xy, self.X2, self.weights, self.gamma2, 1)
		# compute Kalman GAIN
		cblas_dcopy(self.P_yy.shape[0]*self.P_yy.shape[1], &self.P_yy[0,0], 1, &self.P_yy_copy[0,0], 1)
		inverse(self.P_yy_copy, self.P_yy_copy_worker, self.P_yy_copy_IPIV)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.P_xy.shape[0], self.P_yy_copy.shape[1], self.P_xy.shape[1], 1, &self.P_xy[0,0], self.P_xy.shape[1], &self.P_yy_copy[0,0], self.P_yy_copy.shape[1], 0, &self.K[0,0], self.P_yy_copy.shape[1])
		sub_pointwise(&self.y_diff[0], &y[0], &self.pred_Y_mean[0], self.pred_Y_mean.shape[0])
		cblas_dgemv(CblasRowMajor, CblasNoTrans, self.K.shape[0], self.K.shape[1], 1, &self.K[0,0], self.K.shape[1], &self.y_diff[0], 1, 1, &self.pred_X_mean[0], 1)
		cblas_dcopy(self.pred_X_mean.shape[0], &self.pred_X_mean[0],1, &mean[0], 1)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, self.P_yy.shape[0], self.K.shape[0], self.P_yy.shape[1], 1, &self.P_yy[0,0], self.P_yy.shape[1], &self.K[0,0], self.P_yy.shape[1], 0, &self.C[0,0], self.K.shape[0])

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.K.shape[0], self.C.shape[1], self.K.shape[1], 1, &self.K[0,0], self.K.shape[1], &self.C[0,0], self.C.shape[1], 0, &self.D[0,0], self.C.shape[1])
		sub_pointwise(&P[0,0], &self.P_xx[0,0], &self.D[0,0], P.shape[0]*P.shape[1])
		return 0

