from .model cimport AbstractModel

cdef class Kalman:
	cdef double[:,::1]  X
	cdef double[:,::1]  X2
	cdef AbstractModel _model
	cdef double[:]  weights
	cdef double[:]  pred_X_mean
	cdef double[:]  pred_Y_mean
	cdef double[:] y_diff
	cdef double[:,::1] P_xx
	cdef double[:,::1] P_yy
	cdef double[:,::1] P_yy_copy
	cdef double[:] P_yy_copy_worker
	cdef int[:] P_yy_copy_IPIV
	cdef double[:,::1] K
	cdef double[:,::1] P_xy
	cdef double[:,::1] P_M
	cdef double[:,::1] gamma
	cdef double[:,::1] gamma2
	cdef double[:,:,:,:] data
	cdef double KAPPA
	cdef double[:,::1] D
	cdef double[:,::1] C

	cdef int update_kalman_parameters(self, double[:], double[:,:], double[:]) nogil except *
	cdef void linear(self, double[:], double[:], double[:,:], double[:, :, :, :]) nogil except *
	cdef int compute_sigma_points(self, double[:,:], double[:,:], double[:], double[:,:], double) nogil except *
	cdef void compute_convex_weights(self, double[:], double, double) nogil except *
