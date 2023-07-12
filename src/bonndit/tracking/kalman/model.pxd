
cdef class AbstractModel:

	cdef double[:,:] MEASUREMENT_NOISE
	cdef double[:,:] PROCESS_NOISE
	cdef double _lambda_min
	cdef int num_tensors
	cdef double[:] m
	cdef double GLOBAL_TENSOR_UNPACK_VALUE
	cdef void normalize(self, double[:], double[:], int)  #nogil expcept *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:],double[:]) except *
	cdef void constrain(self, double[:,:]) except * #nogil expcept *



cdef class fODFModel(AbstractModel):
	cdef double[:] res
	cdef int order
	cdef double[:,:,:,:,:] vector_field
	cdef void normalize(self, double[:], double[:], int) #nogil expcept *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:]) except *
	cdef void constrain(self, double[:,:]) except * #nogil expcept *


cdef class WatsonModel(AbstractModel):
	cdef double[:] res
	cdef int order
	cdef double[:,:,:,:,:] vector_field
	cdef double[:] rank_1_rh_o4
	cdef double[:] angles
	cdef double[:] rot_pysh_v
	cdef double[:] pysh_v
	cdef double[:] dipy_v
	#cdef void normalize(self, double[:], double[:], int) #nogil expcept *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:]) except *
	cdef void constrain(self, double[:,:]) except * #nogil expcept *
	cdef double sh_norm(self, double[:])


cdef class BinghamModel(WatsonModel):
	cdef double[:,:,:,:] lookup_table
#	cdef int convert_to_index(self, double, double, double) #nogil expcept *
	cdef void sh_bingham_coeffs(self, double, double) except *#nogil expcept *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:]) except *
	cdef void constrain(self, double[:,:]) except * #nogil expcept *

cdef class BinghamQuatModel(BinghamModel):
	cdef double[:,:,:] lookup_table1
	cdef double[:] sh
	cdef double[:,:] lookup_kappa_beta_table
#	cdef int convert_to_index(self, double, double, double) #nogil expcept *
	#cdef void sh_bingham_coeffs(self, double, double) except * #nogil expcept *
	cdef void lookup_kappa_beta(self, double[:], double, double)
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:]) except *
	cdef void constrain(self, double[:,:]) except * #nogil exp

cdef class MultiTensorModel(AbstractModel):
	cdef double[:,:] M
	cdef double[:] q
	cdef double[:] lam
	cdef double[:,:] gradients
	cdef double[:] baseline_signal
	cdef double acq_spec_const
	cdef double[:,:] D
	cdef double[:] c
	cdef void diffusion(self, double[:,:], double[:], double[:], double[:,:]) #nogil expcept *
	cdef void normalize(self, double[:], double[:], int) #nogil expcept *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) #nogil expcept *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:])
	cdef void constrain(self, double[:,:]) #nogil expcept *

#cdef extern from "watsonfit.h":
#	void minimize_watson_mult_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#	void minimize_watson_mult_o6(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#	void minimize_watson_mult_o8(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#	void SHRotateRealCoef(double *, double *, int, double *, double *)
#	void map_dipy_to_pysh_o4(double *, double *)
#	void map_pysh_to_dipy_o4(double *, double *)
#	void sh_watson_coeffs(double, double *, int)
##
