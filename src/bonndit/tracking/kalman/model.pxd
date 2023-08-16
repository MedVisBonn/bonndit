#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True



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
	cdef void single_predicton(self, double[:,:] , double[:,:] , int , int )



cdef class fODFModel(AbstractModel):
	cdef double[:] res
	cdef int order
	cdef double[:,:,:,:,:] vector_field


cdef class WatsonModel(AbstractModel):
	cdef double[:] res
	cdef int order
	cdef double[:,:,:,:,:] vector_field
	cdef double[:] rank_1_rh_o4
	cdef double[:] angles
	cdef double[:] rot_pysh_v
	cdef double[:] pysh_v
	cdef double[:] dipy_v
	cdef double sh_norm(self, double[:])


cdef class BinghamModel(WatsonModel):
	cdef double[:,:,:,:] lookup_table
	cdef int num_parameter
	cdef double[:,:,:] lookup_table1
	cdef double[:,:] lookup_kappa_beta_table
	cdef double[:] sh
	cdef void set_mean(self, double[:] , double[:,:], double[:] , int , int )
	cdef void set_angle_for_prediction(self, double[:], int)
	cdef void predict_new_observation(self, double[:,:], double[:,:]) except * #nogil expcept *
	cdef void lookup_kappa_beta(self, double[:], double, double)

cdef class BinghamQuatModel(BinghamModel):
	pass


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

