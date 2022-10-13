cdef class AbstractModel:

	cdef double[:,:] MEASUREMENT_NOISE
	cdef double[:,:] PROCESS_NOISE
	cdef double _lambda_min
	cdef int num_tensors
	cdef double[:] m
	cdef double GLOBAL_TENSOR_UNPACK_VALUE
	cdef void normalize(self, double[:], double[:], int) nogil except *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) nogil except *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:],double[:])
	cdef void constrain(self, double[:,:]) nogil except *



cdef class fODFModel(AbstractModel):
	cdef double[:] res
	cdef int order
	cdef double[:,:,:,:,:] vector_field
	cdef void normalize(self, double[:], double[:], int) nogil except *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) nogil except *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:])
	cdef void constrain(self, double[:,:]) nogil except *


cdef class MultiTensorModel(AbstractModel):
	cdef double[:,:] M
	cdef double[:] q
	cdef double[:] lam
	cdef double[:,:] gradients
	cdef double[:] baseline_signal
	cdef double acq_spec_const
	cdef double[:,:] D
	cdef double[:] c
	cdef void diffusion(self, double[:,:], double[:], double[:], double[:,:]) nogil except *
	cdef void normalize(self, double[:], double[:], int) nogil except *
	cdef void predict_new_observation(self, double[:,:], double[:,:]) nogil except *
	cdef bint kinit(self, double[:], double[:], double[:], double[:,:], double[:])
	cdef void constrain(self, double[:,:]) nogil except *
