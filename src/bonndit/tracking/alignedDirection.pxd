#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True
# warn.unused_results=True

cdef class Probabilities:
	cdef double chosen_angle, old_fa
	cdef double chosen_prob
	cdef double expectation, sigma
	cdef double[:] probability, angles, best_fit
	cdef double[:,:] test_vectors
	cdef int random_choice(self, double[:]) # nogil  except *
	cdef void aligned_direction(self, double[:,:], double[:]) # nogil  except *
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class Gaussian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class Laplacian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class ScalarOld(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class ScalarNew(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class Deterministic(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *

cdef class Deterministic2(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) # nogil except *


cdef class BinghamDirGetter(Probabilities):
	cdef void watson_config(self, double[:,:,:,:], double, double, double, bint) #nogil  except *:
	cdef double bingham_scale(self, double, double) #nogil  except *:
	cdef double bingham(self, double[:], double[:], double, double) #nogil  except *:
	cdef void mc_random_direction(self, double[:], double[:], double, double) #nogil  except *:
	cdef void calculate_probabilities_sampled(self, double[:, :], double[:], double[:, :, :], double[:, :])

cdef class WatsonDirGetter(Probabilities):
	cdef void watson_config(self, double[:,:,:,:], double, double, double, bint)
	cdef void calculate_probabilities_sampled(self, double[:,:], double[:], double[:], double[:], double[:])
	cdef double poly_kummer(self, double)
	cdef double poly_watson(self, double[:], double[:], double, double)
	cdef void mc_random_direction(self, double[:], double[:], double, double)
	cdef void calculate_probabilities_sampled(self, double[:,:], double[:], double[:], double[:], double[:])
