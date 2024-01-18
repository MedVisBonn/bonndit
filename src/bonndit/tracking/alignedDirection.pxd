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
	cdef void calculate_probabilities_sampled(self, double[:,:], double[:], double[:], double[:], double[:])
	cdef void calculate_probabilities_sampled_bingham(self, double[:, :], double[:], double[:, :, :], double[:, :])
	cdef void select_next_dir(self, double[:], double[:])

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
	cdef double max_samplingangle
	cdef double max_kappa
	cdef double max_beta
	cdef double min_beta
	cdef double min_lambda
	cdef double min_kappa
	cdef bint prob_direction
	cdef double[:] c
	cdef double bingham(self, double[:], double[:], double, double[:,:]) #nogil  except *:
	cdef void mc_random_direction(self, double[:], double[:], double, double[:,:]) #nogil  except *:
	cdef void calculate_probabilities_sampled_bingham(self, double[:, :], double[:], double[:, :, :], double[:, :])

cdef class WatsonDirGetter(Probabilities):
	cdef double max_samplingangle
	cdef double max_kappa
	cdef double min_kappa
	cdef bint prob_direction
	cdef void calculate_probabilities_sampled(self, double[:,:], double[:], double[:], double[:], double[:])
	cdef double poly_kummer(self, double)
	cdef double poly_watson(self, double[:], double[:], double, double)
	cdef void mc_random_direction(self, double[:], double[:], double, double)

cdef class TractSegGetter(WatsonDirGetter):
	cdef void select_next_dir(self, double[:], double[:])