#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cdef class Probabilities:
	cdef double chosen_angle, old_fa
	cdef double chosen_prob
	cdef double expectation, sigma
	cdef double[:] probability, angles, best_fit
	cdef double[:,:] test_vectors
	cdef void random_choice(self, double[:]) nogil  except *
	cdef void aligned_direction(self, double[:,:], double[:]) nogil  except *
	cdef void calculate_probabilities(self, double[:,:], double[:]) nogil except *

cdef class Gaussian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) nogil except *

cdef class Laplacian(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) nogil except *

cdef class ScalarOld(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) nogil except *

cdef class ScalarNew(Probabilities):
	cdef void calculate_probabilities(self, double[:,:], double[:]) nogil except *
