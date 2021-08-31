#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=True, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from helper_functions.cython_helpers cimport matrix_mult, add_vectors, sub_vectors, set_zero_vector
import numpy as np
DTYPE = np.float64

cdef class Trafo:

	def __cinit__(self, double[:,:] wm_mask, double wm_boundary):
		self.wm_mask = wm_mask
		self.wm_boundary = wm_boundary
