#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
from bonndit.tracking.ItoW cimport Trafo
cdef class Validator:
	cdef:
		double min_wm
		double[:,:,:] wm_mask
		int[:] shape
		CurvatureNotValidator Curve
		ROIInNotValidator ROIIn


	cdef bint wm_checker(self, double[:]) nogil except *
	cdef bint index_checker(self, double[:]) nogil except *
	cdef bint next_point_checker(self, double[:]) nogil except *
	cdef void set_path_zero(self, double[:,:], double[:,:]) nogil except *


cdef class CurvatureNotValidator:
	cdef:
		double[:,:] points
		double angle
		double max_angle
		Trafo trafo

	cdef bint curvature_checker(self, double[:,:], double[:]) nogil except *

cdef class CurvatureValidator(CurvatureNotValidator):
	cdef bint curvature_checker(self, double[:,:],  double[:]) nogil except *

cdef class ROIInNotValidator:
	cdef:
		double[:,:] inclusion
		double[:] inclusion_check
		int inclusion_num
	cdef void included(self, double[:]) nogil except *
	cdef bint included_checker(self) nogil except *

cdef class ROIInValidator(ROIInNotValidator):


	cdef void included(self, double[:]) nogil except *
	cdef bint included_checker(self)	 nogil except *


