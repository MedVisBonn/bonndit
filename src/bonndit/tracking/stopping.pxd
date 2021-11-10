#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cdef class Validator:
	cdef:
		double min_wm
		double[:,:,:] wm_mask
		int[:] shape
		CurvatureNotValidator Curve
		ROINotValidator ROI


	cdef bint wm_checker(self, double[:]) nogil except *
	cdef bint index_checker(self, double[:]) nogil except *
	cdef bint next_point_checker(self, double[:]) nogil except *
	cdef void set_path_zero(self, double[:,:], double[:,:]) nogil except *


cdef class CurvatureNotValidator:
	cdef:
		double[:,:] points
		double angle
		double max_angle

	cdef bint curvature_checker(self, double[:,:], int, double[:]) nogil except *

cdef class CurvatureValidator(CurvatureNotValidator):
	cdef bint curvature_checker(self, double[:,:], int, double[:]) nogil except *

cdef class ROINotValidator:
	cdef:
		double[:,:] inclusion
		double[:] inclusion_check
		int inclusion_num
	cdef void included(self, double[:]) nogil except *
	cdef bint included_checker(self) nogil except *

cdef class ROIValidator(ROINotValidator):


	cdef void included(self, double[:]) nogil except *
	cdef bint included_checker(self)	 nogil except *


