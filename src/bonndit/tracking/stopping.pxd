#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True
# warn.unused_results=True
from bonndit.tracking.ItoW cimport Trafo
cdef class Validator:
	cdef:
		double min_wm
		double[:,:] inv_trafo
		double[:] point
		double[:] point_world
		double[:,:,:] wm_mask
		int[:] shape
		CurvatureNotValidator Curve
		ROIInNotValidator ROIIn
		ROIExNotValidator ROIEx
		WMChecker WM

	cdef bint index_checker(self, double[:]) # nogil except *
	cdef bint next_point_checker(self, double[:]) # nogil except *
	cdef void set_path_zero(self, double[:,:], double[:,:]) # nogil except *

cdef class WMChecker:
	cdef double[:,:] inv_trafo
	cdef double[:] point
	cdef double[:] point_world
#	cdef int entered_sgm
	cdef double min_wm
	cdef double[:,:,:] wm_mask


	cdef void reset(self)
	cdef bint sgm_checker(self, double[:])
	cdef bint wm_checker(self, double[:])
	cdef bint wm_checker_ex(self, double[:])


#cdef class ACT(WMChecker):
#	cdef:
#		cgm
#		sgm
#		wm
#		csf
#	cdef void reset(self)
#	cdef bint wm_checker(self, double[:] )
#	cdef bint sgm_checker(self, double[:])



cdef class CurvatureNotValidator:
	cdef:
		double step_width
		double[:,:] points
		double angle
		double max_angle
		Trafo trafo

	cdef bint curvature_checker(self, double[:,:], double[:]) # nogil except *

cdef class CurvatureValidator(CurvatureNotValidator):
	cdef bint curvature_checker(self, double[:,:],  double[:]) # nogil except *

cdef class ROIInNotValidator:
	cdef:
		double[:,:] inclusion
		double[:] inclusion_check
		int inclusion_num
	cdef int included(self, double[:]) # nogil except *
	cdef bint included_checker(self) # nogil except *

cdef class ROIInValidator(ROIInNotValidator):
	cdef int included(self, double[:]) # nogil except *
	cdef bint included_checker(self) # nogil except *
	cpdef int included_p(self, double[:]) except *
	cpdef void reset_p(self) except *



cdef class ROIExNotValidator:
	cdef:
		double[:,:] exclusion_cube
		int exclusion_num

	cdef bint excluded(self, double[:]) # nogil except *



cdef class ROIExValidator(ROIExNotValidator):
	cdef bint excluded(self, double[:]) # nogil except *








