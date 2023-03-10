#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import os
from scipy.interpolate import RegularGridInterpolator
from bonndit.utilc.blas_lapack cimport cblas_dgemv, CblasRowMajor, CblasNoTrans

from bonndit.utilc.cython_helpers cimport sub_vectors, angle_deg, sum_c, set_zero_matrix, bigger, smaller, mult_with_scalar, norm
import numpy as np
from bonndit.tracking.ItoW cimport Trafo
from bonndit.utilc.trilinear cimport linear
DTYPE = np.float64

cdef double[:] y = np.zeros((8,))

cdef class Validator:
	def __cinit__(self, int[:] shape, inclusion, exclusion,  Trafo trafo, **kwargs):
		self.inv_trafo = np.linalg.inv(kwargs['trafo_mask'])
		self.point = np.zeros((4,), dtype=DTYPE)
		self.point_world = np.zeros((4,), dtype=DTYPE)
		self.shape = shape
	#	if kwargs['act'] is not None:
	#		self.WM = ACT(kwargs)
	#	else:
		self.WM = WMChecker(kwargs)
		if isinstance(inclusion, np.ndarray):
			self.ROIIn = ROIInValidator(inclusion)
		else:
			self.ROIIn = ROIInNotValidator(np.zeros((3,3)))
		if isinstance(exclusion, np.ndarray):
			self.ROIEx = ROIExValidator(exclusion)
		else:
			self.ROIEx = ROIExNotValidator(np.zeros((3,3)))
		if kwargs['max_angle'] > 0:
			self.Curve = CurvatureValidator(kwargs['max_angle'], trafo, kwargs['stepsize'])
		else:
			self.Curve = CurvatureNotValidator(kwargs['max_angle'], trafo, kwargs['stepsize'])



	#cdef bint wm_checker(self, double[:] point) : # nogil except *:
	#	""" Checks if the wm density is at a given point below a threshold.
	#	@param point: 3 dimensional point
	#	"""
	#	if self.wm_mask[int(point[0]), int(point[1]), int(point[2])] < self.min_wm:
	#		return True
	#	else:
	#		return False

	cdef bint index_checker(self, double[:] point) : # nogil except *:
		"""
		Checks if the index is within the array.
		@param point: 3 dimensional point
		@return: True if the point is not valid.
		"""
		self.point_world[:3] = point
		self.point_world[3] = 1
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,
					&self.point[0], 1)
		if self.point[0] < 0 or self.point[1] < 0 or self.point[2] < 0:
			return True
		elif self.point[0] >= self.shape[0] or self.point[1] >= self.shape[1] or self.point[2] >= self.shape[2]:
			return True
		else:
			return False



	cdef bint next_point_checker(self, double[:] point) : # nogil except *:
		"""
		Check if a given direction is valid e.g. not zero and not infinity
		@param point: given direction
		@return:
		"""
		if sum_c(point) == 0 or sum_c(point) != sum_c(point):
			return True
		else:
			return False

	cdef void set_path_zero(self, double[:,:] path, double[:,:] features) : # nogil except *:
		set_zero_matrix(path)
		set_zero_matrix(features)

cdef class WMChecker:
	def __cinit__(self, kwargs):
		x = np.linspace(0, kwargs['wm_mask'].shape[0] - 1, kwargs['wm_mask'].shape[0])
		y = np.linspace(0, kwargs['wm_mask'].shape[1] - 1, kwargs['wm_mask'].shape[1])
		z = np.linspace(0, kwargs['wm_mask'].shape[2] - 1, kwargs['wm_mask'].shape[2])
		self.inv_trafo = np.linalg.inv(kwargs['trafo_mask'])
		self.point = np.zeros((4,), dtype=DTYPE)
		self.point_world = np.zeros((4,), dtype=DTYPE)
		self.min_wm = kwargs['wmmin']

		#self.entered_sgm = 0
		#self.wm_mask = RegularGridInterpolator((x, y, z), kwargs['wm_mask'])
		self.wm_mask = np.array(kwargs['wm_mask'], dtype=DTYPE)


	cdef void reset(self):
		pass

	cdef bint sgm_checker(self, double[:] point):
		return 0

	cdef bint wm_checker(self, double[:] point) : # nogil except *:
			""" Checks if the wm density is at a given point below a threshold.
			@param point: 3 dimensional point
			"""
			cdef int i=0,j=0,k=0
			self.point_world[:3] = point
			self.point_world[3] = 1
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,&self.point[0], 1)
			if linear(self.point[:3], y, self.wm_mask) > self.min_wm:
				return -1
			else:
				return 0

	cdef bint wm_checker_ex(self, double[:] point) : # nogil except *:
				""" Checks if the wm density is at a given point below a threshold.
				@param point: 3 dimensional point
				"""
				self.point_world[:3] = point
				self.point_world[3] = 1
				cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,&self.point[0], 1)
				if linear(self.point[:3], y, self.wm_mask)  > self.min_wm:
					return -1
				else:
					return 0

#cdef class ACT(WMChecker):
#	"""
#	Format of kwargs['act']:
#    0: Cortical grey matter
#    1: Sub-cortical grey matter
#    2: White matter
#    3: CSF
#    4: Pathological tissue
#	"""
#	def __cinit__(self, kwargs):
#	#	super().__init__(kwargs)
#		x = np.linspace(0, kwargs['act'].shape[0] - 1, kwargs['act'].shape[0])
#		y = np.linspace(0, kwargs['act'].shape[1] - 1, kwargs['act'].shape[1])
#		z = np.linspace(0, kwargs['act'].shape[2] - 1, kwargs['act'].shape[2])
#		self.entered_sgm = 0
#		self.cgm = RegularGridInterpolator((x,y,z), kwargs['act'][...,0])
#		self.sgm = RegularGridInterpolator((x, y, z), kwargs['act'][...,1])
#		self.wm = RegularGridInterpolator((x, y, z), kwargs['act'][...,2])
#		self.csf = RegularGridInterpolator((x, y, z), kwargs['act'][...,3])
#
#	cdef void reset(self):
#		self.entered_sgm = 0
#
#	cdef bint wm_checker(self, double[:] point):
#		self.point_world[:3] = point
#		self.point_world[3] = 1
#		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,&self.point[0], 1)
#
#		cgm = self.cgm(self.point[:3])
#		csf = self.csf(self.point[:3])
#		sgm = self.sgm(self.point[:3])
#		wm = self.wm(self.point[:3])
#		#check case 6:
#		if self.entered_sgm:
#			if sgm<0.5:
#				return 0
#			else:
#				return  -1
#		# continue
#		if wm > 0.5:
#			return -1
#
#		# ACT cases 1 => accept
#		if cgm > 0.5:
#			return 0
#		# case 2 => reject
#		if csf > 0.5:
#			return 2
#		# case 3 => accept
#		if cgm + csf + sgm + wm < 0.3:
#			return 0
#		#case 6
#		if sgm>0.5:
#			self.entered_sgm = 1
#			return -1
#		return 0
#
#
#
#
#	cdef bint sgm_checker(self, double[:] point):
#		self.point_world[:3] = point
#		self.point_world[3] = 1
#		cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,&self.point[0], 1)
#
#		sgm = self.sgm(self.point[:3])
#		if sgm>0.5:
#			return 1
#		else:
#			return 0


cdef class CurvatureNotValidator:
	def __cinit__(self, max_angle, trafo, double step_width):
		self.max_angle = max_angle
		self.angle = 0
		self.step_width = step_width
		self.points = np.zeros([5,3])
		self.trafo = trafo

	cdef bint curvature_checker(self, double[:,:] path,  double[:] features) : # nogil except *:
		return False

cdef class CurvatureValidator(CurvatureNotValidator):
	#def __cinit__(self, double max_angle):
#		super().__cinit__(max_angle)

	cdef bint curvature_checker(self, double[:,:] path,  double[:] features) : # nogil except *:
			"""
			Checks the angles between the current direction and the directions anlong the polygon. If a angle is to large returns True
			@param path: polygon to check
			@param features: save the angle between the current direction and the direction k points ago into the features.
			@return:
			"""
			cdef int l = 1, k = path.shape[0]
			cdef double length = 0
			self.trafo.itow(path[k])
			mult_with_scalar(self.points[3], 1, self.trafo.point_itow)
			sub_vectors(self.points[1], path[k-1], path[k])
			length += norm(self.points[1])
			#with gil: print(k, length, l)
			while k >= 2 and length < 30 and l < k:
				l += 1
				sub_vectors(self.points[0], path[k-l], path[k-l+1])
				length += norm(self.points[0])
				self.angle = angle_deg(self.points[1], self.points[0])
			#	with gil:
			#		print("test", self.angle)
				if self.angle > self.max_angle:
					return True
			else:
				features[0] = self.angle
				return False

cdef class ROIInNotValidator:
	def __cinit__(self, double[:,:] inclusion):
		self.inclusion = np.zeros([3,3])
		self.inclusion_num = 0
		self.inclusion_check = np.zeros(1)


	cdef int included(self, double[:] point) : # nogil except *:
		return 0

	cdef bint included_checker(self) : # nogil except *:
		return False

cdef class ROIInValidator(ROIInNotValidator):
	def __cinit__(self, double[:,:] inclusion):
		self.inclusion = inclusion[:,:3]
		self.inclusion_num = inclusion.shape[0]//2
		self.inclusion_check = np.zeros(inclusion.shape[0]//2)

	cdef int included(self, double[:] point) : # nogil except *:
		cdef int i
		if sum_c(self.inclusion_check) == self.inclusion_num:
			return 0

		for i in range(self.inclusion_num):
			if not bigger(point, self.inclusion[2*i]):
				continue
			if not smaller(point, self.inclusion[2*i + 1]):
				continue
			self.inclusion_check[i] = 1
			return i + 1
		return -1

	cpdef int included_p(self, double[:] point) except *:
		return self.included(point)

	cpdef void reset_p(self) except *:
		self.inclusion_check = np.zeros(self.inclusion_num)


	cdef bint included_checker(self) : # nogil except *:
		return sum_c(self.inclusion_check) != self.inclusion_num


cdef class ROIExNotValidator:
	def __cinit__(self, double[:,:] exclusion):
		self.exclusion_cube = np.zeros([3,3])
		self.exclusion_num = 0

	cdef bint excluded(self, double[:] point) : # nogil except *:
		return False


cdef class ROIExValidator(ROIExNotValidator):
	def __cinit__(self, double[:,:] exclusion):
		self.exclusion_cube = exclusion
		self.exclusion_num = len(exclusion.shape[0])

	cdef bint excluded(self, double[:] point) : # nogil except *:
		cdef int i
		for i in range(self.exclusion_num):
			if not bigger(point, self.exclusion_cube[2*i]):
				continue
			if not smaller(point, self.exclusion_cube[2*i + 1]):
				continue
			return True
		return False




