
#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from libc.math cimport fabs, floor, pow
import numpy as np
from bonndit.utilc.blas_lapack cimport *

from bonndit.utilc.cython_helpers cimport dm2toc

cdef void bilinear(double[:] ret, double[:,:,:] data, double x, double y):
	"""
		Given a 2D grid with values in 3 dimension (data) and x,y values this function calculates the bilinear interpolation 
		and saves it in ret
	"""
	cdef double[:,:] vlinear = np.zeros((2, data.shape[2]))
	for i in range(2):
		cblas_dcopy(data.shape[2], &data[0,i,0], 1, &vlinear[i,0], 1)
		cblas_dscal(data.shape[2], (x - floor(x)), &vlinear[i,0], 1)
		cblas_daxpy(data.shape[2], (1 + floor(x) - x), &data[1, i, 0], 1, &vlinear[i, 0], 1)
	#ret = (y - floor(y))* vlinear[0] +  (1 + floor(y) - y) * vlinear[1]
	cblas_dcopy(data.shape[2], &vlinear[i,0], 1, &ret[0], 1)
	cblas_dscal(data.shape[2], (y - floor(y)), &ret[0], 1)
	cblas_daxpy(data.shape[2], (1 + floor(y) - y), &vlinear[1,0 ], 1, &ret[0], 1)

cdef double linear(double[:] point, double[:] vlinear, double[:, :, :] data) nogil except *:
		cdef int i, j, k, m,n,o
		for i in range(8):
			j = <int> floor(i / 2) % 2
			k = <int> floor(i / 4) % 2
			m = <int> point[0] + i%2
			n = <int> point[1] + j
			o = <int> point[2] + k
			vlinear[i] =  data[m,n,o]
			#print(i, vlinear[i])
		for i in range(4):
			vlinear[i] = (point[2] - floor(point[2])) * vlinear[4+i] + (1 + floor(point[2]) - point[2]) * vlinear[i]
			#print(i, vlinear[i])
		for i in range(2):
			vlinear[i] = (point[1] - floor(point[1]))* vlinear[2 + i ] +  (1 + floor(point[1]) - point[1]) * vlinear[i]
		return  (point[0] - floor(point[0])) * vlinear[1] +  (1 + floor(point[0]) - point[0]) * vlinear[0]


cdef void trilinear_v(double[:] point, double[:] y, double[:,:] vlinear, double[:, :, :, :] data) nogil except *:
	cdef int i, j, k, m,n,o
	for i in range(8):
		j = <int> floor(i / 2) % 2
		k = <int> floor(i / 4) % 2
		m = <int> point[0] + i%2
		n = <int> point[1] + j
		o = <int> point[2] + k

		dm2toc(&vlinear[i, 0], data[m,n,o,:],  vlinear.shape[1])


	for i in range(4):
		cblas_dscal(vlinear.shape[1], (1 + floor(point[2]) - point[2]), &vlinear[i, 0], 1)
		cblas_daxpy(vlinear.shape[1], (point[2] - floor(point[2])), &vlinear[4+i, 0], 1, &vlinear[i,0], 1)
	for i in range(2):
		cblas_dscal(vlinear.shape[1], (1 + floor(point[1]) - point[1]), &vlinear[i, 0], 1)
		cblas_daxpy(vlinear.shape[1], (point[1] - floor(point[1])), &vlinear[2 + i, 0], 1, &vlinear[i, 0], 1)
	cblas_dscal(vlinear.shape[1], (1 + floor(point[0]) - point[0]), &vlinear[0, 0], 1)
	cblas_daxpy(vlinear.shape[1], (point[0] - floor(point[0])), &vlinear[1,0], 1, &vlinear[0,0], 1)
	cblas_dcopy(vlinear.shape[1], &vlinear[0,0], 1, &y[0], 1)


cpdef double linear_p(double[:] point, double[:] vlinear, double[:, :, :] data):
	return linear(point, vlinear, data)

