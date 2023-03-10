
#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

from libc.math cimport fabs, floor, pow

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

cpdef double linear_p(double[:] point, double[:] vlinear, double[:, :, :] data):
	return linear(point, vlinear, data)

