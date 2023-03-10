#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
cimport cython
import numpy as np
from libc.math cimport pow, floor, acos, pi, sqrt, atan2, sin, cos
from libc.stdio cimport printf
from .blas_lapack cimport *

cdef double min_c(double a, double b) nogil:
	if a < b:
		return a
	else:
		return b

cdef void sub_pointwise(double *o, double *a, double *b, int n) nogil except *:
	cdef int i
	for i in range(n):
		o[i] = a[i] - b[i]

cdef void dinit(int n, double *v, double *x, int j) nogil except *:
	cdef int i
	for i in range(n):
		v[i] = x[i%j]


cdef void dctov(double * v, double[:] a) nogil except *:
	cdef int i
	for i in range(a.shape[0]):
		a[i] = v[i]

@cython.cdivision(True)
cdef double fa(double l1, double l2, double l3) nogil except *:
	cdef double mean, a,  b
	mean=(l1+l2+l3)/3
	a = pow(l1-mean, 2) + pow(l2-mean, 2) + pow(l3-mean, 2)
	b = l1 * l1 + l2 * l2 + l3 * l3
	if b!=0:
		return pow(3/2, 0.5) * pow(a/b , 0.5)
	else:
		return 0

cdef void dm2toc(double *v, double[:] a, int num) nogil except *:

	cdef int i
	for i in range(num):
		v[i] = a[i]

cdef sphere2world(double r, double sigma, double phi):
	return r*np.array([np.sin(sigma)*np.cos(phi), np.sin(sigma)*np.sin(phi), np.cos(sigma)])

cdef world2sphere(double x,double y, double z):
	r = np.sqrt(x**2 + y**2 + z**2)
	sigma = np.arccos(z/r)
	if x > 0:
		phi = np.arctan(y/x)
	elif x<0 and y >= 0:
		phi = np.arctan(y / x) + np.pi
	elif x<0 and y < 0:
		phi = np.arctan(y / x) - np.pi
	elif y > 0:
		phi = np.pi/2
	elif y < 0:
		phi = -np.pi/2
	else:
		raise Exception()
	return r, sigma, phi

cdef r_z_r_y_r_z(double a, double b, double c):
	# getestet und für gut befunden!
	return np.array(
		[[cos(a) * cos(b) * cos(c) - sin(a) * sin(c), - cos(a)*cos(b)*sin(c) - sin(a)*cos(c), cos(a) * sin(b)],
		 [sin(a) * cos(b) * cos(c) + cos(a) * sin(c), - sin(a) * cos(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b)],
		 [-sin(b) * cos(c), sin(b) * sin(c), cos(b)]])

cdef orthonormal_from_sphere(double sigma, double phi):
	return [sphere2world(1, sigma, phi),sphere2world(1, sigma + np.pi/2, phi),np.array([-np.sin(sigma)*np.sin(phi),np.sin(sigma)*np.cos(phi), 0])]

cdef void sphere2cart(double[:] sphere, double[:] cart) nogil:
	cdef double sin_theta
	sin_theta = sin(sphere[0])
	cart[0] = cos(sphere[1]) * sin_theta
	cart[1] = sin(sphere[1]) * sin_theta
	cart[2] = cos(sphere[0])


cdef int inverse(double[:,:] A, double[:] WORKER, int [:] IPIV) nogil except *:
	"""
		Calculates inverse of matrix A. No boundary checks are performed.
	Parameters
	----------
	A: NxN Matrix double
		Holds the Mattrix to invert and is also the return value with the inverted matrix.
	Worker: NxN Martix double
	IPIV: N Matrix int64
	Returns
		0 if the matrix is successfully inverted
		i if there is a singularity.
	-------
	"""
	cdef int INFO = 0
	cdef int LWORK = A.shape[0] * A.shape[0]
	INFO = LAPACKE_dgetrf(CblasRowMajor, A.shape[0], A.shape[0], &A[0,0], A.shape[0], &IPIV[0])
	if INFO != 0:
		return INFO
	INFO = LAPACKE_dgetri_work(CblasRowMajor, A.shape[0], &A[0,0], A.shape[0], &IPIV[0], &WORKER[0], LWORK)
	return 0



cdef void ddiagonal(double * M, double[:] v, int columns, int rows):
	for i in range(rows):
		for j in range(columns):
			if i == j:
				M[i*(columns+1)] = v[i%v.shape[0]]
			else:
				M[i * columns + j] = 0


cdef void special_mat_mul(double[:,:] M, double[:,:] A, double[:] B, double[:,:] C, double scale) nogil except *:
	"""
	Calc
		M = A*diag(B)*C
	Parameters
	----------
	M
	A
	B
	C

	Returns
	-------
	"""

	cdef int i, j, k
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			M[i,  j] = 0
			for k in range(A.shape[1]):
				M[i, j] += scale * A[i, k] * B[k] * C[j, k]



cdef double max_c(double a, double b) nogil:
	if a < b:
		return b
	else:
		return a

cdef bint bigger(double[:] a, double[:] b) nogil:
	cdef int i
	for i in range(a.shape[0]):
		if a[i] < b[i]:
			return False
	else:
		return True

cdef bint smaller(double[:] a, double[:] b) nogil:
	cdef int i
	for i in range(a.shape[0]):
		if a[i] > b[i]:
			return False
	else:
		return True

cdef double dist(double[:] v, double[:]  w) nogil:
	cdef int i, n =v.shape[0]
	cdef double res = 0
	for i in range(n):
		res += pow(v[i] - w[i], 2)
	return pow(res, 0.5)

cdef bint point_validator(double v, int a) nogil:
	if a != 0:
		if v == 0:
			return False
	if v != v:
		return False
	return True

@cython.cdivision(True)
cdef double angle_deg(double[:] vec1, double[:] vec2) nogil:
	""" Calculate the angle between two vectors

	Parameters
	----------
	vec1 :
	vec2 : two vectors of same dimensions.

	Returns
	-------

	"""
	if vec1.shape[0] != vec2.shape[0]:
		printf("Wrong dimensions \n")
		return 0
	if sum_c(vec1) != sum_c(vec1) or sum_c(vec2) != sum_c(vec2) or norm(vec1) == 0 or norm(vec2) == 0:
		return 90
	if acos(clip(scalar(vec1, vec2) / (norm(vec1) * (norm(vec2))), -1,1)) * 180/ pi < 0:
		printf("Something is wrong \n")
		return 0

	return acos(clip(scalar(vec1, vec2) / (norm(vec1) * (norm(vec2))), -1,1)) * 180 / pi

cdef void matrix_mult(double[:] res, double[:,:] A, double[:] v) nogil:
	"""Easy matrix multiplication.
	:param res: Result vector size m
	:param A: Matrix size mxn
	:param v: vector size n
	:return: res size n
	"""

	cdef int i, j, n = A.shape[0], m=A.shape[1]
	if n != res.shape[0] or m != v.shape[0]:
		printf("Wrong dimensions\n")

	for i in range(n):
		for j in range(m):
			res[i] += A[i,j] * v[j]

cdef int argmax(double[:] a) nogil:
	cdef int index = 0, n = a.shape[0], i = 0
	cdef double m = 0
	for i in range(n):
		if m < a[i]:
			index = i
			m = a[i]
	return index


cdef double clip(double a, double min_value, double max_value) nogil:
	return min_c(max_c(a, min_value), max_value)

cdef double sum_c(double[:] v) nogil:
	cdef int i, n = v.shape[0]
	cdef double res = 0
	for i in range(n):
		res += v[i]
	return res

cdef int sum_c_int(int[:] v) nogil:
	cdef int i, n = v.shape[0]
	cdef int res = 0
	for i in range(n):
		res += v[i]
	return res

cdef void floor_pointwise_matrix(double[:,:] res, double[:,:] v) nogil:
	cdef int i, j, n = v.shape[0], m = v.shape[1]
	for i in range(n):
		for j in range(m):
			res[i, j] = floor(v[i, j])

cdef void add_pointwise(double[:,:] res, double[:,:] v, double[:] w) nogil:
	cdef int i, j, n = v.shape[0], m = v.shape[1]
	for i in range(n):
		for j in range(m):
			res[i,j] = v[i,j] + w[j]

cdef double scalar(double[:] v, double[:] w) nogil:
	cdef int i, k = v.shape[0]
	cdef double res = 0
	for i in range(k):
		res += v[i] * w[i]
	return res

cdef double norm(double[:] v) nogil:
	return pow(scalar(v,v), 1/2)


cdef double scalar_minus(double[:] v, double[:] w) nogil:
	cdef int i, k = v.shape[0]
	cdef double res = 0
	for i in range(k):
		res = -v[i] * w[i]
	return res

cdef void mult_with_scalar(double[:] res, double s, double[:] v) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		res[i] = s*v[i]

cdef void mult_with_scalar_int(int[:] res, int s, int [:] v) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		res[i] = s*v[i]


cdef void add_vectors(double[:] res, double[:] v, double[:] w) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		res[i] = v[i] + w[i]

cdef void sub_vectors(double[:] res, double[:] v, double[:] w) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		res[i] = v[i] - w[i]

cdef void sub_vectors_int(int[:] res, int[:] v, int[:] w) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		res[i] = v[i] - w[i]

cdef double fak(int l) nogil:
	cdef int i
	cdef int x = 1
	for i in range(1,l+1):
		x *= i
	return x

@cython.cdivision(True)
cdef double binom(int n,int k) nogil:
	return fak(n)/(fak(k)*fak(n-k))

cdef void set_zero_matrix(double[:,:] v) nogil:
	cdef int i, j, m = v.shape[0], n = v.shape[1]
	for i in range(m):
		for j in range(n):
			v[i, j] = 0

cdef void set_zero_matrix_int(int[:,:] v) nogil:
	cdef int i, j, m = v.shape[0], n = v.shape[1]
	for i in range(m):
		for j in range(n):
			v[i, j] = 0

cdef void set_zero_vector(double[:] v) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		v[i] = 0

cdef void set_zero_vector_int(int[:] v) nogil:
	cdef int i, k = v.shape[0]
	for i in range(k):
		v[i] = 0

cdef void set_zero_3d(double[:,:,:] v) nogil:
	cdef int i, j, k, m = v.shape[0], n = v.shape[1], o = v.shape[2]
	for i in range(m):
		for j in range(n):
			for k in range(o):
				v[i,j,k] = 0

