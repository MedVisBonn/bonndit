#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cdef:
	double clip(double, double, double) nogil
	void add_pointwise(double[:,:], double[:,:], double[:]) nogil
	double angle_deg(double[:], double[:]) nogil
	void matrix_mult(double[:], double[:,:], double[:]) nogil
	double sum_c(double[:]) nogil
	void floor_pointwise_matrix(double[:,:], double[:,:]) nogil
	double norm(double[:]) nogil
	double scalar(double[:], double[:]) nogil
	double scalar_minus(double[:], double[:]) nogil
	void mult_with_scalar(double[:], double, double[:]) nogil
	void mult_with_scalar_int(int[:], int, int[:]) nogil
	void add_vectors(double[:], double[:], double[:]) nogil
	void sub_vectors(double[:], double[:], double[:]) nogil
	void sub_vectors_int(int[:], int[:], int[:]) nogil
	double fak(int) nogil
	double binom(int, int) nogil
	void set_zero_matrix(double[:,:]) nogil
	void set_zero_matrix_int(int[:,:]) nogil
	void set_zero_vector(double[:]) nogil
	void set_zero_3d(double[:,:,:]) nogil
	int sum_c_int(int[:]) nogil
	void set_zero_vector_int(int[:]) nogil
	int argmax(double[:]) nogil
	double dist(double[:], double[:]) nogil
	bint bigger(double[:], double[:]) nogil
	bint smaller(double[:], double[:]) nogil

