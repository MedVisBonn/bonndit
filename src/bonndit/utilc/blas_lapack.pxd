cdef extern from "mkl.h":
	enum CBLAS_LAYOUT:
		CblasRowMajor = 101
		CblasColMajor = 102
	enum CBLAS_TRANSPOSE:
		CblasNoTrans = 111
		CblasTrans = 112
		CblasConjTrans = 113
	void cblas_dgemm(const CBLAS_LAYOUT Layout , const CBLAS_TRANSPOSE transa , const CBLAS_TRANSPOSE transb , const int m , const int n , const int k , const double alpha , const double *a , const int lda , const double *b , const int ldb , const double beta , double * c , const int ldc ) nogil except *
	void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) nogil except *
	void cblas_daxpy (const int n, const double a, const double *x, const int incx, double *y, const int incy) nogil except *
	void cblas_dscal(const int n, const double a, double *x, const int incx) nogil except *
	double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) nogil except *
	void cblas_dgemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, const int m, const int n, const double alpha, const double *a, const int lda, const double *x, const int incx, const double beta, double * y, const int incy) nogil except *
	double cblas_dnrm2 (const int n, const double *x, const int incx) nogil except *
	int LAPACKE_dpotrf(int matrix_layout, char uplo, int n, double * a, int lda) nogil except *
	int LAPACKE_dgetrf(int matrix_layout, int m, int n, double * a, int lda, int * ipiv) nogil except *
	int LAPACKE_sgesv(int matrix_layout, int n ,int nhs, double * a, int lda, int * ipiv, double *a, int ldb) nogil except *
	int LAPACKE_dgetri_work(int matrix_layout, int n, double * a, int lda, const int * ipiv, double * WORK, const int LWORK) nogil except *
	void cblas_dswap(const int n, double *x, const int incx, double *y , const int incy) nogil except *

cdef extern from "<math.h>" nogil:
	double exp(double x)
	double sqrt(double x)
