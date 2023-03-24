/*	https://www.github.com/david-cortes/findblas
	This is intended to be used along with the 'build_ext_with_blas' class in the 'findblas' package.
	Will include correct BLAS header for the vendor version. Ideally it should be either "mkl.h" or "cblas.h",
	but there can be some weird installs missing them. If no header is present, will declare
	the function protoypes here, which might be unreliable.

	This way, you only need to include "findblas.h" and it will choose the correct BLAS
	header for you, or will declare the basic functions if no header is present.
	*/

#if defined MKL_OWN_INCL_CBLAS
  #include "mkl_cblas.h"
#elif defined(USE_MKL) && !defined(NO_CBLAS_HEADER)
  #include "mkl.h"
#elif defined OPENBLAS_OWN_INCL
  #include "cblas-openblas.h"
#elif defined GSL_OWN_INCL_CBLAS
  #include "gsl_cblas.h"
#elif defined INCL_CBLAS
  #include "cblas.h"
#elif defined INCL_BLAS
  #include "blas.h"
#else

/*	Define prototypes for the entire cblas catalog - most of this is copy-paste from OpenBLAS with automatic substitutions
	https://github.com/xianyi/OpenBLAS */
#ifdef __cplusplus
extern "C" {
#endif

#if !defined(_FINDBLAS_MOCK_DEFINE) && !defined(CBLAS_H)
#include <stddef.h>
#define CBLAS_H
#define CBLAS_INDEX size_t
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

/* Inclusion of a standard header file is needed for definition of __STDC_*
   predefined macros with some compilers (e.g. GCC 4.7 on Linux).  This occurs
   as a side effect of including either <features.h> or <stdc-predef.h>. */
#include <stdio.h>

/* C99 supports complex floating numbers natively, which GCC also offers as an
   extension since version 3.0.  If neither are available, use a compatible
   structure as fallback (see Clause 6.2.5.13 of the C99 standard). */
#if ((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L || \
      (__GNUC__ >= 3 && !defined(__cplusplus))) && !(defined(FORCE_OPENBLAS_COMPLEX_STRUCT))) && !defined(_MSC_VER)

#ifndef __cplusplus
  #include <complex.h>
#endif
  typedef float _Complex openblas_complex_float;
  typedef double _Complex openblas_complex_double;
  #define openblas_make_complex_float(real, imag)    ((real) + ((imag) * _Complex_I))
  #define openblas_make_complex_double(real, imag)   ((real) + ((imag) * _Complex_I))
  #define openblas_complex_float_real(z)             (creal(z))
  #define openblas_complex_float_imag(z)             (cimag(z))
  #define openblas_complex_double_real(z)            (creal(z))
  #define openblas_complex_double_imag(z)            (cimag(z))
#else
  #define OPENBLAS_COMPLEX_STRUCT
  typedef struct { float real, imag; } openblas_complex_float;
  typedef struct { double real, imag; } openblas_complex_double;
  #define openblas_make_complex_float(real, imag)    {(real), (imag)}
  #define openblas_make_complex_double(real, imag)   {(real), (imag)}
  #define openblas_complex_float_real(z)             ((z).real)
  #define openblas_complex_float_imag(z)             ((z).imag)
  #define openblas_complex_double_real(z)            ((z).real)
  #define openblas_complex_double_imag(z)            ((z).imag)
#endif /* OPENBLAS_CONFIG_H */
#endif /* _FINDBLAS_MOCK_DEFINE */

float  cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy);
double cblas_dsdot (const int n, const float *x, const int incx, const float *y, const int incy);
float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy);
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);

openblas_complex_float  cblas_cdotu(const int n, const float  *x, const int incx, const float  *y, const int incy);
openblas_complex_float  cblas_cdotc(const int n, const float  *x, const int incx, const float  *y, const int incy);
openblas_complex_double cblas_zdotu(const int n, const double *x, const int incx, const double *y, const int incy);
openblas_complex_double cblas_zdotc(const int n, const double *x, const int incx, const double *y, const int incy);

void  cblas_cdotu_sub(const int n, const float  *x, const int incx, const float  *y, const int incy, openblas_complex_float  *ret);
void  cblas_cdotc_sub(const int n, const float  *x, const int incx, const float  *y, const int incy, openblas_complex_float  *ret);
void  cblas_zdotu_sub(const int n, const double *x, const int incx, const double *y, const int incy, openblas_complex_double *ret);
void  cblas_zdotc_sub(const int n, const double *x, const int incx, const double *y, const int incy, openblas_complex_double *ret);

float  cblas_sasum (const int n, const float  *x, const int incx);
double cblas_dasum (const int n, const double *x, const int incx);
float  cblas_scasum(const int n, const float  *x, const int incx);
double cblas_dzasum(const int n, const double *x, const int incx);

float  cblas_snrm2 (const int N, const float  *X, const int incX);
double cblas_dnrm2 (const int N, const double *X, const int incX);
float  cblas_scnrm2(const int N, const float  *X, const int incX);
double cblas_dznrm2(const int N, const double *X, const int incX);

CBLAS_INDEX cblas_isamax(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_idamax(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_icamax(const int n, const float  *x, const int incx);
CBLAS_INDEX cblas_izamax(const int n, const double *x, const int incx);

void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy);
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
void cblas_caxpy(const int n, const float *alpha, const float *x, const int incx, float *y, const int incy);
void cblas_zaxpy(const int n, const double *alpha, const double *x, const int incx, double *y, const int incy);

void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy);
void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy);
void cblas_ccopy(const int n, const float *x, const int incx, float *y, const int incy);
void cblas_zcopy(const int n, const double *x, const int incx, double *y, const int incy);

void cblas_sswap(const int n, float *x, const int incx, float *y, const int incy);
void cblas_dswap(const int n, double *x, const int incx, double *y, const int incy);
void cblas_cswap(const int n, float *x, const int incx, float *y, const int incy);
void cblas_zswap(const int n, double *x, const int incx, double *y, const int incy);

void cblas_srot(const int N, float *X, const int incX, float *Y, const int incY, const float c, const float s);
void cblas_drot(const int N, double *X, const int incX, double *Y, const int incY, const double c, const double  s);

void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_drotg(double *a, double *b, double *c, double *s);

void cblas_srotm(const int N, float *X, const int incX, float *Y, const int incY, const float *P);
void cblas_drotm(const int N, double *X, const int incX, double *Y, const int incY, const double *P);

void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);

void cblas_sscal(const int N, const float alpha, float *X, const int incX);
void cblas_dscal(const int N, const double alpha, double *X, const int incX);
void cblas_cscal(const int N, const float *alpha, float *X, const int incX);
void cblas_zscal(const int N, const double *alpha, double *X, const int incX);
void cblas_csscal(const int N, const float alpha, float *X, const int incX);
void cblas_zdscal(const int N, const double alpha, double *X, const int incX);

void cblas_sgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const float alpha, const float  *a, const int lda,  const float  *x, const int incx,  const float beta,  float  *y, const int incy);
void cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,  double  *y, const int incy);
void cblas_cgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const float *alpha, const float  *a, const int lda,  const float  *x, const int incx,  const float *beta,  float  *y, const int incy);
void cblas_zgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
		 const double *alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double *beta,  double  *y, const int incy);

void cblas_sger (const enum CBLAS_ORDER order, const int M, const int N, const float   alpha, const float  *X, const int incX, const float  *Y, const int incY, float  *A, const int lda);
void cblas_dger (const enum CBLAS_ORDER order, const int M, const int N, const double  alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);
void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N, const float  *alpha, const float  *X, const int incX, const float  *Y, const int incY, float  *A, const int lda);
void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N, const float  *alpha, const float  *X, const int incX, const float  *Y, const int incY, float  *A, const int lda);
void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N, const double *alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);
void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N, const double *alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);

void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);
void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);

void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);
void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const float *A, const int lda, float *X, const int incX);
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda, double *X, const int incX);

void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda);
void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda);
void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *A, const int lda);
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, double *A, const int lda);

void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,const int N, const float alpha, const float *X,
                const int incX, const float *Y, const int incY, float *A, const int lda);
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X,
                const int incX, const double *Y, const int incY, double *A, const int lda);
void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float *alpha, const float *X, const int incX,
                const float *Y, const int incY, float *A, const int lda);
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double *alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda);

void cblas_sgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void cblas_cgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const float *alpha, const float *A, const int lda, const float *X, const int incX, const float *beta, float *Y, const int incY);
void cblas_zgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double *alpha, const double *A, const int lda, const double *X, const int incX, const double *beta, double *Y, const int incY);

void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);


void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);
void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);

void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);
void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, float *X, const int incX);
void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, double *X, const int incX);

void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);
void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);

void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);
void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);

void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float *alpha, const float *A,
                 const int lda, const float *X, const int incX, const float *beta, float *Y, const int incY);
void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double *alpha, const double *A,
                 const int lda, const double *X, const int incX, const double *beta, double *Y, const int incY);


void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *Ap,
                 const float *X, const int incX, const float beta, float *Y, const int incY);
void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *Ap,
                 const double *X, const int incX, const double beta, double *Y, const int incY);

void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *Ap);
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, double *Ap);

void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, float *A);
void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X,const int incX, double *A);

void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float alpha, const float *X, const int incX, const float *Y, const int incY, float *A);
void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A);
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const float *alpha, const float *X, const int incX, const float *Y, const int incY, float *Ap);
void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const double *alpha, const double *X, const int incX, const double *Y, const int incY, double *Ap);

void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K,
		 const float *alpha, const float *A, const int lda, const float *X, const int incX, const float *beta, float *Y, const int incY);
void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, const int K,
		 const double *alpha, const double *A, const int lda, const double *X, const int incX, const double *beta, double *Y, const int incY);

void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
		 const float *alpha, const float *Ap, const float *X, const int incX, const float *beta, float *Y, const int incY);
void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
		 const double *alpha, const double *Ap, const double *X, const int incX, const double *beta, double *Y, const int incY);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float *beta, float *C, const int ldc);
void cblas_cgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float *beta, float *C, const int ldc);
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double *beta, double *C, const int ldc);
void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double *beta, double *C, const int ldc);


void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float *beta, float *C, const int ldc);
void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double *beta, double *C, const int ldc);

void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc);
void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const float *alpha, const float *A, const int lda, const float *beta, float *C, const int ldc);
void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		 const int N, const int K, const double *alpha, const double *A, const int lda, const double *beta, double *C, const int ldc);

void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float *beta, float *C, const int ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
		  const int N, const int K, const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double *beta, double *C, const int ldc);

void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float *alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double *alpha, const double *A, const int lda, double *B, const int ldb);

void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const float *alpha, const float *A, const int lda, float *B, const int ldb);
void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N, const double *alpha, const double *A, const int lda, double *B, const int ldb);

void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float *beta, float *C, const int ldc);
void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double *beta, double *C, const int ldc);

void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float *A, const int lda, const float beta, float *C, const int ldc);
void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc);

void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const float *alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const double *alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);

void cblas_xerbla(int p, char *rout, char *form, ...);

/*** BLAS extensions ***/

void cblas_saxpby(const int n, const float alpha, const float *x, const int incx,const float beta, float *y, const int incy);

void cblas_daxpby(const int n, const double alpha, const double *x, const int incx,const double beta, double *y, const int incy);

void cblas_caxpby(const int n, const float *alpha, const float *x, const int incx,const float *beta, float *y, const int incy);

void cblas_zaxpby(const int n, const double *alpha, const double *x, const int incx,const double *beta, double *y, const int incy);

void cblas_somatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float calpha, const float *a, 
		     const int clda, float *b, const int cldb); 
void cblas_domatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double calpha, const double *a,
		     const int clda, double *b, const int cldb); 
void cblas_comatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float* calpha, const float* a, 
		     const int clda, float*b, const int cldb); 
void cblas_zomatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double* calpha, const double* a, 
		     const int clda,  double *b, const int cldb); 

void cblas_simatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float calpha, float *a, 
		     const int clda, const int cldb); 
void cblas_dimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double calpha, double *a,
		     const int clda, const int cldb); 
void cblas_cimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const float* calpha, float* a, 
		     const int clda, const int cldb); 
void cblas_zimatcopy(const enum CBLAS_ORDER CORDER, const enum CBLAS_TRANSPOSE CTRANS, const int crows, const int ccols, const double* calpha, double* a, 
		     const int clda, const int cldb); 

void cblas_sgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const float calpha, float *a, const int clda, const float cbeta, 
		  float *c, const int cldc); 
void cblas_dgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const double calpha, double *a, const int clda, const double cbeta, 
		  double *c, const int cldc); 
void cblas_cgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const float *calpha, float *a, const int clda, const float *cbeta, 
		  float *c, const int cldc); 
void cblas_zgeadd(const enum CBLAS_ORDER CORDER,const int crows, const int ccols, const double *calpha, double *a, const int clda, const double *cbeta, 
		  double *c, const int cldc); 

#ifdef __cplusplus
}
#endif

#endif /*!defined(_FINDBLAS_MOCK_DEFINE) && !defined(CBLAS_H) */
