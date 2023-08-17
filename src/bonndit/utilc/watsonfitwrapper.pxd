#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
cdef extern from "watsonfit.h":
        void minimize_watson_mult_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil except *
        void minimize_watson_mult_o6(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil except *
        void minimize_watson_mult_o8(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil except *
        void SHRotateRealCoef(double *, double *, int, double *, double *) nogil
        void SHRotateRealCoefFast(double *, int, double *, int, int, double *) nogil
        void map_dipy_to_pysh_o4(double *, double *) nogil
        void map_pysh_to_dipy_o4(double *, double *) nogil
        void sh_watson_coeffs(double, double *, int) nogil except *
        void map_pysh_to_dipy_o4_scaled(double, double*, double*, int) nogil

cdef void mw_openmp_mult(double[:,:], double[:,:], double[:,:], double[:,:], double[:,:,:,:], double[:,:,:,:], double[:,:], double[:], int, int, int, int) nogil
cdef void mw_openmp_mult_o4(double[:,:], double[:,:], double[:,:], double[:,:], double[:,:,:,:], double[:,:,:,:], double[:,:], double[:], int, int, int) nogil
cdef void mw_openmp_mult_o6(double[:,:], double[:,:], double[:,:], double[:,:], double[:,:,:,:], double[:,:,:,:], double[:,:], double[:], int, int, int) nogil
cdef void mw_openmp_mult_o8(double[:,:], double[:,:], double[:,:], double[:,:], double[:,:,:,:], double[:,:,:,:], double[:,:], double[:], int, int, int) nogil
cdef void c_sh_rotate_real_coef(double* , double* , int , double* , double* )
cdef void c_sh_rotate_real_coef_fast(double* , int, double* , int , int , double* )
cdef void c_map_dipy_to_pysh_o4(double* , double* )
cdef void c_map_pysh_to_dipy_o4(double* , double* )
cdef void c_sh_watson_coeffs(double , double* , int )
cdef void c_map_pysh_to_dipy_o4_scaled(double, double*, double*, int)
cpdef void p_sh_rotate_real_coef(double[:] , double[:] , int , double[:] , double[:,:,:] )
cpdef void p_map_dipy_to_pysh_o4(double[:] , double[:] )
cpdef void p_map_pysh_to_dipy_o4(double[:] , double[:] )
cpdef void p_sh_watson_coeffs(double , double[:] , int )