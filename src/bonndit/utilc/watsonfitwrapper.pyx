#cython: language_level=3, boundscheck=False, wraparound=False


#import numpy as np
#cimport numpy as np
#np.import_array()

#cdef extern from "watsonfit.h":
#        void sh_watson_coeffs(double, double *, int)
#        void minimize_watson_mult_o4(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#        void minimize_watson_mult_o6(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#        void minimize_watson_mult_o8(double* parameters, double* signal_p, double* est_signal_p, double* dipy_v_p, double* pysh_v_p, double* rot_pysh_v_p, double* angles_v_p, double* loss_p, int amount, int num_of_dir_p, int no_spread) nogil
#        void SHRotateRealCoef(double *, double *, int, double *, double *)
#        void map_dipy_to_pysh_o4(double *, double *)
#        void map_pysh_to_dipy_o4(double *, double *)

def mw_openmp_mult_p(double[:,:] x, double[:,:] signal, double[:,:] est_signal, double[:,:] dipy_v, double[:,:,:,:] pysh_v, double[:,:,:,:] rot_pysh_v, double[:,:] angles_v, double[:] loss, int amount, int order, int num_of_dir, int no_spread):
        if order == 4:
                mw_openmp_mult_o4(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)
        elif order == 6:
                mw_openmp_mult_o8(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)
        elif order == 8:
                mw_openmp_mult_o8(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)

cdef void mw_openmp_mult(double[:,:] x, double[:,:] signal, double[:,:] est_signal, double[:,:] dipy_v, double[:,:,:,:] pysh_v, double[:,:,:,:] rot_pysh_v, double[:,:] angles_v, double[:] loss, int amount, int order, int num_of_dir, int no_spread) nogil:
        if order == 4:
                mw_openmp_mult_o4(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)
        elif order == 6:
                mw_openmp_mult_o8(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)
        elif order == 8:
                mw_openmp_mult_o8(x,signal,est_signal,dipy_v,pysh_v,rot_pysh_v,angles_v,loss,amount,num_of_dir,no_spread)

cdef void mw_openmp_mult_o4(double[:,:] x, double[:,:] signal, double[:,:] est_signal, double[:,:] dipy_v, double[:,:,:,:] pysh_v, double[:,:,:,:] rot_pysh_v, double[:,:] angles_v, double[:] loss, int amount, int num_of_dir, int no_spread) nogil:
        minimize_watson_mult_o4(&x[0,0],&signal[0,0],&est_signal[0,0],&dipy_v[0,0],&pysh_v[0,0,0,0],&rot_pysh_v[0,0,0,0],&angles_v[0,0],&loss[0],amount,num_of_dir,no_spread)

cdef void mw_openmp_mult_o6(double[:,:] x, double[:,:] signal, double[:,:] est_signal, double[:,:] dipy_v, double[:,:,:,:] pysh_v, double[:,:,:,:] rot_pysh_v, double[:,:] angles_v, double[:] loss, int amount, int num_of_dir, int no_spread) nogil:
        minimize_watson_mult_o6(&x[0,0],&signal[0,0],&est_signal[0,0],&dipy_v[0,0],&pysh_v[0,0,0,0],&rot_pysh_v[0,0,0,0],&angles_v[0,0],&loss[0],amount,num_of_dir,no_spread)

cdef void mw_openmp_mult_o8(double[:,:] x, double[:,:] signal, double[:,:] est_signal, double[:,:] dipy_v, double[:,:,:,:] pysh_v, double[:,:,:,:] rot_pysh_v, double[:,:] angles_v, double[:] loss, int amount, int num_of_dir, int no_spread) nogil:
        minimize_watson_mult_o8(&x[0,0],&signal[0,0],&est_signal[0,0],&dipy_v[0,0],&pysh_v[0,0,0,0],&rot_pysh_v[0,0,0,0],&angles_v[0,0],&loss[0],amount,num_of_dir,no_spread)

cdef void c_sh_rotate_real_coef(double* x, double* y, int z, double* a, double* b):
        SHRotateRealCoef(x,y,z,a,b)
#
cdef void c_sh_rotate_real_coef_fast(double* x, int x_space, double* y, int y_space, int z, double* a):
        SHRotateRealCoefFast(x,x_space, y,y_space, z,a)
#
cdef void c_map_dipy_to_pysh_o4(double* x, double* y):
       map_dipy_to_pysh_o4(x, y)
cdef void c_map_pysh_to_dipy_o4(double* x, double* y):
        map_pysh_to_dipy_o4(x,y)
cdef void c_sh_watson_coeffs(double x, double* y, int z):
        sh_watson_coeffs(x, y, z)
#
cdef void c_map_pysh_to_dipy_o4_scaled(double scaled, double* x, double* y, int i):
        map_pysh_to_dipy_o4_scaled(scaled, x, y, i)
#
cpdef void p_sh_rotate_real_coef(double[:] x, double[:] y, int z, double[:] a, double[:,:,:] b):
        SHRotateRealCoef(&x[0],&y[0],z,&a[0],&b[0][0][0])
#
cpdef void p_map_dipy_to_pysh_o4(double[:] x, double[:] y):
       map_dipy_to_pysh_o4(&x[0], &y[0])
cpdef void p_map_pysh_to_dipy_o4(double[:] x, double[:] y):
        map_pysh_to_dipy_o4(&x[0],&y[0])
cpdef void p_sh_watson_coeffs(double x, double[:] y, int z):
        sh_watson_coeffs(x, &y[0], z)
#