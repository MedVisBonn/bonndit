cdef void hota_4o3d_sym_eval(double[:], double, double[:]) nogil
cdef double hota_4o3d_sym_tsp(double[:], double[:]) nogil
cdef double hota_4o3d_sym_norm(double[:]) nogil
cdef hota_8o3d_sym_eval_cons(double[:,:], double[:,:])

cdef double hota_6o3d_sym_s_form(double[:], double[:]) nogil
cdef void hota_6o3d_sym_eval(double[:] , double , double[:] ) nogil
cdef void hota_4o3d_sym_make_iso(double[:], double) nogil
cdef void hota_4o3d_sym_v_form(double[:], double[:], double[:]) nogil
cdef double hota_4o3d_mean(double[:]) nogil
cdef double hota_4o3d_sym_s_form(double[:], double[:]) nogil
cdef void hota_8o3d_sym_eval(double[:], double, double[:]) nogil
cdef int[:] order_4_mult
cdef void hota_6o3d_sym_eval(double[:], double, double[:]) nogil
cdef double hota_6o3d_sym_tsp(double[:], double[:]) nogil
cdef double hota_6o3d_sym_norm(double[:]) nogil
cpdef void hota_6o3d_hessian(double[:,:], double[:], double[:])

cdef void hota_6o3d_sym_make_iso(double[:], double) nogil
cdef void hota_6o3d_sym_v_form(double[:], double[:], double[:]) nogil
cdef double hota_6o3d_mean(double[:]) nogil
cdef int[:] order_6_mult
cpdef void hota_6o3d_hessian_sh(double[:,:], double[:,:], double[:], double[:])