#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

cdef void hota_4o3d_sym_eval(double[:], double, double[:]) nogil
cdef double hota_4o3d_sym_tsp(double[:], double[:]) nogil
cdef double hota_4o3d_sym_norm(double[:]) nogil
cdef hota_8o3d_sym_eval_cons(double[:,:], double[:,:])
cdef void hota_4o3d_sym_make_iso(double[:], double) nogil
cdef void hota_4o3d_sym_v_form(double[:], double[:], double[:]) nogil
cdef double hota_4o3d_mean(double[:]) nogil
cdef double hota_4o3d_sym_s_form(double[:], double[:]) nogil
cdef void hota_8o3d_sym_eval(double[:], double, double[:]) nogil
