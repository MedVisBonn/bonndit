#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
cimport cython
import numpy as np
from libc.math cimport sqrt, fabs
from .cython_helpers cimport sub_vectors, mult_with_scalar, add_vectors, scalar
from .structures cimport TijkRefineRank1Parm, TijkRefineRankkParm, DIM
from .cython_helpers cimport set_zero_matrix, set_zero_vector
from .hota cimport hota_4o3d_sym_eval, hota_4o3d_sym_norm, hota_4o3d_sym_make_iso, hota_4o3d_sym_v_form, \
    hota_4o3d_mean, hota_4o3d_sym_s_form


DTYPE = np.float64

cdef double[:] _candidates_3d_d = np.array([
    -0.546405, 0.619202, 0.563943,
    -0.398931,-0.600006, 0.693432,
    0.587973, 0.521686, 0.618168,
    0.055894,-0.991971,-0.113444,
    -0.666933,-0.677984, 0.309094,
    0.163684, 0.533013, 0.830123,
    0.542826, 0.133898, 0.829102,
    -0.074751,-0.350412, 0.933608,
    0.845751, -0.478624,-0.235847,
    0.767148,-0.610673, 0.196372,
    -0.283810, 0.381633, 0.879663,
    0.537228,-0.616249, 0.575868,
    -0.711387, 0.197778, 0.674398,
    0.886511, 0.219025, 0.407586,
    0.296061, 0.842985, 0.449136,
    -0.937540,-0.340990, 0.068877,
    0.398833, 0.917023, 0.000835,
    0.097278,-0.711949, 0.695460,
    -0.311534, 0.908623,-0.278121,
    -0.432043,-0.089758, 0.897375,
    -0.949980, 0.030810, 0.310788,
    0.146722,-0.811981,-0.564942,
    -0.172201,-0.908573, 0.380580,
    0.507209,-0.848578,-0.150513,
    -0.730808,-0.654136,-0.194999,
    0.077744, 0.094961, 0.992441,
    0.383976,-0.293959, 0.875300,
    0.788208,-0.213656, 0.577130,
    -0.752333,-0.301447, 0.585769,
    -0.975732, 0.165497,-0.143382])

cdef int _max_candidates_3d=30

cdef void init_max_3d(double[:] s, double[:] v, double[:] tens) nogil:
    """
    Find best initial direction on sphere for rank1 approximation of a fourth order tensor.
    """
    cdef double norm_max, val
    cdef int i
    s[0] = hota_4o3d_sym_s_form(tens, _candidates_3d_d[:3])
    norm_max = s[0]
    v[:] = _candidates_3d_d[0: DIM]
    for i in range(1, _max_candidates_3d):
        val = hota_4o3d_sym_s_form(tens, _candidates_3d_d[DIM*i: DIM*(i+1)])
        if val>norm_max:
            norm_max = val
            s[0] = val
            v[:] = _candidates_3d_d[DIM*i: DIM*(i+1)]


cdef double refine_rankk_3d(double[:] ls, double[:,:] vs, double[:,:] tens, double[:] res, double resnorm,
                                   double orignorm, int k, double[:] valsec, double[:] val, double[:] der,
                                   double[:] testv,double[:] anisoten, double[:] isoten): # nogil except *:
    """
    Caluculate best rank k approximation of an 4th order tensor
    """
    cdef double newnorm = resnorm
    cdef int i, j
    while True:
        resnorm = newnorm
        for i in range(k):
            # add a new term
            if ls[i] == 0:
                if TijkRefineRankkParm.pos:
                    init_max_3d(ls[i: i+1], vs[:,i], res)
                else:
                    init_rank1_3d(ls[i:i+1], vs[:,i], res)
            # refine an existing term
            else:
                for j in range(15):
                    res[j] += tens[i,j]
                ls[i] = hota_4o3d_sym_s_form(res, vs[:,i])
                if TijkRefineRankkParm.pos and ls[i] < 0:
                    ls[i] = 0.1
                  #  set_zero_vector(ls)
                  #  set_zero_matrix(vs)
                  #  return 0

                    #init_max_3d(ls[i: i+1], vs[:,i], res)
    #        refine_rank1_sum_3d(ls[i:i+1], vs[:, i])
            refine_rank1_3d(ls[i: i+1], vs[:,i], res, der, testv, anisoten, isoten)
            if not TijkRefineRankkParm.pos==1 or ls[i]>0:
                hota_4o3d_sym_eval(tens[i, :], ls[i], vs[:,i])
                sub_vectors(res, res, tens[i,:])

            else:
                ls[i] = 0.1
                hota_4o3d_sym_eval(tens[i, :], ls[i], vs[:,i])
                sub_vectors(res, res, tens[i,:])
        newnorm = hota_4o3d_sym_norm(res)
        if not (newnorm > TijkRefineRankkParm.eps_res and resnorm - newnorm > TijkRefineRankkParm.eps_impr*orignorm):
            #print([x for x in res], resnorm, newnorm, orignorm)
            break



cdef double approx_initial(double[:] ls, double[:,:] vs, double[:,:] tens, double[:] ten, int k,
double[:] valsec, double[:] val, double[:] der, double[:] testv,double[:] anisoten, double[:] isoten): # nogil:
    cdef double orignorm, newnorm
    cdef double[:] res
    orignorm = newnorm = hota_4o3d_sym_norm(ten)
    res = ten
    if orignorm < TijkRefineRankkParm.eps_res or k == 0:
        pass
    else:
        refine_rankk_3d(ls, vs, tens, res, newnorm, orignorm, k, valsec, val, der, testv, anisoten, isoten)


cdef double init_rank1_3d(double[:] s, double[:] v, double[:] tens):#  nogil:
    """
    Find best initial direction on sphere for rank1 approximation of a fourth order tensor. Here the abs value is
    used.
    """

    cdef int i
    cdef double absval, absmax=-1, val
  #  with gil:
   #     print(*s, *v, DIM)
    for i in range(_max_candidates_3d):
        val = hota_4o3d_sym_s_form(tens, _candidates_3d_d[DIM*i:DIM*(i+1)])
        absval = fabs(val)
        if absval>absmax:
            absmax=absval
            s[:] = val
            v[:] = _candidates_3d_d[DIM*i:DIM*(i+1)]

"""@cython.cdivision(True)"""
cdef int refine_rank1_3d(double[:] s, double[:] v, double[:] tens, double[:] der, double[:] testv,double[:] anisoten,
                         double[:] isoten): #  nogil except *:
    """
    Gradient descent with armijo stepsize
    """

    cdef int sign = 1 if s[0]>0 else -1
    cdef double iso = hota_4o3d_mean(tens)
    hota_4o3d_sym_make_iso(isoten, iso)
    cdef int i, armijoct,  k=tens.shape[0]
    cdef double anisonorm, anisonorminv, alpha, beta, oldval, val
    for i in range(k):
        anisoten[i] = tens[i] - isoten[i]
    anisonorm = hota_4o3d_sym_norm(anisoten)
    if anisonorm < TijkRefineRank1Parm.eps_start:
        return 1
    else:
        anisonorminv = 1/anisonorm
    alpha = beta = TijkRefineRank1Parm.beta*anisonorminv
    oldval = s[0] - iso
    _4o3d_sym_grad(der, anisoten, v)

    while True:
        armijoct = 0
        while True:
            armijoct += 1
            der_len=sqrt(scalar(der,der))
            if armijoct>TijkRefineRank1Parm.maxTry:
                return 2
            mult_with_scalar(testv, sign*alpha, der)
            add_vectors(testv, v, testv)
            dist = sqrt(scalar(testv,testv))

            mult_with_scalar(testv, 1/dist,testv) #Vektor
            dist = 1 - sqrt(scalar(v,testv))
            val = hota_4o3d_sym_s_form(anisoten, testv)
            if sign*val >= sign*oldval + TijkRefineRank1Parm.sigma*der_len*dist:
                v[:] = testv
                s[0] = val + iso
                _4o3d_sym_grad(der, anisoten, v)
                if alpha < beta:
                    alpha /= TijkRefineRank1Parm.gamma
                break
            alpha *= TijkRefineRank1Parm.gamma
        if sign*(val-oldval) <= TijkRefineRank1Parm.eps_impr *anisonorm:
            break
        oldval = val
    return 1

#cdef int refine_rank1_sum_3d(double[:] s, double[:] v, double[:] tens, double[:] der, double[:] testv,double[:] anisoten,
#                         double[:] isoten) nogil except *:
#    """
#    Gradient descent with armijo stepsize
#    """
#
#    cdef int sign = 1 if s[0]>0 else -1
#    for j in range(tens.shape[1]):
#        iso[j] = hota_4o3d_mean(tens[:,j])
#        hota_4o3d_sym_make_iso(isoten[:,j], iso[j])
#    cdef double iso = hota_4o3d_mean(tens)
#
#    cdef int i, armijoct,  k=tens.shape[0]
#    cdef double anisonorm, anisonorminv, alpha, beta, oldval, val
#    for j in range(tens.shape[1]):
#        for i in range(k):
#            anisoten[i,j] = tens[i,j] - isoten[i,j]
#        anisonorm[j] = hota_4o3d_sym_norm(anisoten[:,j])
#    if sum_c(anisonorm) < TijkRefineRank1Parm.eps_start:
#        return 1
#    else:
#        anisonorminv = 1/sum_c(anisonorm)
#    alpha = beta = TijkRefineRank1Parm.beta*anisonorminv
#    oldval = s[0] - sum_c(iso)
#
#    _4o3d_sym_grad(der, anisoten, v)
#
#    while True:
#        armijoct = 0
#        while True:
#            armijoct += 1
#            der_len=sqrt(scalar(der,der))
#            if armijoct>TijkRefineRank1Parm.maxTry:
#                return 2
#            mult_with_scalar(testv, sign*alpha, der)
#            add_vectors(testv, v, testv)
#            dist = sqrt(scalar(testv,testv))
#
#            mult_with_scalar(testv, 1/dist,testv) #Vektor
#            dist = 1 - sqrt(scalar(v,testv))
#            val = hota_4o3d_sym_s_form(anisoten, testv)
#            if sign*val >= sign*oldval + TijkRefineRank1Parm.sigma*der_len*dist:
#                v[:] = testv
#                s[0] = val + iso
#                _4o3d_sym_grad(der, anisoten, v)
#                if alpha < beta:
#                    alpha /= TijkRefineRank1Parm.gamma
#                break
#            alpha *= TijkRefineRank1Parm.gamma
#        if sign*(val-oldval) <= TijkRefineRank1Parm.eps_impr *anisonorm:
#            break
#        oldval = val
#    return 1

cdef void _4o3d_sym_grad(double[:] res, double[:] a, double[:] v) nogil:
    """
    Calculate spherical gradient of symetric 4th order tensor on sphere.
    """
    hota_4o3d_sym_v_form(res, a, v)
    cdef int i, k = 3
    for i in range(k):
        res[i] *= 4
    cdef double proj = scalar(res, v)
    for i in range(k):
        res[i] -= proj * v[i]
