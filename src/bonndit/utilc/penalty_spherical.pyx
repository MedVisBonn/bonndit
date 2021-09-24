#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
cimport cython
import cython as cython
from .cython_helpers cimport add_vectors, scalar, sub_vectors, mult_with_scalar, set_zero_matrix
from .hota cimport hota_4o3d_sym_eval, hota_4o3d_mean, hota_4o3d_sym_make_iso, hota_4o3d_sym_norm, \
    hota_4o3d_sym_v_form, hota_4o3d_sym_s_form
from .structures cimport TijkRefineRank1Parm, TijkRefineRankkParm
from libc.math cimport sqrt, pow

cdef double refine_average_spherical(double[:] ls, double[:,:] vs, double[:,:] tens, double[:] res, double[:,:] pen,
                                   double nu, double norm, double[:] der, double[:] testv, double[:] three_vector,
                                   double[:,:] three_matrix, double[:] one_vector, double[:] anisoten,
                                double[:] isoten,  double[:] first,  double[:] second,  double[:] w_scaled,
                                double[:] third, double[:] fourth) nogil except *:
    """
    Fit the rank1 tensors iteratively to the 4th order tensor. Continue until convergence is reached.
    """
    cdef double res_norm = norm, new_norm = norm
    cdef int i,
    # While not converged alternate about the three vectors.
    while True:
        # save old norm to check convergence.
        res_norm = new_norm
        ##Algorithm 1: 13-19
        for i in range(3):
            # refine an existing term
            # add it to residual
            add_vectors(res, res, tens[i])
            # calc the new length according to the residual.
            ls[i] = hota_4o3d_sym_s_form(res, vs[:, i])
            refine_rank1_3d(ls[i:i+1], vs[:, i], pen[:,i],  res, nu, der,  testv,
                                        three_vector, anisoten, isoten,  first,  second,  w_scaled, third, fourth)
            hota_4o3d_sym_eval(tens[i, :], ls[i], vs[:, i])
            sub_vectors(res, res, tens[i])
        ##Algorithm 1: 20 Call Algorithm 3
        new_norm = calc_norm_spherical(ls, vs, res, pen, nu, testv)
        if not (new_norm > TijkRefineRankkParm.eps_res and res_norm - new_norm >
                                                                TijkRefineRankkParm.eps_impr*norm):
           break

@cython.cdivision(True)
cdef int refine_rank1_3d(double[:] s, double[:] w, double[:] v, double[:] tens, double nu,
                            double[:] der, double[:] testv, double[:] three_vector, double[:] anisoten,
                                     double[:] isoten, double[:] first, double[:] second, double[:] w_scaled,
                                     double[:] third, double[:] fourth) nogil except *:
    """
    Gradient descent algorithm with armijoct stepsize to optimize the rank1 tensor.
    """

    cdef double iso = hota_4o3d_mean(tens)
    hota_4o3d_sym_make_iso(isoten, iso)
    cdef int i, armijoct,  k=tens.shape[0]
    cdef double anisonorm, anisonorminv, alpha, beta, val, oldval
    for i in range(k):
        anisoten[i] = tens[i]  # - isoten[i]
    anisonorm = hota_4o3d_sym_norm(anisoten) +  nu*( pow(w[0] - v[0],2) +pow(w[1] - v[1],2) + pow(w[2] - v[2],2))
    if anisonorm < TijkRefineRank1Parm.eps_start:
        return 1
    else:
        anisonorminv = 1/anisonorm
    alpha = beta = TijkRefineRank1Parm.beta*anisonorminv
    s[0] =  hota_4o3d_sym_s_form(anisoten, w)
    oldval = value_spherical(anisoten, s[0], v, w, anisoten, nu)
    cdef int sign = 1 #if oldval > 0 else -1
    #hier mehr variablen übergeben

    derivative_spherical(der, w, v, nu, fourth,anisoten,  second, third)

    while True:
        armijoct = 0
        while True:
            armijoct += 1
            der_len=sqrt(scalar(der,der))
            if armijoct>TijkRefineRank1Parm.maxTry:
                return 2
            mult_with_scalar(testv, sign*alpha, der)
            # create testvector along gradient direction
            add_vectors(testv, w, testv)
            dist = sqrt(scalar(testv,testv))
            # renorm to 1 since we are on the sphere
            mult_with_scalar(testv, 1/dist,testv) #Vektor

            dist = 1 - sqrt(scalar(w,testv))
            val = value_spherical(anisoten, s[0], v, testv, three_vector, nu)
            if sign*val >= sign*oldval + TijkRefineRank1Parm.sigma*der_len*dist:
                w[:] = testv
                s[0] = hota_4o3d_sym_s_form(anisoten, testv) #+ iso
                derivative_spherical(der, w, v, nu, fourth, anisoten, second, third)
                if alpha < beta:
                    alpha /= TijkRefineRank1Parm.gamma
                break
            alpha *= TijkRefineRank1Parm.gamma
        if sign*(val-oldval) <= TijkRefineRank1Parm.eps_impr *anisonorm:
            break
        oldval = val
    return 1


# calculate lambda as described.
#cdef double calc_lambda(double[:] w, double[:] v, double[:] tens, double nu) nogil:
#    cdef double res = hota_4o3d_sym_s_form(tens, w)
#    return (res - nu*scalar(w,v))/(1 - nu)
#

## Calculate the spherical derivative of the penalized expression
# diff_r, diff_ls, first, second, w_scaled, third, third_helper are placeholder.
cdef void derivative_spherical(double[:] res, double[:] w, double[:] v,
             double nu, double[:] diff_r, double[:] anisoten, double[:] second, double[:] third) nogil except *:
    cdef double proj, ls
    hota_4o3d_sym_v_form(diff_r, anisoten, w)
    ls = hota_4o3d_sym_s_form(anisoten, w)
    mult_with_scalar(diff_r, 4, diff_r)
    mult_with_scalar(second, ls, diff_r)
    sub_vectors(third,w, v)
    mult_with_scalar(third, nu, third)
    ## SUB OR ADD these Vectors.
    sub_vectors(res, second, third)
    mult_with_scalar(res, 2, res)
    proj = scalar(res, w)
    for i in range(3):
        res[i] -= proj*w[i]


cdef double value_spherical(double[:] anisoten, double l, double[:] v, double[:] testv, double[:] three_vector, \
                                                                                               double nu) nogil \
        except *:
    cdef double last_term

    val = hota_4o3d_sym_s_form(anisoten, testv)

    last_term = pow(testv[0] - v[0],2) +  pow(testv[1] - v[1],2) +  pow(testv[2] - v[2],2)
    return pow(val, 2) - nu * last_term



##Algorithm 1: 21
cdef double calc_norm_spherical(double[:] ls, double[:,:] vs, double[:] res, double[:,:] pen, double nu, double[:]
three_vector) nogil except *:
    cdef double norm = 0
    norm += hota_4o3d_sym_norm(res)
    if norm != 0:
        pow(norm,2)
    for j in range(3):
        sub_vectors(three_vector, vs[:,j], pen[:, j])
        norm -= nu * scalar(three_vector, three_vector)
    return norm
