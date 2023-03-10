from scipy import optimize
from bonndit.utilc.hota cimport hota_4o3d_sym_s_form, hota_4o3d_sym_eval, hota_4o3d_sym_tsp, hota_4o3d_sym_v_form, hota_4o3d_sym_norm
import numpy as np


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

cdef double[:,:] real_candidates = np.zeros((_max_candidates_3d, 15))
get_real_candidates(real_candidates)

cdef get_real_candidates(t):
    cdef int i, j, k
    for i in range(0, _max_candidates_3d):
        hota_4o3d_sym_eval(t[i], 1, _candidates_3d_d[3 * i: 3 * (i + 1)])


cdef init_max_3d(double[:,:] tens, int n):
    """
    Find best initial direction on sphere for rank1 approximation of a fourth order tensor.
    """
    cdef double val, min_val=-1
    cdef int i,j,k
    for i in range(1, _max_candidates_3d):
        for j in range(i, _max_candidates_3d):
            for k in range(j, _max_candidates_3d):
                val = 0
                for l in range(n):
                    val += hota_4o3d_sym_norm(tens[l] - real_candidates[i] - real_candidates[l] - real_candidates[k])
                if min_val == -1 or val < min_val:
                    l,m,o = i,j,k
                    min_val = val
    return np.concatenate([_candidates_3d_d[3*(l-1): 3*l], _candidates_3d_d[3*(m-1): 3*m], _candidates_3d_d[3*(o-1): 3*o], np.array([1,1,1])])


def average(fodfs, n):
    x0 = init_max_3d(fodfs, n)
    res = optimize.m(opt_func, x0, jac=jacobian, args=(fodfs, n), )
    l = calc_lambda(res['x'][:9], fodfs, n)
    return l, res['x'][:9]



def opt_func(v, f, n):
    l =calc_lambda(v, f, n)
    summand0= 0
    for i in range(3):
        hota_4o3d_sym_eval(t[i], 1 ,v[3*i: 3*(i+1)])
    t[0] = np.sum(t, axis=0)
    for i in range(n):
        summand0 += hota_4o3d_sym_norm(f[i] - t[0])**2
    for i in range(3):
        summand0 += v[9+i]*(np.linalg.norm(v[3*i: 3*(i+1)]) - 1)
    return summand0

def calc_lambda(v, fodf, n):
    placeholder1 = np.zeros((15,))
    placeholder2 = np.zeros((15,))
    placeholder3 = np.zeros((15,))
    T = np.zeros((3,))
    tsp = np.zeros((3,))
    hota_4o3d_sym_eval(placeholder1, 1, v[:3])
    hota_4o3d_sym_eval(placeholder2, 1, v[3:6])
    hota_4o3d_sym_eval(placeholder3, 1, v[6:9])
    tsp[0] = hota_4o3d_sym_tsp(placeholder1, placeholder2)
    tsp[1] = hota_4o3d_sym_tsp(placeholder1, placeholder3)
    tsp[2] = hota_4o3d_sym_tsp(placeholder2, placeholder3)
    for i in range(3):
        summand1=0
        for j in range(n):
            summand1 += hota_4o3d_sym_s_form(fodf[j], v[3*i: 3*(i+1)])
        T[i] = summand1
    x = (   T[0]*tsp[2] ** 2*n ** 2 - 4*T[0] - tsp[2]*tsp[0]*T[2]*n ** 2 - tsp[2]*tsp[1]*T[1]*n ** 2 + 2*tsp[0]*T[1]*n + 2*tsp[1]*T[2]*n)/ (tsp[2] ** 2*n **2 - tsp[2]*tsp[0]*tsp[1]*n ** 3 + tsp[0] ** 2*n ** 2 + tsp[1] ** 2*n ** 2 - 4)
    y = (  -T[0]*tsp[2]*tsp[1]*n ** 2 + 2*T[0]*tsp[0]*n + 2*tsp[2]*T[2]*n - tsp[0]*tsp[1]*T[2]*n ** 2 + tsp[1] ** 2*T[1]*n ** 2 - 4*T[1])/ (tsp[2] ** 2*n ** 2 - tsp[2]*tsp[0]*tsp[1]*n ** 3 + tsp[0] ** 2*n ** 2 + tsp[1] ** 2*n ** 2 - 4)
    z = (  -T[0]*tsp[2]*tsp[0]*n ** 2 + 2*T[0]*tsp[1]*n + 2*tsp[2]*T[1]*n + tsp[0] ** 2*T[2]*n ** 2 - tsp[0]*tsp[1]*T[1]*n ** 2 - 4*T[2])/ (tsp[2] ** 2*n ** 2 - tsp[2]*tsp[0]*tsp[1]*n ** 3 + tsp[0] ** 2*n ** 2 + tsp[1] ** 2*n ** 2 - 4)
    return [x,y,z]


def jacobian(v, f, n):
    l =calc_lambda(v, f, n)
    t = np.zeros((45,))
    placeholder = np.zeros((3,))
    out = np.zeros((12,))
    for i in range(3):
        hota_4o3d_sym_eval(t[15*i: 15*(i+1)], 1, v[3*i: 3*(i+1)])
    for i in range(3):
        summand1 = 0
        summand2 = 0
        summand3 = 0
        for j in range(n):
            hota_4o3d_sym_v_form(placeholder, f[j], v[3 * i: 3 * (i + 1)])
            summand1 -= 2 * l[i] * placeholder
        for j in range(3):
            if i == j:
                continue
            hota_4o3d_sym_v_form(placeholder, t[15 * j:15 * (j + 1)], v[3 * i:3 * (i + 1)])
            summand2 += n * l[i] * l[j] * placeholder
        summand3 = v[9:]*v[3*i:3*(i+1)]
        out[3*i:3*(i+1)] = summand1 + summand2 + summand3
    for i in range(3):
        out[9+i] = np.linalg.norm(v[3*i:3*(i+1)])
    return out
