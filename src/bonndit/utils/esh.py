#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import sin, cos, pi, sqrt

import numpy as np
import numpy.linalg as la

from . import tensor as T
from .mmath import binom, multinom, cos_n, sin_n

MAX_ORDER = 12

LENGTH = [1, 0, 6, 0, 15, 0, 28, 0, 45, 0, 66, 0, 91]
KERNEL_LENGTH = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7]


def get_size(order):
    return LENGTH[order]


def get_kernel_size(order):
    return KERNEL_LENGTH[order]


def get_order(e):
    for i, l in enumerate(LENGTH):
        if l == len(e):
            return i
    raise Exception("invalid sh size: " + str(len(e)))


def get_order_from_kernel_length(l):
    return l * 2 - 2


# index(l,m) = INDEX_OFFSET[l] + m
INDEX_OFFSET = [0, 0, 3, 0, 10, 0, 21, 0, 36, 0, 55, 0, 78]


def ggg(a):
    r = sqrt(pi)
    while a > 1:
        a -= 1
        r *= a
    return r


# integral over the sphere of   x^i y^j z^k
def int_sphere(i, j, k):
    return 2 * ggg((i + 1.0) / 2) * ggg((j + 1.0) / 2) * ggg((k + 1.0) / 2) / ggg((i + j + k + 3) / 2)


_sym2esh = [None for i in range(MAX_ORDER + 1)]
_esh2sym = [None for i in range(MAX_ORDER + 1)]


def zero(order):
    return np.zeros(LENGTH[order])


# We use the following convention:
# theta = polar angle from positive z
# phi   = azimuth from positive x


LEGENDRE_A = [
    [2],
    [4, -2, 4],
    [16, -8, 8, -8, 16],
    [32, -16, 64, -32, 32, -32, 64],
    [256, -64 / 3.0, 128 / 3.0, -64, 128 / 3.0, -64 / 3.0, 128, -64 / 3.0, 256 / 3.0],
    [512, -256, 512 / 3.0, -256 / 3.0, 256 / 3.0, -256 / 3.0, 1024 / 3.0, -512 / 3.0, 512, -512, 1024],
    [2048 / 5.0, -1024 / 5.0, 1024 / 5.0, -1024 / 5.0, 4096 / 15.0, -1024 / 15.0, 2048 / 5.0, -1024 / 5.0, 2048 / 5.0,
     -1024 / 5.0, 2048 / 5.0, -1024 / 5.0, 4096 / 5.0]]
LEGENDRE_B = [
    [1],
    [5, 15, 15],
    [9, 45 * 2, 45, 315 * 2, 315],
    [13, 273, 2730, 2730, 91 * 9, 2002 * 9, 6006],
    [17, 17, 2 * 595, 19635, 1309, 17017, 2 * 7293, 12155, 12155],
    [21, 1155, 385, 2 * 5005, 5005, 2002, 2 * 5005, 2 * 85085, 255255, 2 * 4849845, 2 * 969969],
    [1, 2 * 39, 3003, 2 * 1001, 2 * 1001, 17017, 2 * 2431, 138567, 138567, 323323, 2 * 88179, 2028117, 2 * 676039]]
LEGENDRE_C = []
for i in range(len(LEGENDRE_A)):
    LEGENDRE_C += [[sqrt(b / pi) / a for a, b in zip(LEGENDRE_A[i], LEGENDRE_B[i])]]

LEGENDRE_D = [
    [[1]],
    [[2, -1], [1], [1]],
    [[8, -24, 3], [4, -3], [6, -1], [1], [1]],
    [[16, -120, 90, -5], [8, -20, 5], [16, -16, 1], [8, -3], [10, -1], [1], [1]],
    [[128, -1792, 3360, -1120, 35], [64, -336, 280, -35], [32, -80, 30, -1], [16, -20, 3], [40, -24, 1], [4, -1],
     [14, -1], [1], [1]],
    [[256, -5760, 20160, -16800, 3150, -63], [128, -1152, 2016, -840, 63], [384, -1792, 1680, -336, 7],
     [64, -168, 84, -7], [112, -168, 42, -1], [168, -140, 15], [224, -96, 3], [16, -3], [18, -1], [1], [1]],
    [[1024, -33792, 190080, -295680, 138600, -16632, 231], [512, -7040, 21120, -18480, 4620, -231],
     [256, -1920, 3360, -1680, 210, -3], [640, -2880, 3024, -840, 45], [1280, -3584, 2240, -320, 5],
     [128, -224, 80, -5], [1344, -1440, 270, -5], [96, -60, 5], [120, -40, 1], [20, -3], [22, -1], [1], [1]],
]


# symmetric for +-m
def legendre(order, theta):
    res = zero(order)

    res[0] = LEGENDRE_C[0][0]
    if order < 2:
        return res

    st = sin(theta)
    ct = cos(theta)
    st2 = st * st
    ct2 = ct * ct
    res[5] = res[1] = LEGENDRE_C[1][2] * st2
    res[4] = res[2] = LEGENDRE_C[1][1] * ct * st
    res[3] = LEGENDRE_C[1][0] * (3.0 * ct2 - 1.0)
    if order < 4:
        return res

    st4 = st2 * st2
    ct4 = ct2 * ct2
    res[14] = res[6] = LEGENDRE_C[2][4] * st4
    res[13] = res[7] = LEGENDRE_C[2][3] * ct * st * st2
    res[12] = res[8] = LEGENDRE_C[2][2] * (7 * ct2 - 1) * st2
    res[11] = res[9] = LEGENDRE_C[2][1] * (7 * ct2 * ct - 3 * ct) * st
    res[10] = LEGENDRE_C[2][0] * (35.0 * ct4 - 30.0 * ct2 + 3.0)
    if order < 6:
        return res

    st6 = st4 * st2
    ct6 = ct4 * ct2
    res[27] = res[15] = LEGENDRE_C[3][6] * st6
    res[26] = res[16] = LEGENDRE_C[3][5] * st4 * st * ct
    res[25] = res[17] = LEGENDRE_C[3][4] * st4 * (11 * ct2 - 1.0)
    res[24] = res[18] = LEGENDRE_C[3][3] * st2 * st * (11 * ct2 * ct - 3 * ct)
    res[23] = res[19] = LEGENDRE_C[3][2] * st2 * (33 * ct4 - 18 * ct2 + 1.0)
    res[22] = res[20] = LEGENDRE_C[3][1] * st * (33 * ct4 * ct - 30.0 * ct2 * ct + 5 * ct)
    res[21] = LEGENDRE_C[3][0] * (231 * ct6 - 315 * ct4 + 105 * ct2 - 5.0)
    if order < 8:
        return res

    st8 = st4 * st4
    ct8 = ct4 * ct4
    res[44] = res[28] = LEGENDRE_C[4][8] * st8
    res[43] = res[29] = LEGENDRE_C[4][7] * st6 * st * ct
    res[42] = res[30] = LEGENDRE_C[4][6] * st6 * (15 * ct2 - 1)
    res[41] = res[31] = LEGENDRE_C[4][5] * st4 * st * ct * (5 * ct2 - 1)
    res[40] = res[32] = LEGENDRE_C[4][4] * st4 * (65 * ct4 - 26 * ct2 + 1)
    res[39] = res[33] = LEGENDRE_C[4][3] * st2 * st * ct * (39 * ct4 - 26 * ct2 + 3)
    res[38] = res[34] = LEGENDRE_C[4][2] * st2 * (143 * ct6 - 143 * ct4 + 33 * ct2 - 1)
    res[37] = res[35] = LEGENDRE_C[4][1] * st * ct * (715 * ct6 - 1001 * ct4 + 385 * ct2 - 35)
    res[36] = LEGENDRE_C[4][0] * (6435 * ct8 - 12012 * ct6 + 6930 * ct4 - 1260 * ct2 + 35)
    if order < 10:
        return res

    st10 = st8 * st2
    ct10 = ct8 * ct2
    res[65] = res[45] = LEGENDRE_C[5][10] * st10
    res[64] = res[46] = LEGENDRE_C[5][9] * st8 * st * ct
    res[63] = res[47] = LEGENDRE_C[5][8] * st8 * (19 * ct2 - 1)
    res[62] = res[48] = LEGENDRE_C[5][7] * st6 * st * ct * (19 * ct2 - 3)
    res[61] = res[49] = LEGENDRE_C[5][6] * st6 * (323 * ct4 - 102 * ct2 + 3)
    res[60] = res[50] = LEGENDRE_C[5][5] * st4 * st * ct * (323 * ct4 - 170 * ct2 + 15)
    res[59] = res[51] = LEGENDRE_C[5][4] * st4 * (323 * ct6 - 255 * ct4 + 45 * ct2 - 1)
    res[58] = res[52] = LEGENDRE_C[5][3] * st2 * st * ct * (323 * ct6 - 357 * ct4 + 105 * ct2 - 7)
    res[57] = res[53] = LEGENDRE_C[5][2] * st2 * (4199 * ct8 - 6188 * ct6 + 2730 * ct4 - 364 * ct2 + 7)
    res[56] = res[54] = LEGENDRE_C[5][1] * st * ct * (4199 * ct8 - 7956 * ct6 + 4914 * ct4 - 1092 * ct2 + 63)
    res[55] = LEGENDRE_C[5][0] * (46189 * ct10 - 109395 * ct8 + 90090 * ct6 - 30030 * ct4 + 3465 * ct2 - 63)
    if order < 12:
        return res

    st12 = st6 * st6
    ct12 = ct6 * ct6
    res[90] = res[66] = LEGENDRE_C[6][12] * st12
    res[89] = res[67] = LEGENDRE_C[6][11] * st10 * st * ct
    res[88] = res[68] = LEGENDRE_C[6][10] * st10 * (23 * ct2 - 1)
    res[87] = res[69] = LEGENDRE_C[6][9] * st8 * st * ct * (23 * ct2 - 3)
    res[86] = res[70] = LEGENDRE_C[6][8] * st8 * (161 * ct4 - 42 * ct2 + 1)
    res[85] = res[71] = LEGENDRE_C[6][7] * st6 * st * ct * (161 * ct4 - 70 * ct2 + 5)
    res[84] = res[72] = LEGENDRE_C[6][6] * st6 * (3059 * ct6 - 1995 * ct4 + 285 * ct2 - 5)
    res[83] = res[73] = LEGENDRE_C[6][5] * st4 * st * ct * (437 * ct6 - 399 * ct4 + 95 * ct2 - 5)
    res[82] = res[74] = LEGENDRE_C[6][4] * st4 * (7429 * ct8 - 9044 * ct6 + 3230 * ct4 - 340 * ct2 + 5)
    res[81] = res[75] = LEGENDRE_C[6][3] * st2 * st * ct * (7429 * ct8 - 11628 * ct6 + 5814 * ct4 - 1020 * ct2 + 45)
    res[80] = res[76] = LEGENDRE_C[6][2] * st2 * (7429 * ct10 - 14535 * ct8 + 9690 * ct6 - 2550 * ct4 + 225 * ct2 - 3)
    res[79] = res[77] = LEGENDRE_C[6][1] * st * ct * (
        52003 * ct10 - 124355 * ct8 + 106590 * ct6 - 39270 * ct4 + 5775 * ct2 - 231)
    res[78] = LEGENDRE_C[6][0] * (
        676039 * ct12 - 1939938 * ct10 + 2078505 * ct8 - 1021020 * ct6 + 225225 * ct4 - 18018 * ct2 + 231)
    return res


# just for checking the coefficients
def legendre_slow(order, theta):
    res = zero(order)

    c = cos(theta)
    s = sin(theta)
    cc = []
    ss = []
    for l in range(0, order + 1, 2):
        cc += [c ** l, c ** (l + 1)]
        ss += [s ** l, s ** (l + 1)]
        index0 = INDEX_OFFSET[l]
        for m in range(0, l + 1):
            x = 0
            for j, d in enumerate(LEGENDRE_D[l // 2][m]):
                x += d * cc[l - m - 2 * j] * ss[m + 2 * j]
            res[index0 + m] = res[index0 - m] = LEGENDRE_C[l // 2][m] * x

    return res


def eval_basis(order, theta, phi):
    # Evaluate spherical harmonics for given order for point on sphere
    # defined by theta and phi
    res = legendre(order, theta)

    cos_m_phi = [cos(m * phi) for m in range(order + 1)]
    sin_m_phi = [sin(m * phi) for m in range(order + 1)]

    for l in range(0, order + 1, 2):
        for m in range(1, l + 1):
            res[INDEX_OFFSET[l] - m] *= cos_m_phi[m]
        for m in range(1, l + 1):
            res[INDEX_OFFSET[l] + m] *= sin_m_phi[m]
    return res


def eval(coeffs, theta, phi):
    order = get_order(coeffs)
    basis = eval_basis(order, theta, phi)
    return np.dot(basis, coeffs)


# angles: [N,2] (theta, phi)
# returns [N,sh]
def matrix(order, angles):
    N = angles.shape[0]
    sh = np.zeros((N, LENGTH[order]))
    for i in range(N):
        sh[i] = eval_basis(order, angles[i, 0], angles[i, 1])
    return sh


def eval_sp(A, B):
    return np.dot(A, B)


#

# check: OK
def convolve(e, kernel):
    order = get_order(e)
    out = zero(order)
    idx = 0
    for o in range(order / 2 + 1):
        while (idx < LENGTH[o * 2]):
            out[idx] = e[idx] * kernel[o]
            idx += 1
    return out


# check: OK
def deconvolve(e, kernel):
    order = get_order(e)
    out = zero(order)
    idx = 0
    for o in range(order / 2 + 1):
        while (idx < LENGTH[o * 2]):
            out[idx] = e[idx] / kernel[o]
            idx += 1
    return out


# Make a deconvolution kernel that turns a given signal (rotationally
# symmetric in "compressed" form, i.e., one coefficient per SH order)
# into a rank-1 term of given order.


# check: OK
def make_kernel_rank1(signal):
    l = len(signal)
    order = get_order_from_kernel_length(l)
    kernel = np.zeros(l)
    # Need to determine SH coefficients of a z-aligned rank-1 tensor
    v = [0.0, 0.0, 1.0]
    rank1 = T.power(v, order)
    rank1sh = sym_to_esh(rank1)
    # print rank1sh
    for s in signal:
        if s == 0:
            raise Exception("signal has 0")
    for i in range(0, order + 1, 2):
        kernel[i // 2] = signal[i // 2] / rank1sh[INDEX_OFFSET[i]]
    return kernel


# Make a deconvolution kernel that turns a given signal (rotationally
# symmetric in "compressed" form, i.e., one coefficient per SH order)
# into a truncated delta peak of given order.

# check: OK
def make_kernel_delta(signal):
    l = len(signal)
    order = get_order_from_kernel_length(l)
    kernel = np.zeros(l)
    # we need a truncated delta peak of given order
    for i in range(l):
        if signal[i] == 0:
            raise Exception("signal has 0")
    deltash = eval_basis(order, 0, 0)
    for i in range(0, order + 1, 2):
        kernel[i // 2] = signal[i // 2] / deltash[INDEX_OFFSET[i]]
    return kernel


def index(l, m):
    return INDEX_OFFSET[l] + m


def compress_kernel(e):
    order = get_order(e)
    kernel = np.zeros(KERNEL_LENGTH[order])
    for i in range(0, order + 1, 2):
        kernel[i // 2] = e[INDEX_OFFSET[i]]
    return kernel


###############################################################
#  sh/tensor conversion matrices


def x2y2n(n):
    r = np.zeros(n * 2 + 1)
    for i in range(n + 1):
        r[i * 2] = binom(n, i)
    return r


def csmult(csa, csb):
    r = np.zeros(len(csa) + len(csb) - 1)
    for i, a in enumerate(csa):
        for j, b in enumerate(csb):
            r[i + j] += a * b
    return r


def x2y2z2n(n):
    r = np.zeros((n * 2 + 1, n * 2 + 1, n * 2 + 1))
    # (x^2+y^2+z^2)^n = sum_ multinom(
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            r[i * 2, j * 2, k * 2] = multinom(i, j, k)
    return r


def xyzmult(csa, csb):
    m = csa.shape[0] - 1
    n = csb.shape[0] - 1
    r = np.zeros((m + n + 1, m + n + 1, m + n + 1))
    for i, a in enumerate(csa):
        for j in range(n + 1):
            for k in range(n + 1):
                for l in range(n + 1):
                    # x^(m-i) y^i * x^j y^k z^l
                    # = x^(i+j) y^(m-i+k) z^l
                    r[m - i + j, i + k, l] += a * csb[j, k, l]
    return r


def xyzaddz(xyz, dz):
    n = xyz.shape[0] - 1
    r = np.zeros((n + dz + 1, n + dz + 1, n + dz + 1))
    r[:n + 1, :n + 1, dz:n + 1 + dz] = xyz
    return r


def create_esh_to_sym(order):
    M = np.zeros((LENGTH[order], LENGTH[order]))

    ish = 0
    for l in range(0, order + 1, 2):
        for m in range(-l, l + 1):
            mm = abs(m)
            # print("------l{} m{}".format(l,m))
            for j, d in enumerate(LEGENDRE_D[l // 2][mm]):
                diz = l - mm - 2 * j
                ir = mm
                c = d * LEGENDRE_C[l // 2][mm]
                jxy = j
                jxyz = (order - l) // 2
                #	print("xy^{} z^{}  (x²+y²)^{}  (x²+y²+z²)^{}".format(ir,diz,jxy,jxyz))

                if m <= 0:
                    csi = cos_n(mm)
                elif m > 0:
                    csi = sin_n(mm)
                if jxy > 0:
                    csi = csmult(csi, x2y2n(jxy))
                # print(csi)
                #				print(x2y2z2n(jxyz))
                xyz = xyzmult(csi, x2y2z2n(jxyz))
                #				print(xyz)

                for i, a in np.ndenumerate(xyz):
                    if a == 0:
                        continue
                    ii = [i[0], i[1], i[2] + diz]
                    #					print("{}  {}  x^{} y^{} z^{}".format(c,a,*ii))
                    it = T.CINDEX[order].index(ii)
                    #					print("=>",it,ish)
                    M[it, ish] += c * a / T.MULTIPLIER[order][it]
            ish += 1
    return M


def sym_to_esh_matrix(order):
    if _sym2esh[order] is None:
        _sym2esh[order] = la.inv(esh_to_sym_matrix(order))
    return _sym2esh[order]


def esh_to_sym_matrix(order):
    if _esh2sym[order] is None:
        _esh2sym[order] = create_esh_to_sym(order)
    return _esh2sym[order]


def sym_to_esh(ten):
    order = get_order(ten)
    return np.dot(sym_to_esh_matrix(order), ten)


def esh_to_sym(e):
    order = get_order(e)
    return np.dot(esh_to_sym_matrix(order), e)


def matrix_power(R, order):
    A = esh_to_sym_matrix(order)
    B = T.matrix_power(R, order)
    C = sym_to_esh_matrix(order)
    return np.dot(C, np.dot(B, A))
