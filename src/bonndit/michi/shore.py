from math import factorial

import numpy as np
from scipy.special import genlaguerre, gamma

from . import esh
from . import tensor
from .vector import cart_to_sphere

MAX_ORDER = esh.MAX_ORDER  # 12


def _get_size(a, r):
    if (a % 2) == 1 or (r % 2) == 1:
        return 0
    return sum([l * r - l * l + r // 2 + l // 2 * 3 + 1 for l in range(0, a + 1, 2)])


SIZES = [[_get_size(a, r) if r >= a else 0 for a in range(MAX_ORDER + 1)] for r in range(MAX_ORDER + 1)]
# SIZES = [[1, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [2, 0,  7, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [3, 0, 13, 0, 22, 0,   0, 0,   0, 0,   0, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [4, 0, 19, 0, 37, 0,  50, 0,   0, 0,   0, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [5, 0, 25, 0, 52, 0,  78, 0,  95, 0,   0, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [6, 0, 31, 0, 67, 0, 106, 0, 140, 0, 161, 0,   0],
#         [0, 0,  0, 0,  0, 0,   0, 0,   0, 0,   0, 0,   0],
#         [7, 0, 37, 0, 82, 0, 134, 0, 185, 0, 227, 0, 252]]

def _get_kernel_size(a, r):
    if (a % 2) == 1 or (r % 2) == 1:
        return 0

    return sum([(r - l) // 2 + 1 for l in range(0, a + 1, 2)])


KERNEL_SIZES = [[_get_kernel_size(a, r) if r >= a else 0 for a in range(MAX_ORDER + 1)] for r in range(MAX_ORDER + 1)]
# KERNEL_SIZES = [[1, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [2, 0,  3, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [3, 0,  5, 0,  6, 0,  0, 0,  0, 0,  0, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [4, 0,  7, 0,  9, 0, 10, 0,  0, 0,  0, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [5, 0,  9, 0, 12, 0, 14, 0, 15, 0,  0, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [6, 0, 11, 0, 15, 0, 18, 0, 20, 0, 21, 0,  0],
#                [0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
#                [7, 0, 13, 0, 18, 0, 22, 0, 25, 0, 27, 0, 28]]

def get_size(radial_order, angular_order):
    return SIZES[radial_order][angular_order]


#	F = radial_order / 2
#	return int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))

def order(coeff):
    size = len(coeff)
    for i in range(len(SIZES)):
        for j in range(len(SIZES)):
            if size == SIZES[i][j]:
                return i, j
    raise Exception("shore order can not be determined for size " + str(size))


get_order = order


def get_kernel_size(radial_order, angular_order):
    return KERNEL_SIZES[radial_order][angular_order]


# "kernel": only use z-rotational part
def compress(s):
    radial_order, angular_order = get_order(s)
    r = np.zeros(get_kernel_size(radial_order, angular_order))
    counter = 0
    ccounter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, (radial_order - l) // 2 + 1):
            r[ccounter] = s[counter + l]
            counter += 2 * l + 1
            ccounter += 1
    return r


def uncompress(s, radial_order, angular_order):
    r = np.zeros(get_size(radial_order, angular_order))
    counter = 0
    ccounter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, (radial_order - l) // 2 + 1):
            r[counter + l] = s[ccounter]
            counter += 2 * l + 1
            ccounter += 1
    return r


# M_(lnm)i
# M * shore_coeff = signal
def matrix(radial_order, angular_order, zeta, gtab, tau=1 / (4 * np.pi ** 2)):
    assert (radial_order >= angular_order)

    NGRADS = len(gtab.bvals)

    q = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    q[gtab.bvals < 40] = 0

    qgradients = q[:, None] * gtab.bvecs

    # r, theta, phi
    rtp = cart_to_sphere(qgradients)
    rsqrz = rtp[:, 0] ** 2 / zeta
    angles = rtp[:, 1:]

    sh = esh.matrix(angular_order, angles)

    size = get_size(radial_order, angular_order)
    M = np.zeros((NGRADS, size))

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, (radial_order + l) // 2 + 1):
            c = genlaguerre(n - l, l + 0.5)(rsqrz) * np.exp(- rsqrz / 2.0) * _kappa(zeta, n, l) * rsqrz ** (l / 2)
            for m in range(-l, l + 1):
                M[:, counter] = sh[:, esh.index(l, m)] * c
                counter += 1
    return M


def prepare_for_matrix(radial_order, angular_order, zeta, gtab, tau=1 / (4 * np.pi ** 2)):
    assert (radial_order >= angular_order)

    NGRADS = len(gtab.bvals)

    q = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    q[gtab.bvals < 40] = 0

    qgradients = q[:, None] * gtab.bvecs

    # r, theta, phi
    rtp = cart_to_sphere(qgradients)
    rsqrz = rtp[:, 0] ** 2 / zeta
    angles = rtp[:, 1:]

    sh = esh.matrix(angular_order, angles)

    size = get_size(radial_order, angular_order)
    # M = np.zeros((NGRADS, size))


    f_radial = np.zeros((NGRADS, angular_order + 1, radial_order + 1))
    for l in range(0, angular_order + 1, 2):
        for n in range(l, (radial_order + l) // 2 + 1):
            c = genlaguerre(n - l, l + 0.5)(rsqrz) * np.exp(- rsqrz / 2.0) * _kappa(zeta, n, l) * rsqrz ** (l / 2)
            f_radial[:, l, n] = c

    return (radial_order, angular_order, zeta, tau, size, f_radial)


def matrix_opt(opt, gtab):
    radial_order, angular_order, zeta, tau, size, f_radial = opt

    NGRADS = len(gtab.bvals)

    q = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    q[gtab.bvals < 40] = 0

    qgradients = q[:, None] * gtab.bvecs

    # r, theta, phi
    rtp = cart_to_sphere(qgradients)
    rsqrz = rtp[:, 0] ** 2 / zeta
    angles = rtp[:, 1:]

    sh = esh.matrix(angular_order, angles)

    M = np.zeros((NGRADS, size))

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, (radial_order + l) // 2 + 1):
            for m in range(-l, l + 1):
                M[:, counter] = sh[:, esh.index(l, m)] * f_radial[:, l, n]
                counter += 1
    return M


def _kappa(zeta, n, l):
    return np.sqrt((2 * factorial(n - l)) / (zeta ** 1.5 * gamma(n + 1.5)))


# deconvolution matrix into spherical harmonics
#    M_(lnm)(l'm') = kernel_ln * delta_ll' * delta_mm'
def matrix_kernel(kernel, order):
    size = get_size(order, order)
    M = np.zeros((size, esh.LENGTH[order]))

    counter = 0
    for l in range(0, order + 1, 2):
        for n in range(l, (order + l) // 2 + 1):
            for m in range(-l, l + 1):
                # argh, dipy.reconst.shore.shore_order(n,l,m)[1] gives wrong indices!!!
                #   -> using counter instead
                M[counter, esh.index(l, m)] = kernel[l, n]
                counter += 1
    return M


def signal_to_rank1_kernel(signal, order):
    # rank-1 sh
    T = tensor.power(np.array([0, 0, 1]), order)
    sh = esh.sym_to_esh(T)
    # print sh

    # Kernel_ln
    kernel = np.zeros((order + 1, order + 1))

    counter = 0
    for l in range(0, order + 1, 2):
        for n in range(l, (order + l) // 2 + 1):
            kernel[l, n] = signal[counter] / sh[esh.INDEX_OFFSET[l]]
            counter += 1

    # This is what happens
    #	kernel[0,0] = signal[0] / sh[0]
    #	kernel[0,1] = signal[1] / sh[0]
    #	kernel[0,2] = signal[2] / sh[0]
    #	if len(signal) > 3:
    #		kernel[2,2] = signal[3] / sh[3]
    #		kernel[2,3] = signal[4] / sh[3]
    #	if len(signal) > 5:
    #		kernel[4,4] = signal[5] / sh[10]
    return kernel

def signal_to_delta_kernel(signal, order):
    deltash = esh.eval_basis(order, 0, 0)
    # Kernel_ln
    kernel = np.zeros((order + 1, order + 1))
    counter = 0
    ccounter = 0
    for l in range(0, order + 1, 2):
        for n in range(int((order - l) / 2) + 1):
            kernel[l, l + n] = signal[counter] / deltash[ccounter]
            counter += 1
        ccounter += 2 * l + 3
    return kernel
