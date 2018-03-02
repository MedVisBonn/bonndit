from dipy.core.gradients import GradientTable, gradient_table
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm
from dipy.reconst.shore import ShoreModel, shore_order
from dipy.io import read_bvals_bvecs
from scipy.special import genlaguerre, gamma
from math import factorial
import numpy as np
import numpy.linalg as la

import tensor
import tensor.esh as esh
from helper.progress import Progress

SIZES = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [2, 0, 7, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [3, 0, 13, 0, 22, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 0, 19, 0, 37, 0, 50, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [5, 0, 25, 0, 52, 0, 78, 0, 95]]

KERNEL_SIZES = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 5, 0, 6, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 7, 0, 9, 0, 10, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 0, 9, 0, 12, 0, 14, 0, 15]]


def get_size(radial_order, angular_order):
    return SIZES[radial_order][angular_order]


#   F = radial_order / 2
#   return int(np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3)))

def get_order(size):
    for i in range(len(SIZES)):
        for j in range(len(SIZES)):
            if size == SIZES[i][j]:
                return i, j
    raise Exception("shore order can not be determined for size " + str(size))


def get_kernel_size(radial_order, angular_order):
    return KERNEL_SIZES[radial_order][angular_order]


# "kernel": only use z-rotational part
def compress(s):
    radial_order, angular_order = get_order(len(s))
    r = np.zeros(get_kernel_size(radial_order, angular_order))
    counter = 0
    ccounter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(int((radial_order - l) / 2) + 1):
            r[ccounter] = s[counter + l]
            counter += 2 * l + 1
            ccounter += 1
    return r


def uncompress(s, radial_order, angular_order):
    r = np.zeros(get_size(radial_order, angular_order))
    counter = 0
    ccounter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(int((radial_order - l) / 2) + 1):
            r[counter + l] = s[ccounter]
            counter += 2 * l + 1
            ccounter += 1
    return r


# M_(lnm)i
#   copied from dipy:
def matrix(radial_order, angular_order, zeta, gtab, tau=1 / (4 * np.pi ** 2)):
    #   print "-------------- matrix"
    # print radial_order, angular_order
    assert (radial_order >= angular_order)

    zero_mask = (gtab.bvals == 0.0)

    qvals = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    qvals[gtab.b0s_mask] = 0
    bvecs = gtab.bvecs

    qgradients = qvals[:, None] * bvecs

    NGRADS = qgradients.shape[0]
    qgradients[zero_mask, :] = [0, 0, 0.00001]

    r, theta, phi = cart2sphere(qgradients[:, 0], qgradients[:, 1],
                                qgradients[:, 2])
    #   print r
    #   print theta
    #   print phi
    #   print "---"
    theta[np.isnan(theta)] = 0
    size = get_size(radial_order, angular_order)
    M = np.zeros((NGRADS, size))

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                M[:, counter] = real_sph_harm(m, l, theta, phi) * \
                                genlaguerre(n - l, l + 0.5)(r ** 2 / zeta) * \
                                np.exp(- r ** 2 / (2.0 * zeta)) * \
                                _kappa(zeta, n, l) * \
                                (r ** 2 / zeta) ** (l / 2)
                counter += 1
    return M


def prepare_for_matrix(radial_order, angular_order, zeta, gtab, tau=1 / (4 * np.pi ** 2)):
    assert (radial_order >= angular_order)

    qvals = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    qvals[gtab.b0s_mask] = 0
    bvecs = gtab.bvecs

    qgradients = qvals[:, None] * bvecs

    r, theta, phi = cart2sphere(qgradients[:, 0], qgradients[:, 1],
                                qgradients[:, 2])

    size = get_size(radial_order, angular_order)

    counter = 0
    f_radial = np.zeros((r.shape[0], size))
    for l in range(0, angular_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                f_radial[:, counter] = genlaguerre(n - l, l + 0.5)(r ** 2 / zeta) * \
                                       np.exp(- r ** 2 / (2.0 * zeta)) * \
                                       _kappa(zeta, n, l) * \
                                       (r ** 2 / zeta) ** (l / 2)
                counter += 1

    return (radial_order, angular_order, zeta, tau, size, f_radial)


def matrix_opt(opt, gtab):
    radial_order, angular_order, zeta, tau, size, f_radial = opt

    qvals = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    qvals[gtab.b0s_mask] = 0
    bvecs = gtab.bvecs

    qgradients = qvals[:, None] * bvecs

    r, theta, phi = cart2sphere(qgradients[:, 0], qgradients[:, 1],
                                qgradients[:, 2])
    theta[np.isnan(theta)] = 0

    M = np.zeros((r.shape[0], size))

    sh = np.zeros((r.shape[0], angular_order * 2 + 1, angular_order * 2 + 1))
    for l in range(0, angular_order + 1, 2):
        for m in range(-l, l + 1):
            sh[:, l, m + l] = real_sph_harm(m, l, theta, phi)

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                M[:, counter] = sh[:, l, m + l] * f_radial[:, counter]
                counter += 1
    return M


def _kappa(zeta, n, l):
    return np.sqrt((2 * factorial(n - l)) / (zeta ** 1.5 * gamma(n + 1.5)))


# deconvolution matrix into spherical harmonics
#    M_(lnm)(l'm') = kernel_ln * delta_ll' * delta_mm'
def matrix_kernel(kernel, radial_order, angular_order):
    assert (radial_order >= angular_order)
    size = get_size(radial_order, angular_order)
    M = np.zeros((size, esh.LENGTH[angular_order]))

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                # argh, shore_order(n,l,m)[1] gives wrong indices!!!
                #   -> using counter instead
                M[counter, esh.index(l, m)] = kernel[l, n]
                counter += 1
    return M


def fit(data, radial_order, angular_order, gtab, zeta, tau=1 / (4 * np.pi ** 2), mask=None):
    M = matrix(radial_order, angular_order, zeta, gtab, tau)
    s = data.shape[:-1]
    # print data.shape, s, data.shape[-1]
    data = data.reshape((-1, data.shape[-1]))
    N = data.shape[0]
    p = None
    if N > 20:
        p = Progress(N)

    if mask is not None:
        mask = mask.flat

    coeff = np.zeros((N, get_size(radial_order, angular_order)))
    for i in range(N):
        if mask is not None:
            if mask[i] < 0.5:
                continue
        if p is not None:
            p.set(i)
        r = la.lstsq(M, data[i, :])
        coeff[i, :] = r[0]
    # print np.sqrt(r[1]) / la.norm(data[i,:])
    return coeff.reshape(s + (-1,))


def fit_dipy(data, radial_order, angular_order, gtab, zeta, tau=1 / (4 * np.pi ** 2), mask=None):
    assert (radial_order == angular_order)
    asm = ShoreModel(gtab, radial_order, zeta, tau=tau)
    asmfit = asm.fit(data, mask)
    return asmfit._shore_coef


def signal_to_kernel(signal, radial_order, angular_order):
    # rank-1 sh
    T = tensor.power(np.array([0, 0, 1]), angular_order)
    sh = esh.sym_to_esh(T)
    # print sh

    # Kernel_ln
    kernel = np.zeros((9, 9))

    counter = 0
    for l in range(0, angular_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            kernel[l, n] = signal[counter] / sh[esh.INDEX_OFFSET[l]]
            counter += 1

        #   kernel[0,0] = signal[0] / sh[0]
        #   kernel[0,1] = signal[1] / sh[0]
        #   kernel[0,2] = signal[2] / sh[0]
        #   if len(signal) > 3:
        #       kernel[2,2] = signal[3] / sh[3]
        #       kernel[2,3] = signal[4] / sh[3]
        #   if len(signal) > 5:
        #       kernel[4,4] = signal[5] / sh[10]
    # print kernel
    return kernel
