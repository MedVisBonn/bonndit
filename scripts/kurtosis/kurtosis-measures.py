#!/usr/bin/python
# -*- coding: utf-8 -*-
# Computation of axial, radial, and mean kurtosis based on DKI parameters
# Based on equations in Tabesh et al., MRM 2011
# Authors: Thomas Schultz, Michael Ankele

from __future__ import print_function, division

import argparse
import math
import os

import mpmath as mp
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description='Compute invariants of DKI model.')
parser.add_argument('infile', help='Path to DKI parameter file')
parser.add_argument('out', help='Path for the output files')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Flag for verbose output')

parser.add_argument('-k', '--kappa', action='store_true',
                    help='Additionally compute axial, radial and diamond kappa')
args = parser.parse_args()


def alpha(x):
    if x > 0:
        return math.atanh(math.sqrt(x)) / math.sqrt(x)
    else:
        return math.atan(math.sqrt(-x)) / math.sqrt(-x)


def H(a, c):
    if a == c:
        return 1.0 / 15.0
    return (a + 2 * c) ** 2 / (144 * c * c * (a - c) ** 2) * (c * (a + 2 * c) + a * (a - 4 * c) * alpha(1 - a / c))


def F1(a, b, c):
    if a == b:
        return 3 * H(c, a)
    if a == c:
        return 3 * H(b, a)
    return (a + b + c) ** 2 / (18 * (a - b) * (a - c)) * (
            math.sqrt(b * c) / a * float(mp.elliprf(a / b, a / c, 1)) + (3 * a ** 2 - a * b - a * c - b * c) / (
                3 * a * math.sqrt(b * c)) * float(mp.elliprd(a / b, a / c, 1)) - 1)


def F2(a, b, c):
    if b == c:
        return 6 * H(a, c)
    return (a + b + c) ** 2 / (3 * (b - c) ** 2) * (
            (b + c) / (math.sqrt(b * c)) * float(mp.elliprf(a / b, a / c, 1)) + (2 * a - b - c) / (
                3 * math.sqrt(b * c)) * float(mp.elliprd(a / b, a / c, 1)) - 2)


def G1(a, b, c):
    if b == c:
        return (a + 2 * b) ** 2 / (24 * b * b)
    return (a + b + c) ** 2 / (18 * b * (b - c) ** 2) * (2 * b + c * (c - 3 * b) / (math.sqrt(b * c)))


def G2(a, b, c):
    if b == c:
        return (a + 2 * b) ** 2 / (12 * b * b)
    return (a + b + c) ** 2 / (3 * (b - c) ** 2) * ((b + c) / (math.sqrt(b * c)) - 2)


def axialKurtosis(lambdas, W):
    return ((lambdas[0] + lambdas[1] + lambdas[2]) ** 2 / (9 * lambdas[2] ** 2)) * W[14]


def radialKurtosis(lambdas, W):
    return G1(lambdas[2], lambdas[1], lambdas[0]) * W[10] + G1(lambdas[2], lambdas[0], lambdas[1]) * W[0] + G2(
        lambdas[2], lambdas[1], lambdas[0]) * W[3]


def radial_kappa(lambda_mean, W):
    """

    :param lambda_mean:
    :param W:
    :return:
    """
    return lambda_mean ** 2 * (W[0] + W[10] + 3 * W[3]) / 3


def axial_kappa(lambda_mean, W):
    """

    :param lambda_mean:
    :param W:
    :return:
    """
    return lambda_mean ** 2 * W[14]


def diamond_kappa(lambda_mean, W):
    """

    :param lamda_mean:
    :param W:
    :return:
    """
    return 6 * lambda_mean ** 2 * (W[5] + W[12]) / 2


def meanKurtosis(lambdas, W):
    r = F1(lambdas[0], lambdas[1], lambdas[2]) * W[0]
    r += F1(lambdas[1], lambdas[2], lambdas[0]) * W[10]
    r += F1(lambdas[2], lambdas[1], lambdas[0]) * W[14]
    r += F2(lambdas[0], lambdas[1], lambdas[2]) * W[12]
    r += F2(lambdas[1], lambdas[2], lambdas[0]) * W[5]
    r += F2(lambdas[2], lambdas[1], lambdas[0]) * W[3]
    return r


ix4 = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 2, 2], [0, 1, 1, 1], [0, 1, 1, 2],
       [0, 1, 2, 2], [0, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2]]

invix4 = np.zeros((3, 3, 3, 3), dtype=np.int)
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                s = [i, j, k, l]
                s.sort()
                invix4[i, j, k, l] = ix4.index(s)


# L are the eigenvectors such that L[:,i] is ith normalized eigenvector
def rotT4Sym(W, L):
    # build and apply rotation matrix
    rotmat = np.zeros((15, 15))
    for idx in range(15):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    for ll in range(3):
                        rotmat[idx, invix4[ii, jj, kk, ll]] += L[ii, ix4[idx][0]] * L[jj, ix4[idx][1]] * L[
                            kk, ix4[idx][2]] * L[ll, ix4[idx][3]]
    return np.dot(rotmat, W)


# loading data
img = nib.load(args.infile)
data = img.get_data()
affine = img.get_affine()
NX, NY, NZ = data.shape[0:3]

da = np.zeros((NX, NY, NZ))
dm = np.zeros((NX, NY, NZ))
dr = np.zeros((NX, NY, NZ))
fa = np.zeros((NX, NY, NZ))
ka = np.zeros((NX, NY, NZ))
km = np.zeros((NX, NY, NZ))
kr = np.zeros((NX, NY, NZ))

if args.kappa:
    kappa_radial = np.zeros((NX, NY, NZ))
    kappa_axial = np.zeros((NX, NY, NZ))
    kappa_diamond = np.zeros((NX, NY, NZ))

for x in range(NX):
    if args.verbose:
        print('x=', x)
    for y in range(NY):
        for z in range(NZ):
            if data[x, y, z, 0] == 0:
                continue

            # we need DTI eigensystem
            T = np.array([[data[x, y, z, 1], data[x, y, z, 2], data[x, y, z, 3]],
                          [data[x, y, z, 2], data[x, y, z, 4], data[x, y, z, 5]],
                          [data[x, y, z, 3], data[x, y, z, 5], data[x, y, z, 6]]])
            # lambdas are in *ascending* order
            (lambdas, vs) = np.linalg.eigh(T)
            # clamp lambdas to avoid numerical trouble
            lambdas[lambdas < 1e-10] = 1e-10
            # DTI measures can be computed based on this
            da[x, y, z] = lambdas[2]
            dr[x, y, z] = 0.5 * (lambdas[0] + lambdas[1])
            dm[x, y, z] = np.mean(lambdas)
            fa[x, y, z] = math.sqrt(
                ((lambdas[0] - lambdas[1]) ** 2 + (lambdas[1] - lambdas[2]) ** 2 + (lambdas[2] - lambdas[0]) ** 2) / (
                        lambdas[0] ** 2 + lambdas[1] ** 2 + lambdas[2] ** 2) / 2)
            # rotate kurtosis tensor into eigensystem
            W = rotT4Sym(data[x, y, z, 7:], vs)

            ka[x, y, z] = axialKurtosis(lambdas, W)
            km[x, y, z] = meanKurtosis(lambdas, W)
            kr[x, y, z] = radialKurtosis(lambdas, W)

            # if args.kappa:
            #    # The given formulas need the the fiber to be oriented along the z-axis. Therefore we need to rotate the
            #    # kurtosis tensor
            #    W =
            #    kappa_radial[x, y, z] = radial_kappa(dm[x, y, z], W)
            #    kappa_axial[x, y, z] = axial_kappa(dm[x, y, z], W)
            #    kappa_diamond[x, y, z] = diamond_kappa(dm[x, y, z], W)

img = nib.Nifti1Image(da, affine)
nib.save(img, os.path.join(args.out, 'da.nii'))

img = nib.Nifti1Image(dr, affine)
nib.save(img, os.path.join(args.out, 'dr.nii'))

img = nib.Nifti1Image(dm, affine)
nib.save(img, os.path.join(args.out, 'dm.nii'))

img = nib.Nifti1Image(fa, affine)
nib.save(img, os.path.join(args.out, 'fa.nii'))

img = nib.Nifti1Image(ka, affine)
nib.save(img, os.path.join(args.out, 'ka.nii'))

img = nib.Nifti1Image(kr, affine)
nib.save(img, os.path.join(args.out, 'kr.nii'))

img = nib.Nifti1Image(km, affine)
nib.save(img, os.path.join(args.out, 'km.nii'))

if args.kappa:
    img = nib.Nifti1Image(kappa_radial, affine)
    nib.save(img, os.path.join(args.out, 'kappaRadial.nii'))

    img = nib.Nifti1Image(kappa_axial, affine)
    nib.save(img, os.path.join(args.out, 'kappaAxial.nii'))

    img = nib.Nifti1Image(kappa_diamond, affine)
    nib.save(img, os.path.join(args.out, 'kappaDiamond.nii'))
