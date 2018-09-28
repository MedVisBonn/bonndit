#!/usr/bin/python
# -*- coding: utf-8 -*-
# Fitting kurtosis using quadratic cone programming
# Similar to Tabesh et al., MRM 2011
# but uses cone programming to guarantee a minimum diffusivity
# Author: Thomas Schultz

from __future__ import print_function, division

import argparse
import os
from os import sys

import cvxopt
import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs

parser = argparse.ArgumentParser(description='Fit DKI model to input data.')
parser.add_argument('infile', help='Path to the DWI data')
parser.add_argument('bval', help='Path to the b-values')
parser.add_argument('bvec', help='Path to the normalized gradient directions (b-vectors)')
parser.add_argument('outdir', help='Path of the output')
parser.add_argument('-m', '--mask', help='Path to the brain mask')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Flag for verbose output')
parser.add_argument('-b', '--bvaleps', default=0, type=int,
                    help='The maximum b-value which should be treated as zero. (Default: 0)')
args = parser.parse_args()

# loading data
bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
bval_eps = args.bvaleps
# Test before loading large data
if len(bvals[bvals <= bval_eps]) == 0:
    print('There is no measurement for the specified minimum b-value of {}.'.format(bval_eps))
    sys.exit(1)
img = nib.load(args.infile)
data = img.get_data()
data = data.astype(float)
affine = img.affine
NX, NY, NZ = data.shape[0:3]

# re-scale b values to more natural units
bvals = bvals / 1000.0
bval_eps = bval_eps / 1000.0

# we have to bring b vectors into world coordinate system
# we will use the 3x3 linear transformation part of the affine matrix for this
linear = affine[0:3, 0:3]
# according to FSL documentation, we first have to flip the sign of the
# x coordinate if the matrix determinant is positive
if np.linalg.det(linear) > 0:
    bvecs[:, 0] = -bvecs[:, 0]
# now, apply the linear mapping to bvecs and re-normalize
bvecs = np.dot(bvecs, np.transpose(linear))
bvecnorm = np.linalg.norm(bvecs, axis=1)
bvecnorm[bvecnorm == 0] = 1.0  # avoid division by zero
bvecs = bvecs / bvecnorm[:, None]

# mask
mask = np.ones((NX, NY, NZ))
if args.mask != None:
    maskimg = nib.load(args.mask)
    mask = maskimg.get_data()

# Build matrix A (maps DKI params to log signal ratio)
# isolate non-zero gradient directions
grads = bvecs[bvals > bval_eps]
dwibvals = bvals[bvals > bval_eps]
bmax = np.max(dwibvals)
nk = len(dwibvals)
A = np.zeros((nk, 21))
for i in range(nk):
    # note: the order at this point deviates from Tabesh et al.
    # so as to agree with teem conventions
    A[i, 0] = -dwibvals[i] * grads[i, 0] * grads[i, 0]
    A[i, 1] = -dwibvals[i] * 2 * grads[i, 0] * grads[i, 1]
    A[i, 2] = -dwibvals[i] * 2 * grads[i, 0] * grads[i, 2]
    A[i, 3] = -dwibvals[i] * grads[i, 1] * grads[i, 1]
    A[i, 4] = -dwibvals[i] * 2 * grads[i, 1] * grads[i, 2]
    A[i, 5] = -dwibvals[i] * grads[i, 2] * grads[i, 2]
    A[i, 6] = dwibvals[i] ** 2 / 6.0 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 0]
    A[i, 7] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 1]
    A[i, 8] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 2]
    A[i, 9] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 1]
    A[i, 10] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 2]
    A[i, 11] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 0] * grads[i, 0] * grads[i, 2] * grads[i, 2]
    A[i, 12] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    A[i, 13] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    A[i, 14] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    A[i, 15] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    A[i, 16] = dwibvals[i] ** 2 / 6.0 * grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    A[i, 17] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    A[i, 18] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 1] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    A[i, 19] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 1] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    A[i, 20] = dwibvals[i] ** 2 / 6.0 * grads[i, 2] * grads[i, 2] * grads[i, 2] * grads[i, 2]

if np.linalg.cond(A) > 1e6:
    print('Refusing to fit DKI with condition number ', np.linalg.cond(A))
    print('Are you trying to estimate kurtosis from single-shell data?')
    sys.exit(1)
elif args.verbose:
    print('Condition number of A: ', np.linalg.cond(A))

# Build constraint matrix C for diffusivities and kurtosis
C = np.zeros((nk * 2 + 9, 21))
for i in range(nk):
    # orthant constraints go first: min kurtosis
    C[i, 6] = -grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 0]
    C[i, 7] = -4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 1]
    C[i, 8] = -4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 2]
    C[i, 9] = -6 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 1]
    C[i, 10] = -12 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 2]
    C[i, 11] = -6 * grads[i, 0] * grads[i, 0] * grads[i, 2] * grads[i, 2]
    C[i, 12] = -4 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    C[i, 13] = -12 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    C[i, 14] = -12 * grads[i, 0] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    C[i, 15] = -4 * grads[i, 0] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    C[i, 16] = -grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    C[i, 17] = -4 * grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    C[i, 18] = -6 * grads[i, 1] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    C[i, 19] = -4 * grads[i, 1] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    C[i, 20] = -grads[i, 2] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    # max kurtosis constraints as in Tabesh et al.
    C[nk + i, 0] = -3.0 / bmax * grads[i, 0] * grads[i, 0]
    C[nk + i, 1] = -3.0 / bmax * 2 * grads[i, 0] * grads[i, 1]
    C[nk + i, 2] = -3.0 / bmax * 2 * grads[i, 0] * grads[i, 2]
    C[nk + i, 3] = -3.0 / bmax * grads[i, 1] * grads[i, 1]
    C[nk + i, 4] = -3.0 / bmax * 2 * grads[i, 1] * grads[i, 2]
    C[nk + i, 5] = -3.0 / bmax * grads[i, 2] * grads[i, 2]
    C[nk + i, 6] = grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 0]
    C[nk + i, 7] = 4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 1]
    C[nk + i, 8] = 4 * grads[i, 0] * grads[i, 0] * grads[i, 0] * grads[i, 2]
    C[nk + i, 9] = 6 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 1]
    C[nk + i, 10] = 12 * grads[i, 0] * grads[i, 0] * grads[i, 1] * grads[i, 2]
    C[nk + i, 11] = 6 * grads[i, 0] * grads[i, 0] * grads[i, 2] * grads[i, 2]
    C[nk + i, 12] = 4 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    C[nk + i, 13] = 12 * grads[i, 0] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    C[nk + i, 14] = 12 * grads[i, 0] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    C[nk + i, 15] = 4 * grads[i, 0] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    C[nk + i, 16] = grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 1]
    C[nk + i, 17] = 4 * grads[i, 1] * grads[i, 1] * grads[i, 1] * grads[i, 2]
    C[nk + i, 18] = 6 * grads[i, 1] * grads[i, 1] * grads[i, 2] * grads[i, 2]
    C[nk + i, 19] = 4 * grads[i, 1] * grads[i, 2] * grads[i, 2] * grads[i, 2]
    C[nk + i, 20] = grads[i, 2] * grads[i, 2] * grads[i, 2] * grads[i, 2]
# min diffusivity - now a proper psd constraint, independent of directions
# just need to give it the negative diffusion tensor in column major order
C[2 * nk, 0] = -1.0
C[2 * nk + 1, 1] = -1.0
C[2 * nk + 2, 2] = -1.0
C[2 * nk + 3, 1] = -1.0
C[2 * nk + 4, 3] = -1.0
C[2 * nk + 5, 4] = -1.0
C[2 * nk + 6, 2] = -1.0
C[2 * nk + 7, 4] = -1.0
C[2 * nk + 8, 5] = -1.0

d = np.zeros((nk * 2 + 9, 1))
# impose minimum diffusivity
d[2 * nk] = -0.1
d[2 * nk + 4] = -0.1
d[2 * nk + 8] = -0.1
dims = {'l': 2 * nk, 'q': [], 's': [3]}

# set up QP problem from normal equations
cvxopt.solvers.options['show_progress'] = False
P = cvxopt.matrix(np.ascontiguousarray(np.dot(A.T, A)))
G = cvxopt.matrix(np.ascontiguousarray(C))
h = cvxopt.matrix(np.ascontiguousarray(d))

if args.verbose:
    print('Optimizing...')

out = np.zeros((NX, NY, NZ, 22))
for x in range(NX):
    if args.verbose:
        print('x=', x)
    for y in range(NY):
        for z in range(NZ):
            if mask[x, y, z] == 0:
                continue

            S0 = np.mean(data[x, y, z, bvals <= bval_eps])
            if S0 <= 0:
                continue
            S = data[x, y, z, bvals > bval_eps]
            S[S <= 1e-10] = 1e-10  # clamp negative values
            S = np.log(S / S0)
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(A.T, S)))

            sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
            if sol['status'] != 'optimal':
                print('WARNING: First-pass optimization unsuccessful.')
            c = np.array(sol['x'])[:, 0]
            out[x, y, z, 0] = 1
            out[x, y, z, 1:7] = c[:6]
            # divide out d-bar-square to get kurtosis tensor
            Dbar = (c[0] + c[3] + c[5]) / 3.0
            out[x, y, z, 7:] = c[6:] / Dbar ** 2

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
img = nib.Nifti1Image(out, affine)
nib.save(img, os.path.join(args.outdir, 'kurtosis_fit.nii'))

# Needed for comparison with bonndit results in bonndittests
np.savez(os.path.join(args.outdir, 'kurtosis_fit.npz'), data=out, mask=mask,
         bvals=bvals, bvecs=bvecs)
