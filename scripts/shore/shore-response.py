#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
======================================================
Find multi-tissue response functions for deconvolution
======================================================
"""
from __future__ import print_function, division

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import get_sphere
from dipy.data import get_data, dsi_voxels
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs, GradientTable
from dipy.core.geometry import vec2vec_rotmat
from dipy.viz import fvtk
from dipy.data import get_sphere

import dwmri.shore as shore
from helper.progress import Progress

import argparse


def create_sphere_func(sphere, shore_coeff, b):
    gtab = GradientTable(sphere.vertices * b)
    order = shore.get_order(len(shore_coeff))
    M = shore.matrix(order[0], order[1], shore_zeta, gtab, shore_tau)
    x = np.dot(M, shore_coeff)
    n = len(x)
    return x.reshape((1, 1, 1, n))


def get_shore(gtab, data, radial_order, angular_order, shore_zeta, shore_tau):
    return shore.fit(data, radial_order, angular_order, gtab, shore_zeta, shore_tau)


def reorient_gtab(u, gtab, bvals):
    N0 = len(np.ones(len(bvals))[gtab.b0s_mask])

    # rotate gradients to align 1st eigenvector to (0,0,1)
    # R = np.eye(3)
    R = vec2vec_rotmat(np.array([0, 0, 1]), u)
    return reorient_bvecs(gtab, np.tile(R.reshape((1, 3, 3)), (len(bvals) - N0, 1, 1)))


def render_shore(shore_coeff, b, filename):
    sphere = get_sphere('symmetric724')

    # odf = asmfit.odf(sphere)
    odf = create_sphere_func(sphere, shore_coeff, b)
    # print(odf)

    r = fvtk.ren()
    sfu = fvtk.sphere_funcs(odf, sphere, colormap='jet')
    sfu.RotateX(-90)
    fvtk.add(r, sfu)
    fvtk.record(r, n_frames=1, out_path=filename, size=(600, 600))


def accumulate_shore(shore_coeff, mask, radial_order, angular_order):
    N = shore_coeff.shape[0]
    shore_accum = np.zeros(shore.get_size(radial_order, angular_order))
    accum_count = 0
    nan_count = 0
    for i in range(N):
        if mask[i] == 0:
            continue
        for s in shore_coeff[i, :]:
            if np.isnan(s):
                nan += 1
                break
        else:
            shore_accum += shore_coeff[i, :]
            accum_count += 1

    print('{} voxel'.format(accum_count))
    if nan_count > 0:
        print('{} nans'.format(nan_count))
    if accum_count == 0:
        return shore_accum
    return shore_accum / accum_count


def get_response_reorient(data, gtab, mask, vecs, radial_order, angular_order, bvals, shore_zeta, shore_tau):
    N = data.shape[0]
    shore_coeff = np.zeros((N, shore.get_size(radial_order, angular_order)))

    count = 0

    p = Progress(N)
    for i in range(N):
        p.set(i)
        if mask[i] == 0:
            continue
        gtab2 = reorient_gtab(vecs[i], gtab, bvals)
        shore_coeff[i, :] = get_shore(gtab2, data[i, :], radial_order, angular_order, shore_zeta, shore_tau)
        # if count == 17:
        #    render_shore(shore_coeff[i,:], 2000, "first.png")
        count += 1

    return accumulate_shore(shore_coeff, mask, radial_order, angular_order)


def get_response(data, gtab, mask, radial_order, angular_order, shore_zeta, shore_tau):
    shore_coeff = shore.fit(data, radial_order, angular_order, gtab, shore_zeta, shore_tau, mask)
    return accumulate_shore(shore_coeff, mask, radial_order, angular_order)


def main():
    parser = argparse.ArgumentParser(
        description='This is a script to find the multi-tissue response functions for deconvolution.')
    parser.add_argument('-i', '--indir', required=True, help='Path to the folder containing all required input files.')
    parser.add_argument('-o', '--outdir', required=True, help='Folder in which the output will be saved.')
    parser.add_argument('-r', '--order', default=4, help='Order of the shore basis')
    parser.add_argument('-z', '--zeta', default=700, help='Radial scaling factor')
    parser.add_argument('-t', '--tau', default=1 / (4 * np.pi ** 2), help='q-scaling')
# What means WM?
    parser.add_argument('-f', '--fawm', default=0.7, help='The WM FA threshold')

    args = parser.parse_args()
    radial_order = args.order
    angular_order = args.order
    shore_zeta = args.zeta
    shore_tau = args.tau

    print('Radial Order: {}    Angular Order: {}'.format(radial_order, angular_order))
    """
    Read the data (for now, only accepts nifti as input).
    """

    indir = args.indir
    if indir[-1] != '/':
        indir += '/'
    outdir = args.outdir
    if outdir[-1] != '/':
        outdir += '/'

    # Create outfolder if not existing
    if not os.path.exists(directory):
        os.makedirs(directory)

    # input: tissue segmentation masks
    fcsf = indir + "fast_pve_0.nii.gz"
    fgm = indir + "fast_pve_1.nii.gz"
    fwm = indir + "fast_pve_2.nii.gz"

    # input: fa map
    ffa = indir + "dti_FA.nii.gz"

    # output mask filenames
    ofcsfmask = outdir + "csf_mask.nii.gz"
    ofgmmask = outdir + "gm_mask.nii.gz"
    ofwmmask = outdir + "wm_mask.nii.gz"

    print(ffa)
    faimg = nib.load(ffa)
    fa = faimg.get_data()
    affine = faimg.get_affine()

    # use DTI mask if available
    try:
        maskimg = nib.load(indir + "mask.nii.gz")
        mask = maskimg.get_data()
        print("Using provided DTI mask.")
    except:
        NX, NY, NZ = fa.shape
        mask = np.ones((NX, NY, NZ))
        print("No DTI mask found.")

    # masks

    # CSF
    csfimg = nib.load(fcsf)
    csf = csfimg.get_data()

    mask_csf = np.logical_and(mask, np.logical_and(csf > 0.95, fa < 0.2)).astype('int')

    img = nib.Nifti1Image(mask_csf, affine)
    nib.save(img, ofcsfmask)

    # GM
    gmimg = nib.load(fgm)
    gm = gmimg.get_data()

    mask_gm = np.logical_and(mask, np.logical_and(gm > 0.95, fa < 0.2)).astype('int')

    img = nib.Nifti1Image(mask_gm, affine)
    nib.save(img, ofgmmask)

    # WM
    wmimg = nib.load(fwm)
    wm = wmimg.get_data()

    mask_wm = np.logical_and(mask, np.logical_and(wm > 0.95, fa > float(args.fawm))).astype('int')

    img = nib.Nifti1Image(mask_wm, affine)
    nib.save(img, ofwmmask)

    vecsimg = nib.load(indir + "dti_V1.nii.gz")
    vecs = vecsimg.get_data()

    bvals, bvecs = read_bvals_bvecs(indir + "bvals", indir + "bvecs")

    # all that should matter for this script is that bvecs and evecs are
    # in the same coordinate system; hopefully, this will be the case

    gtab = gradient_table(bvals, bvecs)

    try:
        fdata = indir + "data.nii"
        dataimg = nib.load(fdata)
    except:
        fdata = indir + "data.nii.gz"
        dataimg = nib.load(fdata)

    print("loading data....")
    data = dataimg.get_data()
    print("reshaping...")

    NX, NY, NZ = data.shape[0:3]
    N = NX * NY * NZ
    data = data.reshape((N, -1))
    vecs = vecs.reshape((N, 3))
    mask_csf = mask_csf.flatten()
    mask_gm = mask_gm.flatten()
    mask_wm = mask_wm.flatten()

    print("csf response:")
    shore_coeff = get_response(data, gtab, mask_csf, radial_order, angular_order, shore_zeta, shore_tau)
    # print(shore_coeff)
    signal_csf = shore.compress(shore_coeff)
    print(signal_csf[:shore.get_kernel_size(radial_order, 0)])

    print("grey matter response:")
    shore_coeff = get_response(data, gtab, mask_gm, radial_order, angular_order, shore_zeta, shore_tau)
    # print(shore_coeff)
    signal_gm = shore.compress(shore_coeff)
    print(signal_gm[:shore.get_kernel_size(radial_order, 0)])

    print("white matter response:")
    shore_coeff = get_response_reorient(data, gtab, mask_wm, vecs, radial_order, angular_order, bvals,  shore_zeta,
                                        shore_tau)
    # print(shore_coeff)
    signal_wm = shore.compress(shore_coeff)
    print(signal_wm)

    np.savez(outdir + 'response.npz', csf=signal_csf, gm=signal_gm, wm=signal_wm,
             zeta=shore_zeta, tau=shore_tau)

    # render_shore(shore_coeff, 1000, _dir + "x.png")
    # shore_iso = shore.uncompress(cshore, radial_order, angular_order)
    # render_shore(shore_iso, 1000, _dir + "x_iso.png")


if __name__ == "__main__":
    main()
