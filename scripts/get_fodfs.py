#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
================================================================================
Compute fiber orientation distribution functions for diffusion weighted MRI data
================================================================================
"""

import argparse
import os
import sys
import numpy as np
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from bonndit.shore import ShoreModel, ShoreFit
import nibabel as nib


# To Do: Add option to first compute the diffustion tensors which are needed to estimate the response functions.

# To Do: Add option to automatically build masks for csf gm wm.

# To Do: Handle nrrd as well as nii input

# To Do: Enable specification of output format

def main():
    parser = argparse.ArgumentParser(
        description='This script computes fiber orientation distribution functions (fODFs) as described in "Versatile, Robust and Efficient Tractography With Constrained Higher Order Tensor fODFs" by Ankele et al. (2017)')

    parser.add_argument('-i', '--indir', required=True, help='Path to the folder containing all required input files.')
    parser.add_argument('-o', '--outdir', default=None, help='Folder in which the output will be saved.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Set to show a progress bar')
    parser.add_argument('-r', '--order', default=4, help='Order of the shore basis')
    parser.add_argument('-z', '--zeta', default=700, help='Radial scaling factor')
    parser.add_argument('-t', '--tau', default=1 / (4 * np.pi ** 2), help='q-scaling')
    parser.add_argument('-f', '--fawm', default=0.7, help='The WM FA threshold')
    parser.add_argument('-R', '--responseonly', action='store_true', help='Calculate and save only the response functions.')

    # Print help if not called correctly
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    order = args.order
    zeta = args.zeta
    tau = args.tau
    fawm = args.fawm
    verbose = args.verbose
    indir = args.indir
    if not args.outdir:
        outdir = args.indir
    else:
        outdir = args.outdir

    # Load fractional anisotropy
    dti_fa = nib.load(os.path.join(indir, "dti_FA.nii.gz"))

    # Load DTI mask
    dti_mask = nib.load(os.path.join(indir, "mask.nii.gz"))

    # Load and adjust tissue segmentation masks
    csf_mask = nib.load(os.path.join(indir, "fast_pve_0.nii.gz"))
    gm_mask = nib.load(os.path.join(indir, "fast_pve_1.nii.gz"))
    wm_mask = nib.load(os.path.join(indir, "fast_pve_2.nii.gz"))

    # Load DTI first eigenvectors
    dti_vecs = nib.load(os.path.join(indir, "dti_V1.nii.gz"))

    # Load the data
    data = nib.load(os.path.join(indir, "data.nii.gz"))

    bvals, bvecs = read_bvals_bvecs(os.path.join(indir, "bvals"),
                                    os.path.join(indir, "bvecs"))
    gtab = gradient_table(bvals, bvecs)

    # Check if response functions already exist.
    if os.path.exists(os.path.join(outdir, 'response.npz')):
        fit = ShoreFit.old_load(os.path.join(outdir, 'response.npz'))
        if verbose:
            print('Loaded existing response functions.')
    else:
        model = ShoreModel(gtab, order, zeta, tau)
        fit = model.fit(data, wm_mask, gm_mask, csf_mask, dti_mask,
                    dti_fa, dti_vecs, fawm, verbose=verbose)
        fit.old_save(outdir)

    # Check if the response only flag is set.
    if not args.responseonly:
        out, wmout, gmout, csfout = fit.fodf(data, verbose=verbose)





if __name__ == "__main__":
    main()
