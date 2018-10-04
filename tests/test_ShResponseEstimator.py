#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.mtShoreModel` class."""

import os

import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

import bonndit as bd
from bonndit import ShResponseEstimator, ShResponse
from bonndit.io import fsl_vectors_to_worldspace, fsl_gtab_to_worldspace
from .constants import DATA_DIR, SH_RESPONSE

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DATA_DIR, "dti_FA.nii.gz"))

# Load DTI mask
dti_mask = nib.load(os.path.join(DATA_DIR, "mask.nii.gz"))

# Load and adjust tissue segmentation masks
wm_mask = nib.load(os.path.join(DATA_DIR, "fast_pve_2.nii.gz"))

wm_mask = bd.shoredeconv.fa_guided_mask(wm_mask, dti_fa, dti_mask,
                                        fa_lower_thresh=0.7)

# Load DTI first eigenvector
dti_vecs = nib.load(os.path.join(DATA_DIR, "dti_V1.nii.gz"))

# Load DW-MRI data
data = nib.load(os.path.join(DATA_DIR, "data.nii.gz"))

# Load bvals and bvecs
bvals, bvecs = read_bvals_bvecs(os.path.join(DATA_DIR, "bvals"),
                                os.path.join(DATA_DIR, "bvecs"))

# We want to work with a single shell
bval_indices = bvals <= 700
bvals = bvals[bval_indices]
bvecs = bvecs[bval_indices, :]
gtab = gradient_table(bvals, bvecs)
new_data = data.get_data()[:, :, :, bval_indices]
data = nib.Nifti1Image(new_data, data.affine)

# Rotation to worldspace and sign flip according to fsl documentation
gtab = fsl_gtab_to_worldspace(gtab, data.affine)
dti_vecs = fsl_vectors_to_worldspace(dti_vecs)

model = ShResponseEstimator(gtab)
fit = model.fit(data, dti_vecs, wm_mask)

reference_fit = ShResponse.load(SH_RESPONSE)

ALLOWED_ERROR = 1e-7
def test_ShoreModel_signal_wm():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """
    assert ((
                reference_fit.wm_response - fit.wm_response) / reference_fit.wm_response < ALLOWED_ERROR).all()
