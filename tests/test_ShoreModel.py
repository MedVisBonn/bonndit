#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreModel` class."""

import os
from bonndit import ShoreModel, ShoreFit
import bonndit.shore as bdshore
from bonndit.io import fsl_flip_signs_vec, fsl_to_worldspace
from dipy.core.gradients import gradient_table
import nibabel as nib
from dipy.io import read_bvals_bvecs

from .constants import DECONVOLUTION_DIR, SHORE_FIT_TEST

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_FA.nii.gz"))

# Load DTI mask
dti_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "mask.nii.gz"))

# Load and adjust tissue segmentation masks
csf_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_0.nii.gz"))
gm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_1.nii.gz"))
wm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_2.nii.gz"))

wm_mask, gm_mask, csf_mask = bdshore.dti_masks(wm_mask, gm_mask, csf_mask,
                                                   dti_fa, dti_mask, fawm=0.7)

# Load DTI first eigenvector
dti_vecs = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_V1.nii.gz"))

# Load DW-MRI data
data = nib.load(os.path.join(DECONVOLUTION_DIR, "data.nii.gz"))

# Load bvals and bvecs
bvals, bvecs = read_bvals_bvecs(os.path.join(DECONVOLUTION_DIR, "bvals"),
                                os.path.join(DECONVOLUTION_DIR, "bvecs"))
gtab = gradient_table(bvals, bvecs)


# Rotation to worldspace and sign flip according to fsl documentation
gtab = fsl_to_worldspace(data.affine, gtab)
dti_vecs = fsl_flip_signs_vec(dti_vecs)

model = ShoreModel(gtab)
fit = model.fit(data, dti_vecs, wm_mask, gm_mask, csf_mask)

reference_fit = ShoreFit.load(SHORE_FIT_TEST)



def test_ShoreModel_signal_csf():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """

    assert ((reference_fit.signal_csf - fit.signal_csf) < 1e-9).all()


def test_ShoreModel_signal_gm():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """

    assert ((reference_fit.signal_gm - fit.signal_gm) < 1e-9).all()

def test_ShoreModel_signal_wm():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """

    assert ((reference_fit.signal_wm - fit.signal_wm) < 1e-9).all()
