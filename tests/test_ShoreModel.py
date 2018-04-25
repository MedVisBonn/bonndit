#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreModel` class."""

import os
from bonndit import ShoreModel, ShoreFit
from dipy.core.gradients import gradient_table
import nibabel as nib
from dipy.io import read_bvals_bvecs

from .constants import DECONVOLUTION_DIR, SHORE_FIT_TEST

"""
I would prefer to load the data and do all the transformation explicitly using a 
class written for this purpose. - Explicit is better than implicit.
For now we use the old loading procedure which does all the transformation 
implicitly. 
"""

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_FA.nii.gz")).get_data()

# Load DTI mask
dti_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "mask.nii.gz")).get_data()

# Load and adjust tissue segmentation masks
csf_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_0.nii.gz")).get_data()
gm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_1.nii.gz")).get_data()
wm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_2.nii.gz")).get_data()

# Load DTI first eigenvector
dti_vecs = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_V1.nii.gz")).get_data()

# Load DW-MRI data
data = nib.load(os.path.join(DECONVOLUTION_DIR, "data.nii.gz")).get_data()

# Load bvals and bvecs
bvals, bvecs = read_bvals_bvecs(os.path.join(DECONVOLUTION_DIR, "bvals"),
                                os.path.join(DECONVOLUTION_DIR, "bvecs"))
gtab = gradient_table(bvals, bvecs)


model = ShoreModel(gtab)
fit = model.fit(data, wm_mask, gm_mask, csf_mask, dti_mask,
                dti_fa, dti_vecs)

reference_fit = ShoreFit.old_load(SHORE_FIT_TEST)


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
