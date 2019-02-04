#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.mtShoreModel` class."""

import os

import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

from bonndit import ShoreMultiTissueResponseEstimator, ShoreMultiTissueResponse
from bonndit.io import fsl_gtab_to_worldspace, fsl_vectors_to_worldspace
from bonndit.shoredeconv import fa_guided_mask
from .constants import DATA_DIR, SHORE_FIT_TEST

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DATA_DIR, "dti_FA.nii.gz"))

# Load DTI mask
dti_mask = nib.load(os.path.join(DATA_DIR, "mask.nii.gz"))

# Load and adjust tissue segmentation masks
csf_mask = nib.load(os.path.join(DATA_DIR, "fast_pve_0.nii.gz"))
gm_mask = nib.load(os.path.join(DATA_DIR, "fast_pve_1.nii.gz"))
wm_mask = nib.load(os.path.join(DATA_DIR, "fast_pve_2.nii.gz"))

wm_mask = fa_guided_mask(wm_mask, dti_fa, dti_mask, tissue_threshold=0.95,
                         fa_lower_thresh=0.7)
gm_mask = fa_guided_mask(gm_mask, dti_fa, dti_mask, tissue_threshold=0.95,
                         fa_upper_thresh=0.2)
csf_mask = fa_guided_mask(csf_mask, dti_fa, dti_mask, tissue_threshold=0.95,
                          fa_upper_thresh=0.2)

# Load DTI first eigenvector
dti_vecs = nib.load(os.path.join(DATA_DIR, "dti_V1.nii.gz"))

# Load DW-MRI data
data = nib.load(os.path.join(DATA_DIR, "data.nii.gz"))

# Load bvals and bvecs
bvals, bvecs = read_bvals_bvecs(os.path.join(DATA_DIR, "bvals"),
                                os.path.join(DATA_DIR, "bvecs"))
gtab = gradient_table(bvals, bvecs)


# Rotation to worldspace and sign flip according to fsl documentation
gtab = fsl_gtab_to_worldspace(gtab, data.affine)
dti_vecs = fsl_vectors_to_worldspace(dti_vecs)

model = ShoreMultiTissueResponseEstimator(gtab)
fit = model.fit(data, dti_vecs, wm_mask, gm_mask, csf_mask, cpus=5)

reference_fit = ShoreMultiTissueResponse.load(SHORE_FIT_TEST)

ALLOWED_ERROR = 1e-7
def test_ShoreModel_signal_csf():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """
    assert ((reference_fit.signal_csf - fit.signal_csf) / reference_fit.signal_csf < ALLOWED_ERROR).all()


def test_ShoreModel_signal_gm():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """
    assert ((reference_fit.signal_gm - fit.signal_gm) / reference_fit.signal_gm < ALLOWED_ERROR).all()

def test_ShoreModel_signal_wm():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """
    assert ((reference_fit.signal_wm - fit.signal_wm) / reference_fit.signal_wm < ALLOWED_ERROR).all()
