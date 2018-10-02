#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

from bonndit import DkiModel, DkiFit
from bonndit.io import fsl_gtab_to_worldspace
from .constants import DATA_DIR, DKI_TENSOR

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DATA_DIR, "dti_FA.nii.gz"))

# Load DTI mask
dti_mask = nib.load(os.path.join(DATA_DIR, "mask.nii.gz"))

# Load DW-MRI data
data = nib.load(os.path.join(DATA_DIR, "data.nii.gz"))

# Load bvals and bvecs
bvals, bvecs = read_bvals_bvecs(os.path.join(DATA_DIR, "bvals"),
                                os.path.join(DATA_DIR, "bvecs"))
gtab = gradient_table(bvals, bvecs)

# Rotation to worldspace and sign flip according to fsl documentation
gtab = fsl_gtab_to_worldspace(gtab, data.affine)

reference_fit = DkiFit.load(DKI_TENSOR)

ALLOWED_ERROR = 1e-10


def test_DkiModel_tensor():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """

    model = DkiModel(gtab, constraint=True)
    fit = model.fit(data.get_data(), mask=dti_mask.get_data())
    assert ((reference_fit.coeffs - fit.coeffs) /
            reference_fit.coeffs < ALLOWED_ERROR).all()
