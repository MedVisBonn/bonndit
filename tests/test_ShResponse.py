#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreMultiTissueResponse` class."""

import os

import nibabel as nib
from dipy.io import read_bvals_bvecs

from bonndit.michi import fields
from bonndit.shdeconv import ShResponse
from .constants import CSD_RESULTS_DIR as DRD
from .constants import DATA_DIR, SH_RESPONSE_PRECOMPUTED, DWMRI_DUMMY_DATA, \
    CSD_ODF_HPSD_DELTA, CSD_ODF_NONNEG_DELTA, \
    CSD_ODF_NO_CONSTRAINT_DELTA, CSD_ODF_HPSD_RANK1, \
    CSD_ODF_NONNEG_RANK1, CSD_ODF_NO_CONSTRAINT_RANK1

data = nib.load(DWMRI_DUMMY_DATA)
bvals, bvecs = read_bvals_bvecs(os.path.join(DATA_DIR, "bvals"),
                                os.path.join(DATA_DIR, "bvecs"))

bval_indices = bvals <= 700
new_data = data.get_data()[:, :, :, bval_indices]
data = nib.Nifti1Image(new_data, data.affine)
ALLOWED_ERROR = 1e-10


def test_ShoreFit_deconvolution_hpsd_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='hpsd', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_HPSD_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_hpsd_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_nonneg_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='nonneg', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_NONNEG_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_nonneg_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_no_constraint_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='none', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_NO_CONSTRAINT_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_none_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_hpsd_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='hpsd', kernel="delta")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_HPSD_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_hpsd_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_nonneg_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='nonneg', kernel="delta")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_NONNEG_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_nonneg_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_no_constraint_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = ShResponse.load(SH_RESPONSE_PRECOMPUTED)
    out, wmout = fit.fodf(data, pos='none', kernel="delta")
    tensors, mask, meta = fields.load_tensor(CSD_ODF_NO_CONSTRAINT_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_none_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all()
