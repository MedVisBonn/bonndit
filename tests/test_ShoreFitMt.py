#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreMultiTissueResponse` class."""

import os

import nibabel as nib

import bonndit.shoredeconv as bs
from bonndit.michi import fields
from .constants import DECONVOLUTION_RESULTS_DIR as DRD
from .constants import SHORE_FIT_PRECOMPUTED, DWMRI_DUMMY_DATA, \
    ODF_RESULT_HPSD_DELTA, ODF_RESULT_NONNEG_DELTA, \
    ODF_RESULT_NO_CONSTRAINT_DELTA, ODF_RESULT_HPSD_RANK1, \
    ODF_RESULT_NONNEG_RANK1, ODF_RESULT_NO_CONSTRAINT_RANK1

data = nib.load(DWMRI_DUMMY_DATA)
ALLOWED_ERROR = 1e-10


def test_ShoreFit_deconvolution_hpsd_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_hpsd_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout -
                 fields.load_scalar(os.path.join(DRD, 'gm_hpsd_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((csfout -
                 fields.load_scalar(os.path.join(DRD, 'csf_hpsd_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_nonneg_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='nonneg', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_nonneg_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout - fields.load_scalar(
        os.path.join(DRD, 'gm_nonneg_rank1.nrrd'))[
        0]) < ALLOWED_ERROR).all() \
           and ((csfout - fields.load_scalar(
        os.path.join(DRD, 'csf_nonneg_rank1.nrrd'))[
        0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_no_constraint_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='none', kernel="rank1")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT_RANK1)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_none_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout -
                 fields.load_scalar(os.path.join(DRD, 'gm_none_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((csfout -
                 fields.load_scalar(os.path.join(DRD, 'csf_none_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_hpsd_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd', kernel="delta")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_hpsd_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout -
                 fields.load_scalar(os.path.join(DRD, 'gm_hpsd_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((csfout -
                 fields.load_scalar(os.path.join(DRD, 'csf_hpsd_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_nonneg_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='nonneg', kernel="delta")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_nonneg_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout -
                 fields.load_scalar(os.path.join(DRD, 'gm_nonneg_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((csfout - fields.load_scalar(
        os.path.join(DRD, 'csf_nonneg_delta.nrrd'))[0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_no_constraint_delta():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreMultiTissueResponse.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout = fit.fodf(data, pos='none', kernel="delta")
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT_DELTA)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_none_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout -
                 fields.load_scalar(os.path.join(DRD, 'gm_none_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((csfout -
                 fields.load_scalar(os.path.join(DRD, 'csf_none_delta.nrrd'))[
                     0]) < ALLOWED_ERROR).all()
