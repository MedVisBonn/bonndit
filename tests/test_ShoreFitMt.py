#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreFitMt` class."""

import os

import nibabel as nib

import bonndit.shoremt as bs
from bonndit.michi import fields
from .constants import DECONVOLUTION_RESULTS_DIR as DRD
from .constants import SHORE_FIT_PRECOMPUTED, DWMRI_DUMMY_DATA, \
    ODF_RESULT_HPSD, ODF_RESULT_NONNEG, ODF_RESULT_NO_CONSTRAINT

data = nib.load(DWMRI_DUMMY_DATA)
ALLOWED_ERROR = 1e-10


def test_ShoreFit_deconvolution_hpsd_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("rank1")
    out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD)

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
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("rank1")
    out, wmout, gmout, csfout = fit.fodf(data, pos='nonneg')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG)

    assert ((out - tensors) < ALLOWED_ERROR).all() \
           and ((wmout -
                 fields.load_scalar(os.path.join(DRD, 'wm_nonneg_rank1.nrrd'))[
                     0]) < ALLOWED_ERROR).all() \
           and ((gmout - fields.load_scalar(
        os.path.join(DRD, 'gm_rank1nonneg_rank1.nrrd'))[
        0]) < ALLOWED_ERROR).all() \
           and ((csfout - fields.load_scalar(
        os.path.join(DRD, 'csf_rank1nonneg_rank1.nrrd'))[
        0]) < ALLOWED_ERROR).all()


def test_ShoreFit_deconvolution_no_constraint_rank1():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("rank1")
    out, wmout, gmout, csfout = fit.fodf(data, pos='none')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT)

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
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("delta")
    out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD)

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
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("delta")
    out, wmout, gmout, csfout = fit.fodf(data, pos='nonneg')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG)

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
    fit = bs.ShoreFitMt.load(SHORE_FIT_PRECOMPUTED)
    fit.set_kernel("delta")
    out, wmout, gmout, csfout = fit.fodf(data, pos='none')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT)

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
