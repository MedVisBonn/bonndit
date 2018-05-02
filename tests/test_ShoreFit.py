#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.shore.ShoreFit` class."""

from bonndit.michi import fields
import bonndit.shore as bs
from .constants import SHORE_FIT_PRECOMPUTED, DWMRI_DUMMY_DATA, \
    ODF_RESULT_HPSD, ODF_RESULT_NONNEG, ODF_RESULT_NO_CONSTRAINT
from .constants import DECONVOLUTION_RESULTS_DIR as DRD
import os
import nibabel as nib


data = nib.load(DWMRI_DUMMY_DATA)

def test_ShoreFit_deconvolution_hpsd():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreFit.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout, mask = fit.fodf(data, pos='hpsd')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD)

    print((out - tensors) / tensors)
    assert (out == tensors).all() \
    and (wmout == fields.load_scalar(os.path.join(DRD, 'wmhpsd.nrrd'))[0]).all() \
    and (gmout == fields.load_scalar(os.path.join(DRD, 'gmhpsd.nrrd'))[0]).all() \
    and (csfout == fields.load_scalar(os.path.join(DRD, 'csfhpsd.nrrd'))[0]).all() \


def test_ShoreFit_deconvolution_nonneg():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreFit.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout, mask = fit.fodf(data, pos='nonneg')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG)

    print((out - tensors) / tensors)
    assert (out == tensors).all() \
    and (wmout == fields.load_scalar(os.path.join(DRD, 'wmnonneg.nrrd'))[0]).all() \
    and (gmout == fields.load_scalar(os.path.join(DRD, 'gmnonneg.nrrd'))[0]).all() \
    and (csfout == fields.load_scalar(os.path.join(DRD, 'csfnonneg.nrrd'))[0]).all()


def test_ShoreFit_deconvolution_no_constraint():
    """ Here we test the deconvolution with the hpsd constraint.
    The result calculated with bonndit are compared to results from the old code
    """
    fit = bs.ShoreFit.load(SHORE_FIT_PRECOMPUTED)
    out, wmout, gmout, csfout, mask = fit.fodf(data, pos='none')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT)

    print((out-tensors)/tensors)
    assert (out == tensors).all() \
    and (wmout == fields.load_scalar(os.path.join(DRD, 'wmnone.nrrd'))[0]).all() \
    and (gmout == fields.load_scalar(os.path.join(DRD, 'gmnone.nrrd'))[0]).all() \
    and (csfout == fields.load_scalar(os.path.join(DRD, 'csfnone.nrrd'))[0]).all()


