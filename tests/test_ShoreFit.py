import numpy as np
import nibabel as nib
import bonndit.shore as bs
from .constants import SHORE_FIT_FILE, \
    DWMRI_DUMMY_DATA, ODF_RESULT_HPSD, ODF_RESULT_NO_CONSTRAINT, ODF_RESULT_HPSD_WORLDC

"""
def test_ShoreModel_deconvolution_hpsd():
    ''' Here we test the deconvolution with the hpsd constraint.

    '''
    fit = bs.ShoreFit.load(SHORE_FIT_FILE)

    data = nib.load(DWMRI_DUMMY_DATA)
    out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd')


    assert (out == nib.load(ODF_RESULT_HPSD).get_data()).all()

def test_ShoreModel_deconvolution_nonneg():
    ''' Here we test the deconvolution with the nonneg constraint.

    '''
    fit = bs.ShoreFit.load(SHORE_FIT_FILE)

    data = nib.load(DWMRI_DUMMY_DATA)
    out, wmout, gmout, csfout = fit.fodf(data, pos='nonneg')


    assert (out == nib.load(ODF_RESULT_NONNEG).get_data()).all()

def test_ShoreModel_deconvolution_no_constraint():
    ''' Here we test the deconvolution without any constraint.

    '''
    fit = bs.ShoreFit.load(SHORE_FIT_FILE)

    data = nib.load(DWMRI_DUMMY_DATA)
    out, wmout, gmout, csfout = fit.fodf(data, pos='none')


    assert (out == nib.load(ODF_RESULT_NO_CONSTRAINT).get_data()).all()
"""

def test_ShoreModel_deconvolution_hpsd_worldc():
    ''' Here we test the deconvolution with the hpsd constraint calculated in world coordinates

    '''
    fit = bs.ShoreFit.load(SHORE_FIT_FILE)
    import os
    #data = nib.load(DWMRI_DUMMY_DATA)
    #out, wmout, gmout, csfout = fit.fodf(data, pos='hpsd')
    out, wmout, gmout, csfout = fit.fodf(DWMRI_DUMMY_DATA, pos='hpsd')
    reference_output = nib.load(ODF_RESULT_HPSD_WORLDC).get_data().astype('float32')
    #assert (out == reference_output).all()
    assert (((out - reference_output) / reference_output) < 1e-7).all()
