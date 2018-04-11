from bonndit.michi import fields
import bonndit.shore as bs
from .constants import SHORE_FIT_NPZ, DWMRI_DUMMY_DATA, \
    ODF_RESULT_HPSD, ODF_RESULT_NONNEG, ODF_RESULT_NO_CONSTRAINT


def test_ShoreFit_deconvolution_hpsd():
    """ Here we test the deconvolution with the hpsd constraint calculated in world coordinates

    """
    fit = bs.ShoreFit.old_load(SHORE_FIT_NPZ)
    out, wmout, gmout, csfout, mask, meta = fit.fodf(DWMRI_DUMMY_DATA, pos='hpsd')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_HPSD)

    assert (out == tensors).all()

def test_ShoreFit_deconvolution_nonneg():
    """ Here we test the deconvolution with the hpsd constraint calculated in world coordinates

    """
    fit = bs.ShoreFit.old_load(SHORE_FIT_NPZ)
    out, wmout, gmout, csfout, mask, meta = fit.fodf(DWMRI_DUMMY_DATA, pos='nonneg')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NONNEG)

    assert (out == tensors).all()

def test_ShoreFit_deconvolution_no_constraint():
    """ Here we test the deconvolution with the hpsd constraint calculated in world coordinates

    """
    fit = bs.ShoreFit.old_load(SHORE_FIT_NPZ)
    out, wmout, gmout, csfout, mask, meta = fit.fodf(DWMRI_DUMMY_DATA, pos='none')
    tensors, mask, meta = fields.load_tensor(ODF_RESULT_NO_CONSTRAINT)

    assert (out == tensors).all()
