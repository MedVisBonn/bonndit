import os
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from bonndit.michi import dwmri

from bonndit.shore import ShoreModel, ShoreFit
from .constants import DECONVOLUTION_DIR, DECONVOLUTION_RESULTS_DIR

# Load fractional anisotropy
dti_fa = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_FA.nii.gz"))

# Load DTI mask
dti_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "mask.nii.gz"))

# Load and adjust tissue segmentation masks
csf_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_0.nii.gz"))
gm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_1.nii.gz"))
wm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_2.nii.gz"))

dti_vecs = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_V1.nii.gz"))

#data = nib.load(os.path.join(DECONVOLUTION_DIR, "data.nii.gz"))

bvals, bvecs = read_bvals_bvecs(os.path.join(DECONVOLUTION_DIR, "bvals"),
                                os.path.join(DECONVOLUTION_DIR, "bvecs"))
#gtab = gradient_table(bvals, bvecs)
data, gtabm, meta = dwmri.load(os.path.join(DECONVOLUTION_DIR, "data.nii.gz"))

gtab = gradient_table(gtabm.bvals, gtabm.bvecs)

def test_ShoreModel():
    """ Here we test the calculation of the response functions
    The result calculated with bonndit are compared to results from the old code
    """
    model = ShoreModel(gtab)
    fit = model.fit(data, wm_mask, gm_mask, csf_mask, dti_mask,
                    dti_fa, dti_vecs)

    reference_fit = ShoreFit.old_load(os.path.join(DECONVOLUTION_RESULTS_DIR, "response.npz"))

    print((reference_fit.signal_wm - fit.signal_wm)/reference_fit.signal_wm)
    assert (reference_fit.signal_csf == fit.signal_csf).all() \
    and (reference_fit.signal_gm == fit.signal_gm).all() \
    and (reference_fit.signal_wm == fit.signal_wm).all()

