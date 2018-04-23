import os

from bonndit import ShoreModel, ShoreFit
from bonndit.michi import dwmri, fields
from dipy.core.gradients import gradient_table

from .constants import DECONVOLUTION_DIR, SHORE_FIT_TEST

"""
I would prefer to load the data and do all the transformation explicitly using a 
class written for this purpose. - Explicit is better than implicit.
For now we use the old loading procedure which does all the transformation 
implicitly. 
"""

# Load fractional anisotropy
#dti_fa = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_FA.nii.gz"))
dti_fa, meta = fields.load_scalar(os.path.join(DECONVOLUTION_DIR, "dti_FA.nii.gz"))

# Load DTI mask
#dti_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "mask.nii.gz"))
dti_mask, _ = fields.load_scalar(os.path.join(DECONVOLUTION_DIR, "mask.nii.gz"))

# Load and adjust tissue segmentation masks
#csf_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_0.nii.gz"))
csf_mask, _ = fields.load_scalar(os.path.join(DECONVOLUTION_DIR, "fast_pve_0.nii.gz"))
#gm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_1.nii.gz"))
gm_mask, _ = fields.load_scalar(os.path.join(DECONVOLUTION_DIR, "fast_pve_1.nii.gz"))
#wm_mask = nib.load(os.path.join(DECONVOLUTION_DIR, "fast_pve_2.nii.gz"))
wm_mask, _ = fields.load_scalar(os.path.join(DECONVOLUTION_DIR, "fast_pve_2.nii.gz"))

#dti_vecs = nib.load(os.path.join(DECONVOLUTION_DIR, "dti_V1.nii.gz"))
dti_vecs, _ = fields.load_vector(os.path.join(DECONVOLUTION_DIR, "dti_V1.nii.gz"))

#data = nib.load(os.path.join(DECONVOLUTION_DIR, "data.nii.gz"))

#bvals, bvecs = read_bvals_bvecs(os.path.join(DECONVOLUTION_DIR, "bvals"),
#                                os.path.join(DECONVOLUTION_DIR, "bvecs"))
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

    reference_fit = ShoreFit.old_load(SHORE_FIT_TEST)

    # print((reference_fit.signal_wm - fit.signal_wm) )#/reference_fit.signal_wm)
    # print((reference_fit.signal_gm - fit.signal_gm) )#/ reference_fit.signal_gm)
    # print((reference_fit.signal_csf - fit.signal_csf))  # / reference_fit.signal_csf)

    print(reference_fit.signal_csf)
    print(fit.signal_csf)
    # A small difference to the original result is introduced by using
    # dipy shore_matrix instead of shore.matrix
    assert ((reference_fit.signal_csf - fit.signal_csf) < 1e-9).all() \
           and ((reference_fit.signal_gm - fit.signal_gm) < 1e-9).all() \
           and ((reference_fit.signal_wm - fit.signal_wm) < 1e-9).all()
