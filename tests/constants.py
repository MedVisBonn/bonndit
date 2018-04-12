import os

dir_path = os.path.dirname(os.path.realpath(__file__))


DECONVOLUTION_DIR = os.path.join(dir_path, 'data/deconvolution/')
DECONVOLUTION_VOLUMES_DIR = os.path.join(DECONVOLUTION_DIR, 'volume_fractions')

DWMRI_DUMMY_DATA = os.path.join(DECONVOLUTION_DIR, 'dwmridummy.nii.gz')
SHORE_FIT_NPZ = os.path.join(DECONVOLUTION_DIR, 'response.npz')

ODF_RESULT_HPSD = os.path.join(DECONVOLUTION_DIR, 'odfhpsd.nrrd')
ODF_RESULT_NONNEG = os.path.join(DECONVOLUTION_DIR, 'odfnonneg.nrrd')
ODF_RESULT_NO_CONSTRAINT = os.path.join(DECONVOLUTION_DIR, 'odfnoconstraint.nrrd')
