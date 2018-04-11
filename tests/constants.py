import os

dir_path = os.path.dirname(os.path.realpath(__file__))


DEFAULT_DATA_DIR = dir_path

if not os.path.exists(DEFAULT_DATA_DIR):
    os.makedirs(DEFAULT_DATA_DIR)


DWMRI_DUMMY_DATA = os.path.join(DEFAULT_DATA_DIR, 'data/dwmridummy.nii.gz')
SHORE_FIT_NPZ = os.path.join(DEFAULT_DATA_DIR, 'data/response.npz')

ODF_RESULT_HPSD = os.path.join(DEFAULT_DATA_DIR, 'data/odfhpsd.nrrd')
ODF_RESULT_NONNEG = os.path.join(DEFAULT_DATA_DIR, 'data/odfnonneg.nrrd')
ODF_RESULT_NO_CONSTRAINT = os.path.join(DEFAULT_DATA_DIR, 'data/odfnoconstraint.nrrd')
