import os

dir_path = os.path.dirname(os.path.realpath(__file__))


DEFAULT_DATA_DIR = dir_path

if not os.path.exists(DEFAULT_DATA_DIR):
    os.makedirs(DEFAULT_DATA_DIR)

SHORE_FIT_FILE = os.path.join(DEFAULT_DATA_DIR, 'data/shorefit.pkl')
DWMRI_DUMMY_DATA = os.path.join(DEFAULT_DATA_DIR, 'data/dwmridummy.nii.gz')
ODF_RESULT_HPSD = os.path.join(DEFAULT_DATA_DIR, 'data/odfdummyhpsd.nii.gz')
ODF_RESULT_NO_CONSTRAINT = os.path.join(DEFAULT_DATA_DIR, 'data/odfdummynoconstraint.nii.gz')
ODF_RESULT_HPSD_WORLDC = os.path.join(DEFAULT_DATA_DIR, 'data/odfhpsdworldc.nii.gz')
# Works not in michis script
#ODF_RESULT_NONNEG = os.path.join(DEFAULT_DATA_DIR, 'data/odfdummynonneg.nii.gz')

