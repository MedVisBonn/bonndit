import os

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(dir_path, 'data')

DECONVOLUTION_RESULTS_DIR = os.path.join(DATA_DIR, 'results/deconvolution')
KURTOSIS_RESULTS_DIR = os.path.join(DATA_DIR, 'results/kurtosis')

DWMRI_DUMMY_DATA = os.path.join(DATA_DIR, 'data.nii.gz')

# Responses computed with michi-temp shore-response (added fsl flip sign, rotated to worldspace)
SHORE_FIT_TEST = os.path.join(DECONVOLUTION_RESULTS_DIR, 'response.npz')
SHORE_FIT_PRECOMPUTED = os.path.join(DATA_DIR, 'response_complete.npz')

# fodfs calculated with michi-temp shore_deconvolve (added fsl flip sign, rotated to worldspace)
ODF_RESULT_HPSD = os.path.join(DECONVOLUTION_RESULTS_DIR,
                               'odf_hpsd_rank1.nrrd')
ODF_RESULT_NONNEG = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                 'odf_nonneg_rank1.nrrd')
ODF_RESULT_NO_CONSTRAINT = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                        'odf_none_rank1.nrrd')
