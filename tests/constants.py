import os

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(dir_path, 'data')

DECONVOLUTION_RESULTS_DIR = os.path.join(DATA_DIR,
                                         'results/shore_deconvolution')
CSD_RESULTS_DIR = os.path.join(DATA_DIR, 'results/cs_deconvolution')
KURTOSIS_RESULTS_DIR = os.path.join(DATA_DIR, 'results/dki')

DWMRI_DUMMY_DATA = os.path.join(DATA_DIR, 'data.nii.gz')

# Responses computed with michi-temp shore-response (added fsl flip sign, rotated to worldspace)
SHORE_FIT_TEST = os.path.join(DECONVOLUTION_RESULTS_DIR, 'response.npz')
# Response computed on complete data
SHORE_FIT_PRECOMPUTED = os.path.join(DATA_DIR, 'shore_response_complete.npz')

# Response computed with mic-tools csd-response
SH_RESPONSE = os.path.join(CSD_RESULTS_DIR, 'response.npz')
# Response computed on complete data
SH_RESPONSE_PRECOMPUTED = os.path.join(DATA_DIR, 'sh_response_complete.npz')

# dki tensor computed with kurtosis-cone
DKI_TENSOR = os.path.join(KURTOSIS_RESULTS_DIR, 'kurtosis_fit.npz')

# fodfs calculated with michi-temp shore_deconvolve (added fsl flip sign, rotated to worldspace)
ODF_RESULT_HPSD_RANK1 = os.path.join(DECONVOLUTION_RESULTS_DIR,
                               'odf_hpsd_rank1.nrrd')
ODF_RESULT_NONNEG_RANK1 = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                 'odf_nonneg_rank1.nrrd')
ODF_RESULT_NO_CONSTRAINT_RANK1 = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                        'odf_none_rank1.nrrd')

# fodfs calculated with mic-tools shore-deconvolve using delta kernel
ODF_RESULT_HPSD_DELTA = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                     'odf_hpsd_delta.nrrd')
ODF_RESULT_NONNEG_DELTA = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                       'odf_nonneg_delta.nrrd')
ODF_RESULT_NO_CONSTRAINT_DELTA = os.path.join(DECONVOLUTION_RESULTS_DIR,
                                              'odf_none_delta.nrrd')

# odfs calculated with mic-tools csd-deconvolve using rank1 kernel
CSD_ODF_HPSD_RANK1 = os.path.join(CSD_RESULTS_DIR,
                                  'odf_hpsd_rank1.nrrd')
CSD_ODF_NONNEG_RANK1 = os.path.join(CSD_RESULTS_DIR,
                                    'odf_nonneg_rank1.nrrd')
CSD_ODF_NO_CONSTRAINT_RANK1 = os.path.join(CSD_RESULTS_DIR,
                                           'odf_none_rank1.nrrd')

# odfs calculated with mic-tools csd-deconvolve using delta kernel
CSD_ODF_HPSD_DELTA = os.path.join(CSD_RESULTS_DIR,
                                  'odf_hpsd_delta.nrrd')
CSD_ODF_NONNEG_DELTA = os.path.join(CSD_RESULTS_DIR,
                                    'odf_nonneg_delta.nrrd')
CSD_ODF_NO_CONSTRAINT_DELTA = os.path.join(CSD_RESULTS_DIR,
                                           'odf_none_delta.nrrd')
