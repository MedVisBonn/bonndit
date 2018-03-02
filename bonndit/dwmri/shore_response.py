
from __future__ import print_function, division


def get_response_function(data, gtab, mask, radial_order, zeta, tau)

    return asm

def estimate_response_functions(data, wm_mask, gm_mask, csf_mask, gtab, dti_fa, dti_vecs, dti_mask=None, radial_order=4, zeta=700,
                      tau=1 / (4 * np.pi ** 2), fawm=0.7, verbose = False):
    """
    This function calculates the response Function needed for the deconvolution of the diffusion imaging signal.

    :param data: The diffusion weighted data
    :param wm_mask: The white matter mask
    :param gm_mask: The grey matter mask
    :param csf_mask: The cerebrospinal fluid mask
    :param dti_fa:
    :param dti_vecs:
    :param gtab: This has to be a GradientTable object from dipy. It can be created from b-values and b-vectors using dipy.core.gradients.gradient_table(bvals, bvecs).
    :param dti_mask:
    :param order: The order of the shore basis. The default is 4.
    :param zeta: The radial scaling factor. The default is 700.
    :param tau: The q-scaling factor. The default is 1 / (4 * np.pi ** 2)
    :param fawm: The threshold for the white matter fractional anisotropy. The default is 0.7
    :return: ResponseFunction -- holding all important values.
    """

    # Load DTI fa map
    fa = dti_fa.get_data()

    # Load DTI vecs
    vecs = dti_vecs.get_data()

    # Load DTI mask if available
    if dti_mask is None:
        NX, NY, NZ = fa.shape
        mask = np.ones((NX, NY, NZ))
    else:
        mask = dti_mask.get_data()

    # Create masks
    # CSF
    csf = csf_mask.get_data()
    mask_csf = np.logical_and(mask, np.logical_and(csf > 0.95, fa < 0.2)).astype('int')
    # GM
    gm = gm_mask.get_data()
    mask_gm = np.logical_and(mask, np.logical_and(gm > 0.95, fa < 0.2)).astype('int')
    # WM
    wm = wm_mask.get_data()
    mask_wm = np.logical_and(mask, np.logical_and(wm > 0.95, fa > float(fawm))).astype('int')

    # Load data
    data = data.get_data()

    # Reshape data
    NX, NY, NZ = data.shape[0:3]
    N = NX * NY * NZ
    data = data.reshape((N, -1))
    vecs = vecs.reshape((N, 3))
    mask_csf = mask_csf.flatten()
    mask_gm = mask_gm.flatten()
    mask_wm = mask_wm.flatten()

    response_parameters = (radial_order, zeta, tau)
    # Calculate csf response
    csf_shore_fit = ShoreModel(gtab, radial_order, zeta, tau).fit(data, mask)
    #shore_coeff = get_response(data, gtab, mask_csf, *response_parameters)
    signal_csf = shore.compress(shore_coeff)

    # Calculate gm response
    shore_coeff = get_response(data, gtab, mask_gm, *response_parameters)
    signal_gm = shore.compress(shore_coeff)

    # Calculate wm response
    shore_coeff = get_response_reorient(data, gtab, mask_wm, vecs, *response_parameters)
    signal_wm = shore.compress(shore_coeff)

    return ShoreResponseFunctions(signal_wm, signal_gm, signal_csf, zeta, tau)




class ShoreResponseFunctions():
    def __init__(self, wm_signal, gm_signal, csf_signal, zeta, tau,  *args, ** kwargs):
        """


        :param args:
        :param kwargs:
        """

        self.csf_signal = csf_signal
        self.wm_signal = wm_signal
        self.gm_signal = gm_signal
        self.zeta = zeta
        self.tau = tau



    def save(self, filename = 'response.npz'):
        """

        :param filename:
        """
        np.savez(filename, csf=self.csf_signal, gm=self.gm_signal, wm=self.wm_signal,
                 zeta=self.zeta, tau=self.tau)
