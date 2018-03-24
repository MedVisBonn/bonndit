from __future__ import division
import numpy as np
import numpy.linalg as la
from dipy.reconst.shore import shore_matrix
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table
from tqdm import tqdm


class ShoreModel:
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        self.gtab = gtab
        self.order = order
        self.zeta = zeta
        self.tau = tau

    def fit(self, data, wm_mask, gm_mask, csf_mask, dti_mask, dti_fa, dti_vecs, fawm=0.7, verbose=False):
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

        # Do not show divide by zero warnings
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate csf response
        shore_coeff = self._get_response(data, mask_csf, verbose)
        signal_csf = self._shore_compress(shore_coeff)

        # Calculate gm response
        shore_coeff = self._get_response(data, mask_gm, verbose)
        signal_gm = self._shore_compress(shore_coeff)

        # Calculate wm response
        shore_coeff = self._get_response_reorient(data, mask_wm, vecs, verbose)
        signal_wm = self._shore_compress(shore_coeff)

        return ShoreFit(self, [signal_csf, signal_gm, signal_wm])

    def _shore_compress(self, s):
        """ "kernel": only use z-rotational part

        :param s:
        :return:
        """
        r = np.zeros(get_kernel_size(self.order))
        counter = 0
        ccounter = 0
        for l in range(0, self.order + 1, 2):
            for n in range(l, (self.order - l) // 2 + 1):
                r[ccounter] = s[counter + l]
                counter += 2 * l + 1
                ccounter += 1
        return r

    def _accumulate_shore(self, shore_coeff, mask):
        """Average over all shore coefficients
        """

        shore_accum = np.zeros(shore_get_size(self.order))
        accum_count = 0

        # Iterate over the data indices
        for i in np.ndindex(*mask.shape[:3]):
            if mask[i] == 0:
                continue
            else:
                shore_accum += shore_coeff[i]
                accum_count += 1
        if accum_count == 0:
            return shore_accum
        return shore_accum / accum_count

    def _get_response(self, data, mask, verbose=False):
        """

        """
        shore_coeff = np.zeros(data.shape[:3] + (shore_get_size(self.order),))
        M = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:3]),
                      total=np.prod(data.shape[:3]),
                      disable=not verbose):
            if mask[i] == 0:
                continue

            r = la.lstsq(M, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return self._accumulate_shore(shore_coeff, mask)

    def _get_response_reorient(self, data, mask, vecs, verbose=False):
        """
        vecs: the first principal direction of diffusion for every voxel
        """
        shore_coeff = np.zeros(data.shape[:3] + (shore_get_size(self.order),))

        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:3]),
                      total=np.prod(data.shape[:3]),
                      disable=not verbose):
            if mask[i] == 0:
                continue

            gtab2 = gtab_reorient(self.gtab, vecs[i])
            M = shore_matrix(self.order, self.zeta, gtab2, self.tau)
            r = la.lstsq(M, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return self._accumulate_shore(shore_coeff, mask)


class ShoreFit:
    def __init__(self, model, shore_coef):
        self.model = model
        self.signal_csf = shore_coef[0]
        self.signal_gm = shore_coef[1]
        self.signal_wm = shore_coef[2]
        self.gtab = model.gtab
        self.order = model.order
        self.zeta = model.zeta

    def save(self, outdir):
        np.savez(outdir + 'response.npz', csf=self.signal_csf, gm=self.signal_gm, wm=self.signal_wm,
                 zeta=self.model.zeta, tau=self.model.tau)

    def fiber_orientation_distribution_functions(self):
        # Deconvolve the signal with the 3 response functions
        pass


def gtab_rotate(gtab, rot_matrix):
    N = len(gtab.bvals)
    rot_bvecs = np.zeros((N, 3))
    for i in range(N):
        rot_bvecs[i, :] = np.dot(rot_matrix, gtab.bvecs[i, :])
    return gradient_table(gtab.bvals, rot_bvecs)


def gtab_reorient(gtab, old_vec, new_vec=np.array((0, 0, 1))):
    # rotate gradients to align 1st eigenvector specified direction - default is (0,0,1)
    rot_matrix = vec2vec_rotmat(old_vec, new_vec)
    return gtab_rotate(gtab, rot_matrix)


MAX_ORDER = 12

SIZES = [1, 0, 7, 0, 22, 0, 50, 0, 95, 0, 161, 0, 252]


def shore_get_size(order):
    try:
        return SIZES[order]
    except IndexError:
        raise ValueError('Please specify an order <= 12')


KERNEL_SIZES = [1, 0, 3, 0, 6, 0, 10, 0, 15, 0, 21, 0, 28]


def get_kernel_size(order):
    try:
        return KERNEL_SIZES[order]
    except IndexError:
        raise ValueError('Please specify an order <= 12')
