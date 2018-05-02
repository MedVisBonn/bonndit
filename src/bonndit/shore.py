from __future__ import division

import errno
import os

import cvxopt
import numpy as np
import numpy.linalg as la
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.reconst.shore import shore_matrix
import nibabel as nib
from tqdm import tqdm

from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.michi import shore, esh, tensor, dwmri
from .gradients import gtab_reorient


class ShoreModel(object):
    """ Fit WM, GM and CSF response functions to the given DW data.
    """
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        self.gtab = gtab
        self.order = order
        self.zeta = zeta
        self.tau = tau

    def fit(self, data, dti_vecs, wm_mask, gm_mask, csf_mask, verbose=False):
        """ Fit the response functions and return the shore coefficients

        :param data:
        :param dti_vecs:
        :param verbose:
        :return:
        """

        # Calculate wm response
        shore_coeffs = self._get_response_reorient(data.get_data(), wm_mask.get_data(), dti_vecs.get_data(),
                                                   verbose, desc='WM response')
        shore_coeff = self._accumulate_shore(shore_coeffs, wm_mask.get_data())
        signal_wm = self._shore_compress(shore_coeff)

        # Calculate gm response
        shore_coeffs = self._get_response(data.get_data(), gm_mask.get_data(), verbose, desc='GM response')
        shore_coeff = self._accumulate_shore(shore_coeffs, gm_mask.get_data())
        signal_gm = self._shore_compress(shore_coeff)

        # Calculate csf response
        shore_coeffs = self._get_response(data.get_data(), csf_mask.get_data(), verbose, desc='CSF response')
        shore_coeff = self._accumulate_shore(shore_coeffs, csf_mask.get_data())
        signal_csf = self._shore_compress(shore_coeff)

        return ShoreFit(self, [signal_csf, signal_gm, signal_wm])

    def _shore_compress(self, s):
        """ "kernel": only use z-rotational part

        :param s:
        :return:
        """
        r = np.zeros(shore.get_kernel_size(self.order, self.order))
        counter = 0
        ccounter = 0
        for l in range(0, self.order + 1, 2):
            for n in range((self.order - l) // 2 + 1):
                r[ccounter] = s[counter + l]
                counter += 2 * l + 1
                ccounter += 1
        return r

    def _accumulate_shore(self, shore_coeff, mask):
        """ Average over all shore coefficients

        :param shore_coeff:
        :param mask:
        :return:
        """

        shore_accum = np.zeros(shore.get_size(self.order, self.order))
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

        # Do not show divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            return shore_accum / accum_count

    def _get_response(self, data, mask, verbose=False, desc=''):
        """

        :param data:
        :param mask:
        :param verbose:
        :return:
        """
        shore_coeff = np.zeros(data.shape[:-1] + (shore.get_size(self.order, self.order),))
        with np.errstate(divide='ignore', invalid='ignore'):
            shore_m = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:-1]),
                      total=np.prod(data.shape[:-1]),
                      disable=not verbose, desc=desc):
            if mask[i] == 0:
                continue

            # TODO: Decide if rcond=None would be better
            r = la.lstsq(shore_m, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return shore_coeff

    def _get_response_reorient(self, data, mask, vecs, verbose=False, desc=''):
        """

        :param data:
        :param mask:
        :param vecs: First principal direction of diffusion for every voxel
        :param verbose:
        :return:
        """

        shore_coeff = np.zeros(data.shape[:-1] + (shore.get_size(self.order, self.order),))
        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:-1]),
                      total=np.prod(data.shape[:-1]),
                      disable=not verbose, desc=desc):
            if mask[i] == 0:
                continue
            gtab2 = gtab_reorient(self.gtab, vecs[i])
            with np.errstate(divide='ignore', invalid='ignore'):
                shore_m = shore_matrix(self.order, self.zeta, gtab2, self.tau)
            r = la.lstsq(shore_m, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return shore_coeff


class ShoreFit(object):
    def __init__(self, model, shore_coef):
        self.model = model
        self.signal_csf = shore_coef[0]
        self.signal_gm = shore_coef[1]
        self.signal_wm = shore_coef[2]
        self.gtab = model.gtab
        self.order = model.order
        self.zeta = model.zeta
        self.tau = model.tau

    @classmethod
    def load(cls, filepath):
        """

        :param filepath:
        :return:
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = ShoreModel(gtab, response['order'], response['zeta'], response['tau'])
        return cls(model, (response['csf'], response['gm'], response['wm']))

    def save(self, filepath):
        """

        :param filepath:
        :return:
        """
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(filepath, csf=self.signal_csf, gm=self.signal_gm, wm=self.signal_wm,
                 zeta=self.zeta, tau=self.tau, order=self.order, bvals=self.gtab.bvals, bvecs=self.gtab.bvecs)

    def fodf(self, data, pos='hpsd', verbose=False):
        """Deconvolve the signal with the 3 response functions

        :param filename:
        :param pos:
        :param verbose:
        :return:
        """
        # Load nrrd or nifti with the corresponding transformations
        #data, gtab, meta = dwmri.load(filename)

        #nonzero_bvals = self.gtab.bvals[self.gtab.bvals != 0]
        #affines = np.tile(data.affine, (len(nonzero_bvals),1,1))
        #self.gtab = reorient_bvecs(self.gtab, affines)

        data = data.get_data()
        space = data.shape[:-1]

        mask = np.ones(space)
        # TODO: Add possibility to add mask

        # Kernel_ln
        kernel_csf = shore.signal_to_kernel(self.signal_csf, self.order, self.order)
        kernel_gm = shore.signal_to_kernel(self.signal_gm, self.order, self.order)
        kernel_wm = shore.signal_to_kernel(self.signal_wm, self.order, self.order)

        # Build matrix that maps ODF+volume fractions to signal
        # in two steps: First, SHORE matrix
        shore_m = shore.matrix(self.order, self.order, self.zeta, self.gtab, self.tau)

        # then, convolution
        M_wm = shore.matrix_kernel(kernel_wm, self.order, self.order)
        M_gm = shore.matrix_kernel(kernel_gm, self.order, self.order)
        M_csf = shore.matrix_kernel(kernel_csf, self.order, self.order)
        M = np.hstack((M_wm, M_gm[:, :1], M_csf[:, :1]))

        # now, multiply them together
        M = np.dot(shore_m, M)

        with np.errstate(divide='ignore', invalid='ignore'):
            print('Condition number of M:', la.cond(M))

        NN = esh.LENGTH[self.order]

        # positivity constraints
        if pos in ['nonneg', 'hpsd']:
            cvxopt.solvers.options['show_progress'] = False
            # set up QP problem from normal equations
            P = cvxopt.matrix(np.ascontiguousarray(np.dot(M.T, M)))
            # TODO: consider additional Tikhonov regularization

            if pos == 'nonneg':
                # set up non-negativity constraints
                NC = LOTS_OF_DIRECTIONS.shape[0]
                G = np.zeros((NC + 2, NN + 2))
                G[:NC, :NN] = esh.matrix(self.order, LOTS_OF_DIRECTIONS)
                # also constrain GM/CSF VFs to be non-negative
                G[NC, NN] = -1
                G[NC + 1, NN + 1] = -1
                h = np.zeros(NC + 2)
            else:

                # positive definiteness constraint on ODF
                ind = tensor.H_index_matrix(self.order).reshape(-1)
                N = len(ind)

                # set up positive definiteness constraints
                G = np.zeros((N + 2, NN + 2))
                # constrain GM/CSF VFs to be non-negative: orthant constraints
                G[0, NN] = -1
                G[1, NN + 1] = -1
                esh2sym = esh.esh_to_sym_matrix(self.order)
                for i in range(N):
                    G[i + 2, :NN] = -esh2sym[ind[i], :]
                h = np.zeros(N + 2)
                # initialize with partly GM, CSF, and isotropic ODF
                init = np.zeros(NN + 2)
                init[0] = 0.3
                init[1] = 0.3
                init[2] = 0.3 * self.signal_csf[0] / kernel_csf[0][0]
                init = cvxopt.matrix(np.ascontiguousarray(init))
            G = cvxopt.matrix(np.ascontiguousarray(G))
            h = cvxopt.matrix(np.ascontiguousarray(h))

        # deconvolution
        out = np.zeros(space + (NN,))
        gmout = np.zeros(space)
        wmout = np.zeros(space)
        csfout = np.zeros(space)

        for i in tqdm(np.ndindex(*data.shape[:-1]),
                      total=np.prod(data.shape[:-1]),
                      disable=not verbose,
                      desc='Optimization'):
            if mask[i] == 0:
                continue

            S = data[i]
            if pos in ['nonneg', 'hpsd']:
                q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(M.T, S)))
                if pos == 'nonneg':
                    sol = cvxopt.solvers.qp(P, q, G, h)
                    if sol['status'] != 'optimal':
                        print('Optimization unsuccessful.')
                    c = np.array(sol['x'])[:, 0]
                else:
                    c = self.deconvolve_hpsd(P, q, G, h, init, NN)
            else:
                c = la.lstsq(M, S, rcond=-1)[0]
            out[i] = esh.esh_to_sym(c[:NN])
            f = kernel_csf[0][0] / max(self.signal_csf[0], 1e-10)
            wmout[i] = c[0] * f
            gmout[i] = c[NN] * f
            csfout[i] = c[NN + 1] * f

        return out, wmout, gmout, csfout, mask

    def deconvolve_hpsd(self, P, q, G, h, init, NN):
        """ Use Quadratic Cone Program for one shot deconvolution

        :param P:
        :param q:
        :param G:
        :param h:
        :param init:
        :param NN:
        :return:
        """
        # NS = len(np.array(T{4,6,8}.TT).reshape(-1))
        NS = tensor.LENGTH[self.order // 2]

        # first two are orthant constraints, rest positive definiteness
        dims = {'l': 2, 'q': [], 's': [NS]}

        # This init stuff is a HACK. It empirically removes some isolated failure cases
        # first, allow it to use its own initialization
        try:
            sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
        except ValueError as e:
            print("error-----------", e)
            return np.zeros(NN + 2)
        if sol['status'] != 'optimal':
            # try again with our initialization
            try:
                sol = cvxopt.solvers.coneqp(P, q, G, h, dims, initvals={'x': init})
            except ValueError as e:
                print("error-----------", e)
                return np.zeros(NN + 2)
            if sol['status'] != 'optimal':
                print('Optimization unsuccessful.', sol)
        c = np.array(sol['x'])[:, 0]

        return c

def dti_masks(wm_mask, gm_mask, csf_mask, dti_fa, dti_mask, fawm=0.7):
    """ Use the fractional anisotropy calculated using a DTI approach to improve the tissue masks

    :param wm_mask:
    :param gm_mask:
    :param csf_mask:
    :param dti_mask:
    :param dti_fa:
    :param fawm:
    :return:
    """
    # Load DTI fa map
    fa = dti_fa.get_data()

    # Load DTI mask if available
    if dti_mask is None:
        dti_mask = np.ones(fa.shape)
    else:
        dti_mask = dti_mask.get_data()

    # Create masks
    # WM
    wm = wm_mask.get_data()
    dti_wm = np.logical_and(dti_mask, np.logical_and(wm > 0.95, fa > float(fawm))).astype('int')
    # GM
    gm = gm_mask.get_data()
    dti_gm = np.logical_and(dti_mask, np.logical_and(gm > 0.95, fa < 0.2)).astype('int')
    # CSF
    csf = csf_mask.get_data()
    dti_csf = np.logical_and(dti_mask, np.logical_and(csf > 0.95, fa < 0.2)).astype('int')

    wm_img = nib.Nifti1Image(dti_wm, wm_mask.affine)
    gm_img = nib.Nifti1Image(dti_gm, gm_mask.affine)
    csf_img = nib.Nifti1Image(dti_csf, csf_mask.affine)
    return wm_img, gm_img, csf_img
