from __future__ import division

import errno
import itertools as it
import logging
import multiprocessing as mp
import os
import sys
from functools import partial

import cvxopt
import nibabel as nib
import numpy as np
import numpy.linalg as la
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import shore_matrix
from tqdm import tqdm

from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.michi import shore, esh, tensor
from .gradients import gtab_reorient


class mtShoreModel(object):
    """ Fit WM, GM and CSF response functions to the given diffusion weighted data.
    """
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        self.gtab = gtab
        self.order = order
        self.zeta = zeta
        self.tau = tau

    def fit(self, data, dti_vecs, wm_mask, gm_mask, csf_mask, verbose=False, cpus=None):
        """ Fit the response functions and return the shore coefficients

        :param data: diffusion weighted data
        :param dti_vecs: first eigenvector of a precalculated diffusion tensor
        :param wm_mask: white Matter Mask (0/1)
        :param gm_mask: gray Matter Mask (0/1)
        :param csf_mask: cerebrospinal fluid mask (0/1)
        :param verbose: Set to True for a progress bar
        :return: Fitted response functions in a mtShoreFit object
        """
        # Check if tissue masks give at least a single voxel
        if np.sum(wm_mask.get_data()) < 1:
            raise ValueError('No white matter voxels specified by wm_mask. '
                             'A corresponding response can not be computed.')
        if np.sum(gm_mask.get_data()) < 1:
            raise ValueError('No gray matter voxels specified by gm_mask. '
                             'A corresponding response can not be computed.')
        if np.sum(csf_mask.get_data()) < 1:
            raise ValueError('No cerebrospinal fluid voxels specified by csf_mask. '
                             'A corresponding response can not be computed.')

        # Calculate wm response
        wm_voxels = data.get_data()[wm_mask.get_data() == 1]
        wm_vecs = dti_vecs.get_data()[wm_mask.get_data() == 1]
        wmshore_coeffs = self.get_response_reorient(wm_voxels, wm_vecs, verbose, cpus, desc='WM response')
        wmshore_coeff = self.accumulate_shore(wmshore_coeffs)
        signal_wm = self.shore_compress(wmshore_coeff)

        # Calculate gm response
        gm_voxels = data.get_data()[gm_mask.get_data() == 1]
        gmshore_coeffs = self.get_response(gm_voxels, verbose, cpus, desc='GM response')
        gmshore_coeff = self.accumulate_shore(gmshore_coeffs)
        signal_gm = self.shore_compress(gmshore_coeff)

        # Calculate csf response
        csf_voxels = data.get_data()[csf_mask.get_data() == 1]
        csfshore_coeffs = self.get_response(csf_voxels, verbose, cpus, desc='CSF response')
        csfshore_coeff = self.accumulate_shore(csfshore_coeffs)
        signal_csf = self.shore_compress(csfshore_coeff)

        return mtShoreFit(self, [signal_csf, signal_gm, signal_wm])

    def shore_compress(self, coefs):
        """ "kernel": only use z-rotational part

        :param coefs: shore coefficients
        :return: z-rotational part of the shore coefficients
        """
        r = np.zeros(shore.get_kernel_size(self.order, self.order))
        counter = 0
        ccounter = 0
        for l in range(0, self.order + 1, 2):
            for n in range((self.order - l) // 2 + 1):
                r[ccounter] = coefs[counter + l]
                counter += 2 * l + 1
                ccounter += 1
        return r

    def accumulate_shore(self, shore_coeff):
        """ Average over shore coefficients calculated for voxels of a specific tissue

        :param shore_coeff: array of per voxel shore coefficients
        :return: averaged shore coefficients
        """

        shore_accum = np.zeros_like(shore_coeff[0])
        accum_count = 0

        # Iterate over the data indices
        for i in np.ndindex(*shore_coeff.shape[:-1]):
            shore_accum += shore_coeff[i]
            accum_count += 1
        if accum_count == 0:
            return shore_accum

        # Do not show divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            return shore_accum / accum_count

    def _response_helper(self, b, a, rcond):
        """ This is a helper function for parallelizing the numpy.linalg.lstsq which is needed by
        the get_response. All we do here is to swap the parameters
        a and b to be able to give b as a positional argument even if a is specified as kwarg. We also
        return only the least square solution.

        :param b:
        :param a:
        :param rcond:
        :return:
        """
        return la.lstsq(a, b, rcond)[0]

    def get_response(self, data, verbose=False, cpus=None, desc=''):
        """ Calculate response function for isotropic regions such as gray matter or cerebrospinal fluid.

        :param data: array of voxels for which to calculate shore coefficients
        :param verbose: set to true to show a progress bar
        :param desc: description for the progress bar
        :return: array of per voxel shore coefficients
        """
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))  # 1000 chunks for the progressbar to run smoother

        # Ignore division by zero warning dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            shore_m = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # Iterate over the data indices; show progress with tqdm; multiple processes for python > 3
        if sys.version_info[0] < 3:
            shore_coeff = list(tqdm(it.imap(partial(self._response_helper, a=shore_m, rcond=-1), data),
                                    total=np.prod(data.shape[:-1]),
                                    disable=not verbose,
                                    desc=desc))
        else:
            with mp.Pool(cpus) as p:
                shore_coeff = list(tqdm(p.imap(partial(self._response_helper, a=shore_m, rcond=-1), data, chunksize),
                                        total=np.prod(data.shape[:-1]),
                                        disable=not verbose,
                                        desc=desc))
        return np.array(shore_coeff)

    def _reorient_response_helper(self, data_vecs):
        """

        :param data:
        :param vecs:
        :return:
        """
        data, vecs = data_vecs[0], data_vecs[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            shore_m = shore_matrix(self.order, self.zeta, gtab_reorient(self.gtab, vecs), self.tau)
        return la.lstsq(shore_m, data, rcond=-1)[0]

    def get_response_reorient(self, data, vecs, verbose=False, cpus=None, desc=''):
        """ Calculate white matter response function. Diffusion in white matter is anisotropic. Averaging shore
        coefficients of different voxels can only result in a meaningful response function if fibers in the
        respective voxels are oriented in the same direction (no crossing fibers). Such voxels can be selected by
        filtering for high fractional anisotropy. Gradients for every considered voxel are rotated such that the
        the first eigenvector of a pre calculated diffusion tensor is aligned to the z-axis.

        :param data: array of white matter voxels for which to calculate shore coefficients
        :param vecs: First principal direction of diffusion for every given data voxel
        :param verbose: set to true to show a progress bar
        :param desc: description for the progress bar
        :return: array of per white matter voxel shore coefficients
        """
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))  # 1000 chunks for the progressbar to run smoother

        if sys.version_info[0] < 3:
            shore_coeff = list(tqdm(it.imap(self._reorient_response_helper, zip(list(data), list(vecs))),
                                    total=np.prod(data.shape[:-1]),
                                    disable=not verbose,
                                    desc=desc))
        else:
            with mp.Pool(cpus) as p:
                shore_coeff = list(tqdm(p.imap(self._reorient_response_helper, zip(list(data), list(vecs)), chunksize),
                                        total=np.prod(data.shape[:-1]),
                                        disable=not verbose,
                                        desc=desc))
        return np.array(shore_coeff)


class mtShoreFit(object):

    def __init__(self, model, shore_coef):
        self.model = model
        self.signal_csf = shore_coef[0]
        self.signal_gm = shore_coef[1]
        self.signal_wm = shore_coef[2]
        self.gtab = model.gtab
        self.order = model.order
        self.zeta = model.zeta
        self.tau = model.tau

        # Get deconvolution kernels (Kernel_ln)
        self.kernel_csf = shore.signal_to_kernel(self.signal_csf, self.order, self.order)
        self.kernel_gm = shore.signal_to_kernel(self.signal_gm, self.order, self.order)
        self.kernel_wm = shore.signal_to_kernel(self.signal_wm, self.order, self.order)

    @classmethod
    def load(cls, filepath):
        """ Load a precalculated mtShoreFit object from a file.

        :param filepath: path to the saved mtShoreFit object
        :return: mtShoreFit object which contains response functions for wm, gm and csf
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = mtShoreModel(gtab, response['order'], response['zeta'], response['tau'])
        return cls(model, (response['csf'], response['gm'], response['wm']))

    def save(self, filepath):
        """ Save a mtShoreFit object to a file.

        :param filepath: path to the file
        """
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(filepath, csf=self.signal_csf, gm=self.signal_gm, wm=self.signal_wm,
                 zeta=self.zeta, tau=self.tau, order=self.order, bvals=self.gtab.bvals, bvecs=self.gtab.bvecs)

    def fodf(self, data, pos='hpsd', mask=None, verbose=False, cpus=None):
        """ Multi tissue deconvolution [1]_,


        :param data: diffusion weighted data
        :param pos: constraint choose between hpsd, nonneg and none
        :param mask: specify for which voxel fODFs and volume fractions are calculated
        :param verbose: set to true to show a progress bar
        :param cpus: number of cpus to use (if None use value from os.cpu_count())
        :return: fodfs, wm volume fraction, gm volume fraction and csf volume fraction


        References
        ----------
        .. [1] M. Ankele, L. Lim, S. Groeschel and T. Schultz; "Versatile, Robust and Efficient Tractography
        With Constrained Higher-Order Tensor fODFs"; Int J Comput Assist Radiol Surg. 2017 Aug; 12(8):1257-1270;
        doi: 10.1007/s11548-017-1593-6
        """

        data = data.get_data()
        space = data.shape[:-1]

        if not mask:
            mask = np.ones(space)
        else:
            mask = mask.get_data()
        # Convert integer to boolean mask
        mask = np.ma.make_mask(mask)

        # Create convolution matrix
        conv_matrix = self.shore_convolution_matrix()
        with np.errstate(divide='ignore', invalid='ignore'):
            logging.debug('Condition number of convolution matrtix:', la.cond(conv_matrix))

        chunksize = max(1, int(np.prod(data.shape[:-1]) / 100))  # 100 chunks for the progressbar to run smoother

        # TODO: consider additional Tikhonov regularization
        # Deconvolve the DWI signal
        deconv = {'none': self.deconvolve, 'hpsd': self.deconvolve_hpsd, 'nonneg': self.deconvolve_nonneg}
        data = data[mask, :]
        try:
            func = deconv[pos]
            if sys.version_info[0] < 3:
                result = list(tqdm(it.imap(partial(func, conv_matrix=conv_matrix), data),
                                   total=np.prod(data.shape[:-1]),
                                   disable=not verbose,
                                   desc='Optimization'))
            else:
                with mp.Pool(cpus) as p:
                    result = list(tqdm(p.imap(partial(func, conv_matrix=conv_matrix),
                                             data, chunksize=chunksize),
                                       total=np.prod(data.shape[:-1]),
                                       disable=not verbose,
                                       desc='Optimization'))
        except KeyError:
            raise ValueError(('"{}" is not supported as a constraint,' +
                             ' please choose from [hpsd, nonneg, none]').format(pos))

        # Return fODFs and Volume fractions as separate numpy.ndarray objects
        NN = esh.LENGTH[self.order]
        out = np.zeros(space + (NN,))
        gmout = np.zeros(space)
        wmout = np.zeros(space)
        csfout = np.zeros(space)

        out[mask, :] = [esh.esh_to_sym(x[:NN]) for x in result]
        f = self.kernel_csf[0][0] / max(self.signal_csf[0], 1e-10)
        wmout[mask] = [x[0] * f for x in result]
        gmout[mask] = [x[NN] * f for x in result]
        csfout[mask] = [x[NN + 1] * f for x in result]

        return out, wmout, gmout, csfout

    def deconvolve(self, data, conv_matrix):
        """

        :param data:
        :param conv_matrix:
        :return:
        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN+2,))

        for i in np.ndindex(*data.shape[:-1]):
            signal = data[i]
            deconvolution_result[i] = la.lstsq(conv_matrix, signal, rcond=-1)[0]

        return deconvolution_result

    def deconvolve_hpsd(self, data, conv_matrix):
        """

        :param data:
        :param conv_matrix:
        :return:
        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN + 2,))

        cvxopt.solvers.options['show_progress'] = False
        # set up QP problem from normal equations
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T, conv_matrix)))

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
        init[2] = 0.3 * self.signal_csf[0] / self.kernel_csf[0][0]
        init = cvxopt.matrix(np.ascontiguousarray(init))

        G = cvxopt.matrix(np.ascontiguousarray(G))
        h = cvxopt.matrix(np.ascontiguousarray(h))

        for i in np.ndindex(*data.shape[:-1]):
            signal = data[i]
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(conv_matrix.T, signal)))

            # NS = len(np.array(T{4,6,8}.TT).reshape(-1))
            NS = tensor.LENGTH[self.order // 2]

            # first two are orthant constraints, rest positive definiteness
            dims = {'l': 2, 'q': [], 's': [NS]}

            # This init stuff is a HACK. It empirically removes some isolated failure cases
            # first, allow it to use its own initialization
            try:
                sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
            except ValueError as e:
                logging.error("Error with cvxopt initialization: {}".format(e))
                return np.zeros(NN + 2)
            if sol['status'] != 'optimal':
                # try again with our initialization
                try:
                    sol = cvxopt.solvers.coneqp(P, q, G, h, dims, initvals={'x': init})
                except ValueError as e:
                    logging.error("Error with custom initialization: {}".format(e))
                    return np.zeros(NN + 2)
                if sol['status'] != 'optimal':
                    logging.debug('Optimization unsuccessful - Constraint: {}'.format('hpsd'))

            deconvolution_result[i] = np.array(sol['x'])[:, 0]

        return deconvolution_result

    def deconvolve_nonneg(self, data, conv_matrix):
        """

        :param data:
        :param conv_matrix:
        :return:
        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN + 2,))

        cvxopt.solvers.options['show_progress'] = False
        # set up QP problem from normal equations
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T, conv_matrix)))

        # set up non-negativity constraints
        NC = LOTS_OF_DIRECTIONS.shape[0]
        G = np.zeros((NC + 2, NN + 2))
        G[:NC, :NN] = esh.matrix(self.order, LOTS_OF_DIRECTIONS)
        # also constrain GM/CSF VFs to be non-negative
        G[NC, NN] = -1
        G[NC + 1, NN + 1] = -1
        h = np.zeros(NC + 2)

        G = cvxopt.matrix(np.ascontiguousarray(G))
        h = cvxopt.matrix(np.ascontiguousarray(h))

        for i in np.ndindex(*data.shape[:-1]):
            signal = data[i]
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(conv_matrix.T, signal)))

            sol = cvxopt.solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                logging.debug('Optimization unsuccessful - Voxel: {}, Constraint: {}'.format(i, 'nonneg'))

            deconvolution_result[i] = np.array(sol['x'])[:, 0]

        return deconvolution_result

    def shore_convolution_matrix(self):
        """

        :return:
        """

        # Build matrix that maps ODF+volume fractions to signal
        # in two steps: First, SHORE matrix
        # Ignore division by zero warning dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            shore_m = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # then, convolution
        M_wm = shore.matrix_kernel(self.kernel_wm, self.order, self.order)
        M_gm = shore.matrix_kernel(self.kernel_gm, self.order, self.order)
        M_csf = shore.matrix_kernel(self.kernel_csf, self.order, self.order)
        M = np.hstack((M_wm, M_gm[:, :1], M_csf[:, :1]))

        # now, multiply them together
        return np.dot(shore_m, M)

def dti_masks(wm_mask, gm_mask, csf_mask, dti_fa, dti_mask, fawm=0.7):
    """ Use precalculated fractional anisotropy values for example from DTI to create tissue masks.

    :param wm_mask: white matter mask
    :param gm_mask: gray matter mask
    :param csf_mask: cerebrospinal fluid mask
    :param dti_mask: brain mask
    :param dti_fa: precalculated fractional anisotropy values
    :param fawm: fractional anisotropy threshold for white matter
    :return: Masks for wm, gm and csf
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
