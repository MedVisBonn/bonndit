from __future__ import division

import errno

try:
    from itertools import imap
except ImportError:
    # For Python 3 imap was removed as gloabl map now returns an iterator
    imap = map

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

import bonndit as bd
from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.michi import shore, esh, tensor


class ShoreModelMt(object):

    """ Model the diffusion imaging signal using the shore basis functions.

    The main purpose of this class is to estimate tissue response functions
    for white matter, gray matter and cerebrospinal fluid which are returned
    as a mtShoreFit object which enables multi-tissue multi-shell
    deconvolution as described in [1]_.

    The method for fitting arbitrary data to a shore model is also exposed to
    the user and returns a np.ndarray object holding the estimated shore
    coefficients. Multiprocessing is supported for python >= 3 and can speed up
    the computation a lot. Unfortunately threads spawned by numpy interfere
    with this multiprocessing. For optimal performance you need to set the
    environment variable OMP_NUM_THREADS to 1. You can do this from within a
    python script by inserting the following code before importing numpy:
    ```os.environ["OMP_NUM_THREADS"] = "1"```

    References
    ----------
    .. [1] M. Ankele, L. Lim, S. Groeschel and T. Schultz; "Versatile, Robust
    and Efficient Tractography With Constrained Higher-Order Tensor fODFs";
    Int J Comput Assist Radiol Surg. 2017 Aug; 12(8):1257-1270;
    doi: 10.1007/s11548-017-1593-6
    """

    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        """

        :param gtab:
        :param order:
        :param zeta:
        :param tau:
        """
        self.gtab = gtab
        self.order = order
        self.zeta = zeta
        self.tau = tau

        # Ignore division by zero warning
        # dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.shore_m = shore_matrix(self.order, self.zeta, self.gtab,
                                        self.tau)

    def fit(self, data, dti_vecs, wm_mask, gm_mask, csf_mask,
            verbose=False, cpus=1):
        """ Compute tissue response functions.

        Shore coefficients are fitted for white matter, gray matter and
        cerebrospinal fluid separately. The averaged and compressed
        coefficients are returned in a mtShoreFit object.

        :param data: diffusion weighted data
        :param dti_vecs: first eigenvector of a precalculated diffusion tensor
        :param wm_mask: white Matter Mask (0/1)
        :param gm_mask: gray Matter Mask (0/1)
        :param csf_mask: cerebrospinal fluid mask (0/1)
        :param verbose: Set to True for a progress bar
        :param cpus: Number of cpu workers to use
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
            raise ValueError('No CSF voxels specified by csf_mask. '
                             'A corresponding response can not be computed.')

        # Calculate wm response
        wm_voxels = data.get_data()[wm_mask.get_data() == 1]
        wm_vecs = dti_vecs.get_data()[wm_mask.get_data() == 1]
        wmshore_coeffs = bd.ShoreModel(self.gtab, self.order, self.zeta,
                                       self.tau).fit(wm_voxels, wm_vecs,
                                                     verbose=verbose,
                                                     cpus=cpus,
                                                     desc='WM response').coefs
        wmshore_coeff = self.shore_accumulate(wmshore_coeffs)
        signal_wm = self.shore_compress(wmshore_coeff)

        # Calculate gm response
        gm_voxels = data.get_data()[gm_mask.get_data() == 1]
        gmshore_coeffs = bd.ShoreModel(self.gtab, self.order, self.zeta,
                                       self.tau).fit(gm_voxels,
                                                     verbose=verbose,
                                                     cpus=cpus,
                                                     desc='GM response').coefs
        gmshore_coeff = self.shore_accumulate(gmshore_coeffs)
        signal_gm = self.shore_compress(gmshore_coeff)

        # Calculate csf response
        csf_voxels = data.get_data()[csf_mask.get_data() == 1]
        csfshore_coeffs = bd.ShoreModel(self.gtab, self.order, self.zeta,
                                        self.tau).fit(csf_voxels,
                                                      verbose=verbose,
                                                      cpus=cpus,
                                                      desc='CSF response').coefs
        csfshore_coeff = self.shore_accumulate(csfshore_coeffs)
        signal_csf = self.shore_compress(csfshore_coeff)

        return ShoreFitMt(self, [signal_csf, signal_gm, signal_wm])

    def shore_compress(self, coefs):
        """ Compress the shore coefficients

        An axial symetric response function aligned to the z-axis can be
        described fully using only the z-rotational part of the shore
        coefficients.

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

    def shore_accumulate(self, shore_coeff):
        """ Average over array of shore coefficients.

        This is used to determine the average response of a specific tissue.

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


class ShoreFitMt(object):

    def __init__(self, model, shore_coef, kernel="rank1"):
        """

        :param model:
        :param shore_coef:
        :param kernel:
        """
        self.model = model
        self.signal_csf = shore_coef[0]
        self.signal_gm = shore_coef[1]
        self.signal_wm = shore_coef[2]
        self.gtab = model.gtab
        self.order = model.order
        self.zeta = model.zeta
        self.tau = model.tau
        # The deconvolution kernels are computed in set_kernel
        self.kernel_type = kernel
        self.kernel_csf = None
        self.kernel_gm = None
        self.kernel_wm = None
        self.set_kernel(kernel)

    def set_kernel(self, kernel):
        """

        :param kernel:
        :return:
        """
        # Get deconvolution kernels
        if kernel == "rank1":
            self.kernel_csf = shore.signal_to_rank1_kernel(self.signal_csf,
                                                           self.order)
            self.kernel_gm = shore.signal_to_rank1_kernel(self.signal_gm,
                                                          self.order)
            self.kernel_wm = shore.signal_to_rank1_kernel(self.signal_wm,
                                                          self.order)
        elif kernel == "delta":
            self.kernel_csf = shore.signal_to_delta_kernel(self.signal_csf,
                                                           self.order)
            self.kernel_gm = shore.signal_to_delta_kernel(self.signal_gm,
                                                          self.order)
            self.kernel_wm = shore.signal_to_delta_kernel(self.signal_wm,
                                                          self.order)
        else:
            msg = "{} is not a valid option for kernel. " \
                  "Use 'rank1' or 'delta'.".format(kernel)
            raise ValueError(msg)

    @classmethod
    def load(cls, filepath):
        """ Load a precalculated mtShoreFit object from a file.

        :param filepath: path to the saved mtShoreFit object
        :return: mtShoreFit object which contains response functions for white
        matter, gray matter and CSF
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = ShoreModelMt(gtab, response['order'], response['zeta'],
                             response['tau'])

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

        np.savez(filepath, csf=self.signal_csf, gm=self.signal_gm,
                 wm=self.signal_wm, zeta=self.zeta, tau=self.tau,
                 order=self.order, bvals=self.gtab.bvals,
                 bvecs=self.gtab.bvecs)

    def _convolve_helper(self, fODF_volfracs):
        """

        :param fODF_volfracs:
        :return:
        """
        fODF = fODF_volfracs[0]
        vol_fraction = fODF_volfracs[1]
        conv_matrix = self.shore_convolution_matrix()

        x = np.append(esh.sym_to_esh(fODF), vol_fraction)
        signal = np.dot(conv_matrix, x)

        return signal

    def convolve(self, fODFs, vol_fractions=None, S0=None, kernel="rank1",
                 verbose=False, cpus=None, desc=""):
        """Convolve the Shore Fit with several fODFs.

        The multiprocessing for this function scales along the number of fODFs.
        For a small number of fODFs it makes sense to specify cpus=1 to avoid
        the overhead of spawning multiple processes. If you need to convolve a
        large number of Shore Fits with several fODFs each better use "" which
        scales along the number of Shore Fits.

        :param fODFs:
        :param vol_fractions:
        :param S0:
        :param verbose:
        :param cpus:
        :return:
        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)

        if vol_fractions is None:
            vol_fractions = np.array([[0, 0]] * np.prod(fODFs.shape[:-1]))

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(fODFs.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3 or cpus == 1:
            signals = list(tqdm(imap(self._convolve_helper,
                                        zip(list(fODFs),
                                            list(vol_fractions))),
                                total=np.prod(fODFs.shape[:-1]),
                                disable=not verbose,
                                desc=desc))
        else:
            with mp.Pool(cpus) as p:
                signals = list(tqdm(p.imap(self._convolve_helper,
                                           zip(list(fODFs),
                                               list(vol_fractions)),
                                           chunksize),
                                    total=np.prod(fODFs.shape[:-1]),
                                    disable=not verbose,
                                    desc=desc))

        if S0:
            signals = np.array(signals) * S0
        else:
            signals = np.array(signals)

        return signals

    def fodf(self, data, pos='hpsd', mask=None, kernel="rank1", verbose=False,
             cpus=1):
        """ Deconvolve DWI data with multiple tissue response [1]_.

        :param data: diffusion weighted data
        :param pos: constraint choose between hpsd, nonneg and none
        :param mask: voxels for which fODFs and volume fractions are calculated
        :param verbose: set to true to show a progress bar
        :param cpus: number of cpus (if None use value from os.cpu_count())
        :return: fodfs, wm volume fraction, gm volume fraction and
        csf volume fraction


        References
        ----------
        .. [1] M. Ankele, L. Lim, S. Groeschel and T. Schultz; "Versatile,
        Robust and Efficient Tractography With Constrained Higher-Order
        Tensor fODFs"; Int J Comput Assist Radiol Surg. 2017 Aug;
        12(8):1257-1270; doi: 10.1007/s11548-017-1593-6
        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)

        data = data.get_data()
        space = data.shape[:-1]

        if not mask:
            mask = np.ones(space)
        else:
            mask = mask.get_data()
        # Convert integer to boolean mask
        mask = np.ma.make_mask(mask)

        # Create convolution matrix
        conv_mat = self.shore_convolution_matrix()
        with np.errstate(divide='ignore', invalid='ignore'):
            cond_number = la.cond(conv_mat)
            # For the proposed method of kernel rank1 and order 4 the condition
            # number of the matrix should not be larger than 1000 otherwise
            # show a warnig
            if kernel == "rank1" and self.order == 4 and cond_number > 1000:
                logging.warning("For kernel=rank1 and order=4 the condition"
                                "number of the convolution matrix should be "
                                "smaller than 1000. The condition number is:",
                                cond_number)
            else:
                logging.info('Condition number of convolution matrtix:',
                             cond_number)

        # 100 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

        # TODO: consider additional Tikhonov regularization
        # Deconvolve the DWI signal
        deconv = {'none': self.deconvolve, 'hpsd': self.deconvolve_hpsd,
                  'nonneg': self.deconvolve_nonneg}
        data = data[mask, :]
        try:
            func = deconv[pos]
            if sys.version_info[0] < 3 or cpus == 1:
                result = list(tqdm(imap(partial(func, conv_matrix=conv_mat),
                                        data),
                                   total=np.prod(data.shape[:-1]),
                                   disable=not verbose,
                                   desc='Optimization'))
            else:
                with mp.Pool(cpus) as p:
                    result = list(tqdm(p.imap(partial(func,
                                                      conv_matrix=conv_mat),
                                              data, chunksize=chunksize),
                                       total=np.prod(data.shape[:-1]),
                                       disable=not verbose,
                                       desc='Optimization'))
        except KeyError:
            raise ValueError(('"{}" is not supported as a constraint, please' +
                              ' choose from [hpsd, nonneg, none]').format(pos))

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
            deconvolution_result[i] = la.lstsq(conv_matrix, signal,
                                               rcond=None)[0]

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
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T,
                                                      conv_matrix)))

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
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(conv_matrix.T,
                                                               signal)))

            # NS = len(np.array(T{4,6,8}.TT).reshape(-1))
            NS = tensor.LENGTH[self.order // 2]

            # first two are orthant constraints, rest positive definiteness
            dims = {'l': 2, 'q': [], 's': [NS]}

            # This init stuff is a HACK.
            # It empirically removes some isolated failure cases
            # first, allow it to use its own initialization
            try:
                sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
            except ValueError as e:
                logging.error("Error with cvxopt initialization: {}".format(e))
                return np.zeros(NN + 2)
            if sol['status'] != 'optimal':
                # try again with our initialization
                try:
                    sol = cvxopt.solvers.coneqp(P, q, G, h, dims,
                                                initvals={'x': init})
                except ValueError as e:
                    logging.error("Error with custom initialization: "
                                  "{}".format(e))
                    return np.zeros(NN + 2)
                if sol['status'] != 'optimal':
                    logging.debug('Optimization unsuccessful - '
                                  'Constraint: {}'.format('hpsd'))

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
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T,
                                                      conv_matrix)))

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
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(conv_matrix.T,
                                                               signal)))

            sol = cvxopt.solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                logging.debug('Optimization unsuccessful - '
                              'Voxel: {}, Constraint: {}'.format(i, 'nonneg'))

            deconvolution_result[i] = np.array(sol['x'])[:, 0]

        return deconvolution_result

    def shore_convolution_matrix(self, kernel="rank1"):
        """

        :return:
        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)


        # Build matrix that maps ODF+volume fractions to signal
        # in two steps: First, SHORE matrix

        # Ignore division by zero warning
        # dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            shore_m = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # then, convolution
        M_wm = shore.matrix_kernel(self.kernel_wm, self.order)
        M_gm = shore.matrix_kernel(self.kernel_gm, self.order)
        M_csf = shore.matrix_kernel(self.kernel_csf, self.order)
        M = np.hstack((M_wm, M_gm[:, :1], M_csf[:, :1]))

        # now, multiply them together
        return np.dot(shore_m, M)


def dti_masks(wm_mask, gm_mask, csf_mask, dti_fa, dti_mask=None, fawm=0.7):
    """ Create FA guided tissue masks.

    Use precalculated fractional anisotropy values for example from DTI to
    create tissue masks which contain mainly a single tissue. These masks can
    be used to calculate shore response functions for the individual tissues.

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
    wm_by_fa = np.logical_and(wm > 0.95, fa > float(fawm))
    dti_wm = np.logical_and(dti_mask, wm_by_fa).astype('int')
    # GM
    gm = gm_mask.get_data()
    gm_by_fa = np.logical_and(gm > 0.95, fa < 0.2)
    dti_gm = np.logical_and(dti_mask, gm_by_fa).astype('int')
    # CSF
    csf = csf_mask.get_data()
    csf_by_fa = np.logical_and(csf > 0.95, fa < 0.2)
    dti_csf = np.logical_and(dti_mask, csf_by_fa).astype('int')

    wm_img = nib.Nifti1Image(dti_wm, wm_mask.affine)
    gm_img = nib.Nifti1Image(dti_gm, gm_mask.affine)
    csf_img = nib.Nifti1Image(dti_csf, csf_mask.affine)
    return wm_img, gm_img, csf_img
