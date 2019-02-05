import errno
import logging
import multiprocessing as mp
import os
from functools import partial

import cvxopt
import numpy as np
import numpy.linalg as la
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import real_sph_harm
from tqdm import tqdm

from bonndit.base import ReconstModel, ReconstFit
from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.gradients import gtab_reorient
from bonndit.michi import esh, tensor
from bonndit.multivoxel import MultiVoxel, MultiVoxelFitter


class SphericalHarmonicsModel(ReconstModel):
    def __init__(self, gtab, order=4):
        """ Model the diffusion imaging signal with spherical harmonics

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            b-values and b-vectors in a GradientTable object
        order : int
            An even integer representing the order of the shore basis
        """
        super().__init__(gtab)
        self.order = order

        # These parameters are saved for reinitalization
        self._params_dict = {'bvals': gtab.bvals, 'bvecs': gtab.bvecs,
                             'order': order}

        # Ignore division by zero warning
        # dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.sh_m = esh_matrix(self.order, self.gtab)

    def _fit_helper(self, data, vecs=None, rcond=None, **kwargs):
        """

        Parameters
        ----------
        data
        vecs
        rcond
        kwargs

        Returns
        -------

        """
        # Calculate average b0 signal in data
        b0_avg = np.mean(data[..., self.gtab.b0s_mask])

        # Remove small bvalues, depends on b0_threshold of gtab
        data = data[..., ~self.gtab.b0s_mask]

        if vecs is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                sh_m = esh_matrix(self.order, gtab_reorient(self.gtab, vecs))
                sh_m = sh_m[~self.gtab.b0s_mask, :]

        else:
            sh_m = self.sh_m

        coeffs = la.lstsq(sh_m, data, rcond)[0]
        return SphericalHarmonicsFit(self, np.array(coeffs), b0_avg)

    def fit(self, data, vecs=None, mask=None, **kwargs):
        """

        Parameters
        ----------
        data
        vecs
        mask
        kwargs

        Returns
        -------

        """
        if vecs is not None:
            per_voxel_data = {'vecs': vecs}
        else:
            per_voxel_data = {}
        return MultiVoxelFitter(self, **kwargs).fit(self._fit_helper, data,
                                                    per_voxel_data, mask)


class SphericalHarmonicsFit(ReconstFit):
    def __init__(self, model, coeffs, b0_avg):
        """

        Parameters
        ----------
        model
        coeffs
        b0_avg
        """
        super().__init__(coeffs)
        self.model = model
        self.b0_avg = b0_avg

        self.order = model.order
        self.gtab = model.gtab

    def predict(self, gtab):
        """

        Parameters
        ----------
        gtab

        Returns
        -------

        """
        super().predict(gtab)

    @classmethod
    def load(cls, filepath):
        """

        Parameters
        ----------
        filepath

        Returns
        -------

        """
        return MultiVoxel.load(filepath, model_class=SphericalHarmonicsModel,
                               fit_class=cls)


class ShResponseEstimator(object):
    def __init__(self, gtab, order=4):
        """

        Parameters
        ----------
        gtab
        order
        """
        self.gtab = gtab
        self.order = order

    def fit(self, data, dti_vecs, wm_mask, verbose=False, cpus=1):
        """

        Parameters
        ----------
        data
        dti_vecs
        wm_mask
        verbose
        cpus

        Returns
        -------

        """
        # Check if tissue masks give at least a single voxel
        if np.sum(wm_mask.get_data()) < 1:
            raise ValueError('No white matter voxels specified by wm_mask. '
                             'A corresponding response can not be computed.')

        # Select white matter voxels
        wm_voxels = data.get_data()[wm_mask.get_data() == 1]
        wm_vecs = dti_vecs.get_data()[wm_mask.get_data() == 1]

        # Calculate white matter response
        wm_sh_coeffs = SphericalHarmonicsModel(
            self.gtab, self.order).fit(wm_voxels, vecs=wm_vecs,
                                       verbose=verbose,
                                       cpus=cpus,
                                       desc='WM response').coeffs
        wm_sh_coef = self.sh_accumulate(wm_sh_coeffs)
        signal_wm = self.sh_compress(wm_sh_coef)

        print(self.gtab.bvals)
        return ShResponse(self, signal_wm)

    def sh_accumulate(self, sh_coeffs):
        """

        Parameters
        ----------
        sh_coeffs

        Returns
        -------

        """
        sh_accum = np.zeros_like(sh_coeffs[0])
        accum_count = 0

        # Iterate over the data indices
        for i in np.ndindex(*sh_coeffs.shape[:-1]):
            sh_accum += sh_coeffs[i]
            accum_count += 1
        if accum_count == 0:
            return sh_accum

        # Do not show divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            return sh_accum / accum_count

    def sh_compress(self, coeffs):
        """ Extract the z-rotational part from spherical harmonics coefficients

        An axial symetric response function aligned to the z-axis can be
        described fully using only the z-rotational part of the spherical
        harmonics coefficients. This functions selects the zonal harmonics with
        even order from an array with spherical harmonics coefficients.

        Parameters
        ----------
        coeffs : ndarray (n)
            N-dimensional array holding spherical harmonics coefficients of a
            single model

        Returns
        -------
        ndarray
            z-rotational part of the given spherical harmonics coefficients
        """
        zonal_coeffs = np.zeros(esh.get_kernel_size(self.order))

        counter = 0
        for l in range(0, self.order + 1):
            counter = counter + l
            if l % 2 == 0:
                zonal_coeffs[int(l / 2)] = coeffs[counter]

        # This is what happens above
        # r[0] = coeffs[0]
        # r[1] = coeffs[3]
        # r[2] = coeffs[10]
        # ...

        return zonal_coeffs


class ShResponse(object):
    def __init__(self, model, sh_coef, kernel="rank1"):
        """

        Parameters
        ----------
        model
        sh_coef
        kernel
        """
        self.model = model
        self.gtab = model.gtab
        self.order = model.order

        self.wm_response = sh_coef

        # The deconvolution kernels are computed in set_kernel
        self.kernel_type = kernel
        self.kernel_wm = None
        self.set_kernel(kernel)

    def set_kernel(self, kernel):
        """

        Parameters
        ----------
        kernel

        Returns
        -------

        """
        # Get deconvolution kernel
        if kernel == "rank1":
            self.kernel_wm = esh.make_kernel_rank1(self.wm_response)
        elif kernel == "delta":
            self.kernel_wm = esh.make_kernel_delta(self.wm_response)
        else:
            msg = "{} is not a valid option for kernel. " \
                  "Use 'rank1' or 'delta'.".format(kernel)
            raise ValueError(msg)


    @classmethod
    def load(cls, filepath):
        """ Load a precalculated ShResponse object from a file

        Parameters
        ----------
        filepath : str
            Path to the saved file

        Returns
        -------
        ShResponse
            Object holding a white matter response function

        """
        response = np.load(filepath)
        gtab = gradient_table(response['bvals'], response['bvecs'])

        model = ShResponseEstimator(gtab, response['order'])

        return cls(model, response['wm_resp'])

    def save(self, filepath):
        """ Save the object to a file

        Parameters
        ----------
        filepath : str
            Path to the file

        Returns
        -------
        None

        """
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(filepath, wm_resp=self.wm_response, order=self.order,
                 bvals=self.gtab.bvals, bvecs=self.gtab.bvecs)

    def fodf(self, data, pos='hpsd', mask=None, kernel="rank1", verbose=False,
             cpus=1):
        """

        Parameters
        ----------
        data
        pos
        mask
        kernel
        verbose
        cpus

        Returns
        -------

        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)

        data = data.get_data()
        # Remove small bvalues, depends on b0_threshold of gtab
        data = data[..., ~self.gtab.b0s_mask]
        space = data.shape[:-1]

        if not mask:
            mask = np.ones(space)
        else:
            mask = mask.get_data()
        # Convert integer to boolean mask
        mask = np.ma.make_mask(mask)

        # Create convolution matrix
        conv_mat = self.sh_convolution_matrix(kernel)
        conv_mat = conv_mat[~self.gtab.b0s_mask, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            cond_number = la.cond(conv_mat)
            logging.info('Condition number of convolution matrix: {:.3f}'
                         ''.format(cond_number))

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

        # Deconvolve the DWI signal
        deconv = {'none': self.deconvolve, 'hpsd': self.deconvolve_hpsd,
                  'nonneg': self.deconvolve_nonneg}
        data = data[mask, :]
        try:
            func = deconv[pos]
            if cpus == 1:
                result = list(
                    tqdm(map(partial(func, conv_matrix=conv_mat),
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
            raise ValueError(
                ('"{}" is not supported as a constraint, please' +
                 ' choose from [hpsd, nonneg, none]').format(pos))

        # Return ODFs and Volume fraction as separate numpy.ndarray objects
        NN = esh.LENGTH[self.order]
        out = np.zeros(space + (NN,))
        wmout = np.zeros(space)

        f = self.kernel_wm[0] / self.wm_response[0]
        wmout[mask] = [x[0] * f for x in result]

        out[mask, 0] = 1
        out[mask, :] = [esh.esh_to_sym(x) for x in result]

        return out, wmout

    def deconvolve(self, data, conv_matrix):
        """

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN,))

        for i in np.ndindex(*data.shape[:-1]):
            signal = data[i]
            deconvolution_result[i] = la.lstsq(conv_matrix, signal,
                                               rcond=None)[0]

        return deconvolution_result

    def deconvolve_hpsd(self, data, conv_matrix):
        """

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN,))

        cvxopt.solvers.options['show_progress'] = False
        # set up QP problem from normal equations
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T,
                                                      conv_matrix)))

        # positive definiteness constraint on ODF
        ind = tensor.H_index_matrix(self.order).reshape(-1)
        N = len(ind)

        # set up positive definiteness constraints
        G = np.zeros((N, NN))

        # constrain GM/CSF VFs to be non-negative: orthant constraints
        esh2sym = esh.esh_to_sym_matrix(self.order)
        for i in range(N):
            G[i, :] = -esh2sym[ind[i], :]
        h = np.zeros(N)

        # initialize with partly GM, CSF, and isotropic ODF
        init = np.zeros(NN)
        init[0] = self.wm_response[0] / self.kernel_wm[0]
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
            dims = {'l': 0, 'q': [], 's': [NS]}

            # This init stuff is a HACK.
            # It empirically removes some isolated failure cases
            # first, allow it to use its own initialization
            try:
                sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
            except ValueError as e:
                logging.error("Error with cvxopt initialization: {}".format(e))
                return np.zeros(NN)
            if sol['status'] != 'optimal':
                # try again with our initialization
                try:
                    sol = cvxopt.solvers.coneqp(P, q, G, h, dims,
                                                initvals={'x': init})
                except ValueError as e:
                    logging.error("Error with custom initialization: "
                                  "{}".format(e))
                    return np.zeros(NN)
                if sol['status'] != 'optimal':
                    logging.debug('Optimization unsuccessful - '
                                  'Constraint: {}'.format('hpsd'))

            deconvolution_result[i] = np.array(sol['x'])[:, 0]

        return deconvolution_result

    def deconvolve_nonneg(self, data, conv_matrix):
        """

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

        """
        NN = esh.LENGTH[self.order]
        deconvolution_result = np.zeros(data.shape[:-1] + (NN,))

        cvxopt.solvers.options['show_progress'] = False
        # set up QP problem from normal equations
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(conv_matrix.T,
                                                      conv_matrix)))

        # set up non-negativity constraints
        NC = LOTS_OF_DIRECTIONS.shape[0]
        G = np.zeros((NC, NN))
        G[:NC, :NN] = -esh.matrix(self.order, LOTS_OF_DIRECTIONS)

        h = np.zeros(NC)

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

    def sh_convolution_matrix(self, kernel="rank1"):
        """

        Parameters
        ----------
        kernel

        Returns
        -------

        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)

        # Build matrix that maps ODF to signal
        M = np.zeros((self.gtab.bvals.shape[0], esh.LENGTH[self.order]))
        r, theta, phi = cart2sphere(self.gtab.bvecs[:, 0],
                                    self.gtab.bvecs[:, 1],
                                    self.gtab.bvecs[:, 2])
        theta[np.isnan(theta)] = 0
        counter = 0
        for l in range(0, self.order + 1, 2):
            for m in range(-l, l + 1):
                M[:, counter] = (
                    real_sph_harm(m, l, theta, phi) * self.kernel_wm[l // 2])
                counter += 1

        return M



def esh_matrix(order, gtab):
    """ Matrix that evaluates SH coeffs in the given directions

    Parameters
    ----------
    order
    gtab

    Returns
    -------

    """
    bvecs = gtab.bvecs
    r, theta, phi = cart2sphere(bvecs[:, 0],
                                bvecs[:, 1],
                                bvecs[:, 2])
    theta[np.isnan(theta)] = 0
    M = np.zeros((bvecs.shape[0], esh.LENGTH[order]))
    counter = 0
    for l in range(0, order + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = real_sph_harm(m, l, theta, phi)
            counter += 1
    return M
