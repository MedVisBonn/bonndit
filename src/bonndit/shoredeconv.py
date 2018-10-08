import errno
import logging
import multiprocessing as mp
import os
from functools import partial

import cvxopt
import nibabel as nib
import numpy as np
import numpy.linalg as la
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import shore_matrix
from tqdm import tqdm

import bonndit as bd
from bonndit.base import ReconstModel, ReconstFit
from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.gradients import gtab_reorient
from bonndit.michi import shore, esh, tensor
from bonndit.multivoxel import MultiVoxel, MultiVoxelFitter


class ShoreModel(ReconstModel):
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        """ Model the diffusion imaging signal using the shore basis

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            b-values and b-vectors in a GradientTable object
        order : int
            An even integer representing the order of the shore basis
        zeta : float
            Radial scaling factor
        tau : float
            Diffusion time
        """

        super().__init__(gtab)
        self.order = order
        self.zeta = zeta
        self.tau = tau

        # These parameters are saved for reinitalization
        self._params_dict = {'bvals': gtab.bvals, 'bvecs': gtab.bvecs,
                             'order': order, 'zeta': zeta, 'tau': tau}

        # Ignore division by zero warning
        # dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.shore_m = shore_matrix(self.order, self.zeta, self.gtab,
                                        self.tau)

    def _fit_helper(self, data, vecs=None, rcond=None, **kwargs):
        """ Fitting is done here

        This function is handed to the MultivoxelFitter, to fit models for
        every voxel.

        Parameters
        ----------
        data : ndarray (n)
            Data of a single voxel
        vecs : ndarray (3)
             First eigenvector of the diffusion tensor for a single voxel
        rcond :
            Cut-off ratio for small singular values of the coefficient matrix.
            For further information read documentation of numpy.linalg.lstsq.
        kwargs : dict
            Empty dictionary, not used in this function

        Returns
        -------
        ShoreFit
            Object holding the fitted model parameters

        """

        if vecs is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                shore_m = shore_matrix(self.order, self.zeta,
                                       gtab_reorient(self.gtab, vecs),
                                       self.tau)

        else:
            shore_m = self.shore_m
        coeffs = la.lstsq(shore_m, data, rcond)[0]
        return ShoreFit(np.array(coeffs))

    def fit(self, data, vecs=None, mask=None, **kwargs):
        """ Fit shore coefficients to diffusion weighted imaging data.

        If an array of vectors is specified (vecs), a rotation is applied to
        the b-vectors which would align the given vector to the z-axis.  This
        can be used to compute comparable shore coefficients for white matter
        regions of different orientation. Use the first eigenvectors of
        precomputed diffusion tensors as vectors and use only regions with high
        fractional anisotropy to ensure working only with single fiber voxels.

        Parameters
        ----------
        data : ndarray (..., n)
            Diffusion weighted imaging data
        vecs: ndarray (..., 3)
            First eigenvector of the diffusion tensor for every voxel
            (same shape as data)
        mask : ndarray (..., bool)
            Mask (same shape as data)
        kwargs


        Returns
        -------
        MultiVoxel
            Object which holds the fitted models for all voxels.
        """
        if vecs is not None:
            per_voxel_data = {'vecs': vecs}
        else:
            per_voxel_data = {}
        return MultiVoxelFitter(self, **kwargs).fit(self._fit_helper, data,
                                                    per_voxel_data, mask)


class ShoreFit(ReconstFit):
    def __init__(self, coeffs):
        """ Hold fitted model parameters and provide further functionality

        Parameters
        ----------
        coeffs : ndarray(n)
            Fitted model parameter
        """
        super().__init__(coeffs)

    @classmethod
    def load(cls, filepath):
        """ Load a MultiVoxel object

        Parameters
        ----------
        filepath: str
            Path to the saved file

        Returns
        -------
        MultiVoxel
            Object holding ShoreFits for every voxel
        """
        return MultiVoxel.load(filepath, model_class=ShoreModel,
                               fit_class=cls)

    def predict(self, gtab):
        """ Predict DWI measurements. Not yet implemented

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            Gradients for which to predict the measurements

        Returns
        -------
        ndarray (n)
            Predicted measurements for a single voxel
        """
        super().predict(gtab)


class ShoreMultiTissueResponseEstimator(object):
    """ Model the diffusion imaging signal using the shore basis functions.

    The main purpose of this class is to estimate tissue response functions
    for white matter, gray matter and cerebrospinal fluid which are returned
    as a mtShoreFit object which enables multi-tissue multi-shell
    deconvolution as described in [1]_.

    References
    ----------
    .. [1] M. Ankele, L. Lim, S. Groeschel and T. Schultz; "Versatile, Robust
      and Efficient Tractography With Constrained Higher-Order Tensor fODFs";
      Int J Comput Assist Radiol Surg. 2017 Aug; 12(8):1257-1270;
      doi: 10.1007/s11548-017-1593-6

    """

    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        """

        Parameters
        ----------
        gtab : dipy.data.GradientTable
            b-values and b-vectors in a GradientTable object
        order : int
            An even integer representing the order of the shore basis
        zeta : float
            Radial scaling factor
        tau : float
            Diffusion time
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
        """Compute tissue response fucntions

        Shore coefficients are fitted for white matter, gray matter and
        cerebrospinal fluid separately. For white matter gradient tables are
        rotated such that the first eigenvector of the diffusion tensor would
        be aligned to the z-axis. Shore coefficients for each tissue are
        averaged and compressed by using only the z-rotational part of the
        shore coefficients. The averaged and compressed coefficients are
        returned in a ShoreMultiTissueResponse object.

        Unfortunately threads spawned by numpy interfere with multiprocessing
        used here. For optimal performance you need to set the environment
        variable OMP_NUM_THREADS to 1. You can do this from within a
        python script by inserting the following code before importing numpy:
        ```os.environ["OMP_NUM_THREADS"] = "1"```

        Parameters
        ----------
        data : ``SpatialImage``
            Diffusion weighted data
        dti_vecs : ``SpatialImage``
            First eigenvectors of precalculated diffusion tensors
        wm_mask : ``SpatialImage``
            White matter mask (0 or 1)
        gm_mask : ``SpatialImage``
            Gray matter mask (0 or 1)
        csf_mask : ``SpatialImage``
            Cerebrospinal fluid mask (0 or 1)
        verbose : bool
            Set to True for verbose output including a progress bar
        cpus : int
            Number of cpu workers to use for multiprocessing
        Returns
        -------
        ShoreMultiTissueResponse
            Object holding fitted response functions

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
                                       self.tau).fit(wm_voxels, vecs=wm_vecs,
                                                     verbose=verbose,
                                                     cpus=cpus,
                                                     desc='WM response').coeffs
        wmshore_coeff = self.shore_accumulate(wmshore_coeffs)
        signal_wm = self.shore_compress(wmshore_coeff)

        # Calculate gm response
        gm_voxels = data.get_data()[gm_mask.get_data() == 1]
        gmshore_coeffs = bd.ShoreModel(self.gtab, self.order, self.zeta,
                                       self.tau).fit(gm_voxels,
                                                     verbose=verbose,
                                                     cpus=cpus,
                                                     desc='GM response').coeffs
        gmshore_coeff = self.shore_accumulate(gmshore_coeffs)
        signal_gm = self.shore_compress(gmshore_coeff)

        # Calculate csf response
        csf_voxels = data.get_data()[csf_mask.get_data() == 1]
        csfshore_coeffs = bd.ShoreModel(self.gtab, self.order, self.zeta,
                                        self.tau).fit(csf_voxels,
                                                      verbose=verbose,
                                                      cpus=cpus,
                                                      desc='CSF response').coeffs
        csfshore_coeff = self.shore_accumulate(csfshore_coeffs)
        signal_csf = self.shore_compress(csfshore_coeff)

        return ShoreMultiTissueResponse(self,
                                        [signal_csf, signal_gm, signal_wm])

    def shore_accumulate(self, shore_coeff):
        """ Average over array of shore coefficients.

        This method is used to determine the average response of a specific
        tissue.

        Parameters
        ----------
        shore_coeff : ndarray (..., n)
            N-dimensional array of per voxel shore coefficients

        Returns
        -------
        ndarray
            Averaged shore coefficients

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

    def shore_compress(self, coeffs):
        """ Extract the z-rotational part of the shore coefficients.

        An axial symetric response function aligned to the z-axis can be
        described fully using only the z-rotational part of the shore
        coefficients.

        Parameters
        ----------
        coeffs : ndarray (n)
            N-dimensional array holding shore coefficients of a single model

        Returns
        -------
        ndarray
            z-rotational part of the given shore coefficients

        """
        r = np.zeros(shore.get_kernel_size(self.order, self.order))
        counter = 0
        ccounter = 0
        for l in range(0, self.order + 1, 2):
            for n in range((self.order - l) // 2 + 1):
                r[ccounter] = coeffs[counter + l]
                counter += 2 * l + 1
                ccounter += 1
        return r


class ShoreMultiTissueResponse(object):

    def __init__(self, model, shore_coef, kernel="rank1"):
        """

        Parameters
        ----------
        model
        shore_coef
        kernel
        """
        self.model = model
        self.gtab = model.gtab
        self.order = model.order
        self.zeta = model.zeta
        self.tau = model.tau

        self.signal_csf = shore_coef[0]
        self.signal_gm = shore_coef[1]
        self.signal_wm = shore_coef[2]

        # The deconvolution kernels are computed in set_kernel
        self.kernel_type = kernel
        self.kernel_csf = None
        self.kernel_gm = None
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

        Parameters
        ----------
        filepath : str
            Path to saved ShoreMultiTissueResponse object

        Returns
        -------
        ShoreMultiTissueResponse
            Object which holds response functions for white matter, gray matter
            and CSF
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = ShoreMultiTissueResponseEstimator(gtab, response['order'],
                                                  response['zeta'],
                                                  response['tau'])

        return cls(model, (response['csf'], response['gm'], response['wm']))

    def save(self, filepath):
        """ Save the ShoreMultiTissueResponse object to a file

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

        np.savez(filepath, csf=self.signal_csf, gm=self.signal_gm,
                 wm=self.signal_wm, zeta=self.zeta, tau=self.tau,
                 order=self.order, bvals=self.gtab.bvals,
                 bvecs=self.gtab.bvecs)

    def _convolve_helper(self, fodf_volfracs):
        """

        Parameters
        ----------
        fodf_volfracs

        Returns
        -------

        """
        fODF = fodf_volfracs[0]
        vol_fraction = fodf_volfracs[1]
        conv_matrix = self.shore_convolution_matrix()

        x = np.append(esh.sym_to_esh(fODF), vol_fraction)
        signal = np.dot(conv_matrix, x)

        return signal

    def convolve(self, fODFs, vol_fractions=None, S0=None, kernel="rank1",
                 verbose=False, cpus=None, desc=""):
        """ Convolve the Shore Fit with several fODFs.

        The multiprocessing for this function scales along the number of fODFs.
        For a small number of fODFs it makes sense to specify cpus=1 to avoid
        the overhead of spawning multiple processes.

        Parameters
        ----------
        fODFs
        vol_fractions
        S0
        kernel
        verbose
        cpus
        desc

        Returns
        -------

        """
        if self.kernel_type != kernel:
            self.set_kernel(kernel)

        if vol_fractions is None:
            vol_fractions = np.array([[0, 0]] * np.prod(fODFs.shape[:-1]))

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(fODFs.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if cpus == 1:
            signals = list(tqdm(map(self._convolve_helper,
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
        """ Deconvolve DWI data with multiple tissue responses [2]_.

        Parameters
        ----------
        data : ``SpatialImage``
            Diffusion weighted imaging data
        pos : str
            Constraint for the deconvolution (hpsd, nonneg or none)
        mask : ``SpatialImage``
            Mask specifying for which voxels to compute the fodf
        kernel : str
            Kernel to be used for deconvolution ('rank1' or 'delta')
        verbose : bool
            Set to true for verbose output and a progress bar
        cpus : int
            Number of cpu workers to be used for the calculations. (if None
            use value from os.cpu_count())

        Returns
        -------
        tuple
            fodfs, wm volume fraction, gm volume fraction and csf volume fraction

        References
        ----------
        .. [2] M. Ankele, L. Lim, S. Groeschel and T. Schultz; "Versatile,
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
        conv_mat = self.shore_convolution_matrix(kernel)
        with np.errstate(divide='ignore', invalid='ignore'):
            cond_number = la.cond(conv_mat)
            # For the proposed method of kernel rank1 and order 4 the condition
            # number of the matrix should not be larger than 1000 otherwise
            # show a warnig
            if kernel == "rank1" and self.order == 4 and cond_number > 1000:
                logging.warning("For kernel=rank1 and order=4 the condition"
                                "number of the convolution matrix should be "
                                "smaller than 1000. The condition number is: "
                                "{:.3f}".format(cond_number))
            else:
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
                result = list(tqdm(map(partial(func, conv_matrix=conv_mat),
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

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

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

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

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

        Parameters
        ----------
        data
        conv_matrix

        Returns
        -------

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
        G[:NC, :NN] = -esh.matrix(self.order, LOTS_OF_DIRECTIONS)

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

        Parameters
        ----------
        kernel

        Returns
        -------

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


def fa_guided_mask(tissue_mask, frac_aniso, brainmask=None,
                   tissue_threshold=0.95, fa_lower_thresh=-10,
                   fa_upper_thresh=np.inf):
    """ Create fractional anisotropy guided tissue mask

    Use precalculated fractional anisotropy values to create tissue masks. This
    function can be used to create a mask for white matter voxels which are
    likely to contain only a single fiber by feeding it a white matter mask and
    precalculated fractional anisotropy values.

    Parameters
    ----------
    tissue_mask : ``SpatialImage``
        Voxel fractions for a specific tissues (e.g. white matter)
    frac_aniso : ``SpatialImage``
        Precalculated fractional anisotropy values
    brainmask : ``SpatialImage``
        Mask which sepeartes the complete brain from the background
    tissue_threshold : float
        Float between 0 and 1 applied to the tissue_mask
    fa_lower_thresh : float
        The minimum fractional anisotropy value for a voxel to be considered
    fa_upper_thresh : float
        The maximum fractional anisotropy value for a voxel to be considered

    Returns
    -------
    ``SpatialImage``
        Object holding the calculated mask

    """
    if fa_lower_thresh == -10 and fa_upper_thresh == np.inf:
        msg = "Specify either 'fa_lower_thresh' or 'fa_upper_thresh'"
        raise ValueError(msg)

    # Load DTI fa map
    fa = frac_aniso.get_data()

    # Load DTI mask if available
    if brainmask is None:
        brainmask = np.ones(fa.shape)
    else:
        brainmask = brainmask.get_data()

    # Create new tissue mask
    tissue = tissue_mask.get_data()
    tissue_by_lower_fa = np.logical_and(tissue > tissue_threshold,
                                        fa_lower_thresh < fa)

    tissue_by_fa = np.logical_and(tissue_by_lower_fa, fa < fa_upper_thresh)
    fa_mask = np.logical_and(brainmask, tissue_by_fa).astype('int')

    mask_img = nib.Nifti1Image(fa_mask, tissue_mask.affine)

    return mask_img
