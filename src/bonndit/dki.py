import logging
import math

import cvxopt
import mpmath as mpm
import numpy as np
import numpy.linalg as la

from bonndit.base import ReconstModel, ReconstFit
from bonndit.multivoxel import MultiVoxel, MultiVoxelFitter


class DkiModel(ReconstModel):

    def __init__(self, gtab, constraint=True):
        """

        Parameters
        ----------
        gtab
        constraint
        """
        super().__init__(gtab)

        if self.gtab.b0_threshold < min(self.gtab.bvals):
            msg = "The specified b0 threshold is {}. The minimum b-value in " \
                  "the gradient table is {}. Please specify an appropriate " \
                  "b0 threshold. You can not set the attribute for an " \
                  "existing GradientTable but you need to create a new " \
                  "GradientTable with the b0_threshold " \
                  "parameter.".format(self.gtab.b0_threshold,
                                      min(self.gtab.bvals))
            raise ValueError(msg)



        self.dki_matrix = self.get_dki_matrix()
        cond_number = np.linalg.cond(self.dki_matrix)
        if cond_number > 1e6:
            logging.error('Refusing to create DkiModel. '
                          'Condition number of DKI matrix is to high ({}). '
                          'Are you trying to create a DkiModel from '
                          'single-shell data?'.format(cond_number))
            raise ValueError('Condition Number to high.')
        else:
            logging.info('Condition number of A: {}'.format(cond_number))

        # Let the user choose
        self.constraint = constraint
        if self.constraint:
            self.constraint_matrix = self.c_matrix()
        else:
            self.constraint_matrix = None

        # These parameters are saved for reinitalization
        self._params_dict = {'bvals': gtab.bvals, 'bvecs': gtab.bvecs,
                              'constraint': constraint}

    def _fit_helper(self, data, index, **kwargs):
        """

        Parameters
        ----------
        data
        kwargs

        Returns
        -------

        """
        solver = {False: self._solve, True: self._solve_c}

        data = data.astype(float)
        try:
            func = solver[self.constraint]
            coeffs = func(data, index, **kwargs)
        except KeyError:
            raise ValueError(('"{}" is not supported as a constraint, please' +
                              ' choose from [True, False]').format(
                self.constraint))

        if coeffs is None:
            return None
        else:
            return DkiFit(coeffs)

    def fit(self, data, mask=None, cpus=1, verbose=False, desc='', **kwargs):
        """

        Parameters
        ----------
        data
        mask
        kwargs

        Returns
        -------

        """
        # specify data which different for every voxel
        per_voxel_data = {}
        return MultiVoxelFitter(self, cpus=cpus, verbose=verbose,
                                desc=desc).fit(self._fit_helper, data,
                                               per_voxel_data, mask)

    def _solve(self, data, index, **kwargs):
        """

        Parameters
        ----------
        data
        kwargs

        Returns
        -------
        ndarray (21)
            Fitted kurtosis tensor

        """
        dki_tensor = np.zeros(22)
        data = data[~self.gtab.b0s_mask]
        dki_tensor[0] = 1
        dki_tensor[1:] = la.lstsq(self.dki_matrix, data, rcond=None)[0]
        return dki_tensor

    def _solve_c(self, data, index, **kwargs):
        """

        Parameters
        ----------
        data
        kwargs

        Returns
        -------
        ndarray (21)
            Fitted kurtosis tensor

        """
        dki_tensor = np.zeros(22)

        bvals = self.gtab.bvals[~self.gtab.b0s_mask] / 1000
        n_grads = len(bvals)

        d = np.zeros((n_grads * 2 + 9, 1))
        # impose minimum diffusivity
        d[2 * n_grads] = -0.1
        d[2 * n_grads + 4] = -0.1
        d[2 * n_grads + 8] = -0.1
        dims = {'l': 2 * n_grads, 'q': [], 's': [3]}

        # set up QP problem from normal equations
        cvxopt.solvers.options['show_progress'] = False
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(self.dki_matrix.T,
                                                      self.dki_matrix)))

        G = cvxopt.matrix(np.ascontiguousarray(self.constraint_matrix))
        h = cvxopt.matrix(np.ascontiguousarray(d))
        S0 = np.mean(data[self.gtab.b0s_mask])
        if S0 <= 0:
            logging.info('The average b0 measurement is {} in '
                         'voxel {}. DKI tensor is set to 0'
                         ''.format(S0, index))
            return None

        S = data[~self.gtab.b0s_mask]
        S[S <= 1e-10] = 1e-10  # clamp negative values
        S = np.log(S / S0)
        q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(self.dki_matrix.T,
                                                           S)))
        sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
        if sol['status'] != 'optimal':
            logging.warning('First-pass optimization unsuccessful.')
        c = np.array(sol['x'])[:, 0]
        dki_tensor[0] = 1
        dki_tensor[1:7] = c[:6]
        # divide out d-bar-square to get kurtosis tensor
        Dbar = (c[0] + c[3] + c[5]) / 3.0
        dki_tensor[7:] = c[6:] / Dbar ** 2

        return dki_tensor

    def c_matrix(self):
        """

        Returns
        -------

        """
        bvecs = self.gtab.bvecs[~self.gtab.b0s_mask, :]
        bvals = self.gtab.bvals[~self.gtab.b0s_mask] / 1000

        max_bval = np.max(bvals)

        n_grads = len(bvals)
        C = np.zeros((n_grads * 2 + 9, 21))
        for i in range(n_grads):
            # orthant constraints go first: min kurtosis
            C[i, 6] = -bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0]
            C[i, 7] = -4 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 1]
            C[i, 8] = -4 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 2]
            C[i, 9] = -6 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 1]
            C[i, 10] = -12 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 1] * bvecs[
                i, 2]
            C[i, 11] = -6 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 2] * bvecs[
                i, 2]
            C[i, 12] = -4 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 1]
            C[i, 13] = -12 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 2]
            C[i, 14] = -12 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 2] * bvecs[
                i, 2]
            C[i, 15] = -4 * bvecs[i, 0] * bvecs[i, 2] * bvecs[i, 2] * bvecs[
                i, 2]
            C[i, 16] = -bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 1]
            C[i, 17] = -4 * bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 2]
            C[i, 18] = -6 * bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 2] * bvecs[
                i, 2]
            C[i, 19] = -4 * bvecs[i, 1] * bvecs[i, 2] * bvecs[i, 2] * bvecs[
                i, 2]
            C[i, 20] = -bvecs[i, 2] * bvecs[i, 2] * bvecs[i, 2] * bvecs[i, 2]
            # max kurtosis constraints as in Tabesh et al.
            C[n_grads + i, 0] = -3.0 / max_bval * bvecs[i, 0] * bvecs[i, 0]
            C[n_grads + i, 1] = -3.0 / max_bval * 2 * bvecs[i, 0] * bvecs[i, 1]
            C[n_grads + i, 2] = -3.0 / max_bval * 2 * bvecs[i, 0] * bvecs[i, 2]
            C[n_grads + i, 3] = -3.0 / max_bval * bvecs[i, 1] * bvecs[i, 1]
            C[n_grads + i, 4] = -3.0 / max_bval * 2 * bvecs[i, 1] * bvecs[i, 2]
            C[n_grads + i, 5] = -3.0 / max_bval * bvecs[i, 2] * bvecs[i, 2]
            C[n_grads + i, 6] = bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0]
            C[n_grads + i, 7] = 4 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[
                i, 1]
            C[n_grads + i, 8] = 4 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 0] * bvecs[
                i, 2]
            C[n_grads + i, 9] = 6 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 1] * bvecs[
                i, 1]
            C[n_grads + i, 10] = 12 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 1] * \
                            bvecs[i, 2]
            C[n_grads + i, 11] = 6 * bvecs[i, 0] * bvecs[i, 0] * bvecs[i, 2] * bvecs[
                i, 2]
            C[n_grads + i, 12] = 4 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 1]
            C[n_grads + i, 13] = 12 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 1] * \
                            bvecs[i, 2]
            C[n_grads + i, 14] = 12 * bvecs[i, 0] * bvecs[i, 1] * bvecs[i, 2] * \
                            bvecs[i, 2]
            C[n_grads + i, 15] = 4 * bvecs[i, 0] * bvecs[i, 2] * bvecs[i, 2] * bvecs[
                i, 2]
            C[n_grads + i, 16] = bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 1]
            C[n_grads + i, 17] = 4 * bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 1] * bvecs[
                i, 2]
            C[n_grads + i, 18] = 6 * bvecs[i, 1] * bvecs[i, 1] * bvecs[i, 2] * bvecs[
                i, 2]
            C[n_grads + i, 19] = 4 * bvecs[i, 1] * bvecs[i, 2] * bvecs[i, 2] * bvecs[
                i, 2]
            C[n_grads + i, 20] = bvecs[i, 2] * bvecs[i, 2] * bvecs[i, 2] * bvecs[
                i, 2]
        # min diffusivity - now a proper psd constraint, independent of directions
        # just need to give it the negative diffusion tensor in column major order
        C[2 * n_grads, 0] = -1.0
        C[2 * n_grads + 1, 1] = -1.0
        C[2 * n_grads + 2, 2] = -1.0
        C[2 * n_grads + 3, 1] = -1.0
        C[2 * n_grads + 4, 3] = -1.0
        C[2 * n_grads + 5, 4] = -1.0
        C[2 * n_grads + 6, 2] = -1.0
        C[2 * n_grads + 7, 4] = -1.0
        C[2 * n_grads + 8, 5] = -1.0

        return C

    def get_dki_matrix(self):
        """ Build Diffusion Kurtosis Matrix

        This matrix masp DKI params to the log signal ratio.
        Returns
        -------

        """

        bvecs = self.gtab.bvecs[~self.gtab.b0s_mask, :]
        bvals = self.gtab.bvals[~self.gtab.b0s_mask] / 1000
        A = np.zeros((len(bvals), 21))
        for i in range(len(bvals)):
            # note: the order at this point deviates from Tabesh et al.
            # so as to agree with teem conventions
            A[i, 0] = -bvals[i] * bvecs[i, 0] * bvecs[i, 0]
            A[i, 1] = -bvals[i] * 2 * bvecs[i, 0] * bvecs[i, 1]
            A[i, 2] = -bvals[i] * 2 * bvecs[i, 0] * bvecs[i, 2]
            A[i, 3] = -bvals[i] * bvecs[i, 1] * bvecs[i, 1]
            A[i, 4] = -bvals[i] * 2 * bvecs[i, 1] * bvecs[i, 2]
            A[i, 5] = -bvals[i] * bvecs[i, 2] * bvecs[i, 2]
            A[i, 6] = bvals[i] ** 2 / 6.0 * bvecs[i, 0] * bvecs[i, 0] * \
                      bvecs[i, 0] * bvecs[i, 0]
            A[i, 7] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 0] * bvecs[i, 0] * \
                      bvecs[i, 0] * bvecs[i, 1]
            A[i, 8] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 0] * bvecs[i, 0] * \
                      bvecs[i, 0] * bvecs[i, 2]
            A[i, 9] = bvals[i] ** 2 / 6.0 * 6 * bvecs[i, 0] * bvecs[i, 0] * \
                      bvecs[i, 1] * bvecs[i, 1]
            A[i, 10] = bvals[i] ** 2 / 6.0 * 12 * bvecs[i, 0] * bvecs[
                i, 0] * bvecs[i, 1] * bvecs[i, 2]
            A[i, 11] = bvals[i] ** 2 / 6.0 * 6 * bvecs[i, 0] * bvecs[i, 0] * \
                       bvecs[i, 2] * bvecs[i, 2]
            A[i, 12] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 0] * bvecs[i, 1] * \
                       bvecs[i, 1] * bvecs[i, 1]
            A[i, 13] = bvals[i] ** 2 / 6.0 * 12 * bvecs[i, 0] * bvecs[
                i, 1] * bvecs[i, 1] * bvecs[i, 2]
            A[i, 14] = bvals[i] ** 2 / 6.0 * 12 * bvecs[i, 0] * bvecs[
                i, 1] * bvecs[i, 2] * bvecs[i, 2]
            A[i, 15] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 0] * bvecs[i, 2] * \
                       bvecs[i, 2] * bvecs[i, 2]
            A[i, 16] = bvals[i] ** 2 / 6.0 * bvecs[i, 1] * bvecs[i, 1] * \
                       bvecs[i, 1] * bvecs[i, 1]
            A[i, 17] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 1] * bvecs[i, 1] * \
                       bvecs[i, 1] * bvecs[i, 2]
            A[i, 18] = bvals[i] ** 2 / 6.0 * 6 * bvecs[i, 1] * bvecs[i, 1] * \
                       bvecs[i, 2] * bvecs[i, 2]
            A[i, 19] = bvals[i] ** 2 / 6.0 * 4 * bvecs[i, 1] * bvecs[i, 2] * \
                       bvecs[i, 2] * bvecs[i, 2]
            A[i, 20] = bvals[i] ** 2 / 6.0 * bvecs[i, 2] * bvecs[i, 2] * \
                       bvecs[i, 2] * bvecs[i, 2]

        return A


class DkiFit(ReconstFit):
    def __init__(self, coeffs):
        """ Compute kurtosis measures for a fitted kurtosis model.

        Parameters
        ----------
        coeffs : ndarray
            Kurtosis parameters
        """
        super().__init__(coeffs)
        self.dti_tensor = np.array(
            [[self.coeffs[1], self.coeffs[2], self.coeffs[3]],
             [self.coeffs[2], self.coeffs[4], self.coeffs[5]],
             [self.coeffs[3], self.coeffs[5], self.coeffs[6]]])

        # evals are in *ascending* order
        (self.evals, self.evecs) = np.linalg.eigh(self.dti_tensor)
        # clamp evals to avoid numerical trouble
        self.evals[self.evals < 1e-10] = 1e-10

        self.rot_kurtosis_tensor = rotT4Sym(self.coeffs[7:], self.evecs)

        self._diffusivity_axial = None
        self._diffusivity_radial = None
        self._diffusivity_mean = None

        self._kurtosis_axial = None
        self._kurtosis_radial = None
        self._kurtosis_mean = None

        self._fractional_anisotropy = None

        self._kappa_axial = None
        self._kappa_radial = None
        self._kappa_diamond = None

    @classmethod
    def load(cls, filepath):
        """

        Parameters
        ----------
        filepath

        Returns
        -------

        """
        return MultiVoxel.load(filepath, model_class=DkiModel,
                               fit_class=cls)

    def predict(self, gtab):
        """

        Parameters
        ----------
        gtab

        Returns
        -------

        """
        super().predict(gtab)

    @property
    def diffusivity_axial(self):
        if self._diffusivity_axial is None:
            self._diffusivity_axial = self.evals[2]

        return self._diffusivity_axial

    @property
    def diffusivity_radial(self):
        if self._diffusivity_radial is None:
            self._diffusivity_radial = 0.5 * (self.evals[0] +
                                              self.evals[1])

        return self._diffusivity_radial

    @property
    def diffusivity_mean(self):
        if self._diffusivity_mean is None:
            self._diffusivity_mean = np.mean(self.evals)

        return self._diffusivity_mean


    # Fractional anisotropy
    @property
    def fractional_anisotropy(self):
        if self._fractional_anisotropy is None:
            self._fractional_anisotropy = math.sqrt(
                ((self.evals[0] - self.evals[1]) ** 2
                 + (self.evals[1] - self.evals[2]) ** 2
                 + (self.evals[2] - self.evals[0]) ** 2)
                / (self.evals[0] ** 2
                   + self.evals[1] ** 2
                   + self.evals[2] ** 2) / 2)

        return self._fractional_anisotropy


    # Kurtosis properties
    @property
    def kurtosis_axial(self):
        if self._kurtosis_axial is None:
            self._kurtosis_axial = axial_kurtosis(self.evals,
                                                  self.rot_kurtosis_tensor)

        return self._kurtosis_axial

    @property
    def kurtosis_radial(self):
        if self._kurtosis_radial is None:
            self._kurtosis_radial = radial_kurtosis(self.evals,
                                                    self.rot_kurtosis_tensor)

        return self._kurtosis_radial

    @property
    def kurtosis_mean(self):
        if self._kurtosis_mean is None:
            self._kurtosis_mean = mean_kurtosis(self.evals,
                                                self.rot_kurtosis_tensor)

        return self._kurtosis_mean

    # Kappa properties
    # kurtosis related parameters for a cylindrically constrained kurtosis
    # M. Ankele, T. Schultz; "Quantifying Microstructure in Fiber Crossings
    # with Diffusional Kurtosis"; doi: 10.1007/978-3-319-24553-9_19
    @property
    def kappa_axial(self):
        if self._kappa_axial is None:
            self._kappa_axial = axial_kappa(self.diffusivity_mean,
                                            self.rot_kurtosis_tensor)

        return self._kappa_axial

    @property
    def kappa_radial(self):
        if self._kappa_radial is None:
            self._kappa_radial = radial_kappa(self.diffusivity_mean,
                                              self.rot_kurtosis_tensor)

        return self._kappa_radial

    @property
    def kappa_diamond(self):
        if self._kappa_diamond is None:
            self._kappa_diamond = diamond_kappa(self.diffusivity_mean,
                                                self.rot_kurtosis_tensor)

        return self._kappa_diamond


def radial_kappa(lambda_mean, kurtosis_tensor):
    """

    Parameters
    ----------
    lambda_mean
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        return lambda_mean ** 2 * (kurtosis_tensor[0]
                               + kurtosis_tensor[10]
                               + 3 * kurtosis_tensor[3]) / 3


def axial_kappa(lambda_mean, kurtosis_tensor):
    """

    Parameters
    ----------
    lambda_mean
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        return lambda_mean ** 2 * kurtosis_tensor[14]


def diamond_kappa(lambda_mean, kurtosis_tensor):
    """

    Parameters
    ----------
    lambda_mean
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        return 6 * lambda_mean ** 2 * (kurtosis_tensor[5]
                                   + kurtosis_tensor[12]) / 2


def radial_kurtosis(evals, kurtosis_tensor):
    """

    Parameters
    ----------
    evals
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        return _G1(evals[2], evals[1], evals[0]) * kurtosis_tensor[10] \
               + _G1(evals[2], evals[0], evals[1]) * kurtosis_tensor[0] \
               + _G2(evals[2], evals[1], evals[0]) * kurtosis_tensor[3]


def axial_kurtosis(evals, kurtosis_tensor):
    """

    Parameters
    ----------
    evals
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        return ((evals[0] + evals[1] + evals[2]) ** 2
            / (9 * evals[2] ** 2)) * kurtosis_tensor[14]


def mean_kurtosis(evals, kurtosis_tensor):
    """

    Parameters
    ----------
    evals
    kurtosis_tensor

    Returns
    -------

    """
    if sum(kurtosis_tensor) == 0:
        return None
    else:
        r = _F1(evals[0], evals[1], evals[2]) * kurtosis_tensor[0]
        r += _F1(evals[1], evals[2], evals[0]) * kurtosis_tensor[10]
        r += _F1(evals[2], evals[1], evals[0]) * kurtosis_tensor[14]
        r += _F2(evals[0], evals[1], evals[2]) * kurtosis_tensor[12]
        r += _F2(evals[1], evals[2], evals[0]) * kurtosis_tensor[5]
        r += _F2(evals[2], evals[1], evals[0]) * kurtosis_tensor[3]
        return r


def _alpha(x):
    if x > 0:
        return math.atanh(math.sqrt(x)) / math.sqrt(x)
    else:
        return math.atan(math.sqrt(-x)) / math.sqrt(-x)


def _H(a, c):
    if a == c:
        return 1.0 / 15.0
    return (a + 2 * c) ** 2 / (144 * c * c * (a - c) ** 2) * (
        c * (a + 2 * c) + a * (a - 4 * c) * _alpha(1 - a / c))


def _F1(a, b, c):
    if a == b:
        return 3 * _H(c, a)
    if a == c:
        return 3 * _H(b, a)
    return (a + b + c) ** 2 / (18 * (a - b) * (a - c)) * (
        math.sqrt(b * c) / a * float(mpm.elliprf(a / b, a / c, 1)) + (
        3 * a ** 2 - a * b - a * c - b * c) / (
            3 * a * math.sqrt(b * c)) * float(
        mpm.elliprd(a / b, a / c, 1)) - 1)


def _F2(a, b, c):
    if b == c:
        return 6 * _H(a, c)
    return (a + b + c) ** 2 / (3 * (b - c) ** 2) * (
        (b + c) / (math.sqrt(b * c)) * float(mpm.elliprf(a / b, a / c, 1)) + (
        2 * a - b - c) / (
            3 * math.sqrt(b * c)) * float(mpm.elliprd(a / b, a / c, 1)) - 2)


def _G1(a, b, c):
    if b == c:
        return (a + 2 * b) ** 2 / (24 * b * b)
    return (a + b + c) ** 2 / (18 * b * (b - c) ** 2) * (
        2 * b + c * (c - 3 * b) / (math.sqrt(b * c)))


def _G2(a, b, c):
    if b == c:
        return (a + 2 * b) ** 2 / (12 * b * b)
    return (a + b + c) ** 2 / (3 * (b - c) ** 2) * (
        (b + c) / (math.sqrt(b * c)) - 2)


ix4 = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 1], [0, 0, 1, 2],
       [0, 0, 2, 2], [0, 1, 1, 1], [0, 1, 1, 2],
       [0, 1, 2, 2], [0, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2],
       [1, 2, 2, 2], [2, 2, 2, 2]]

invix4 = np.zeros((3, 3, 3, 3), dtype=np.int)
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                s = [i, j, k, l]
                s.sort()
                invix4[i, j, k, l] = ix4.index(s)


# L are the eigenvectors such that L[:,i] is ith normalized eigenvector
def rotT4Sym(W, L):
    """

    Parameters
    ----------
    W
    L

    Returns
    -------

    """
    # build and apply rotation matrix
    rotmat = np.zeros((15, 15))
    for idx in range(15):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    for ll in range(3):
                        rotmat[idx, invix4[ii, jj, kk, ll]] += L[ii, ix4[idx][
                            0]] * L[jj, ix4[idx][1]] * L[
                                                                   kk,
                                                                   ix4[idx][
                                                                       2]] * L[
                                                                   ll,
                                                                   ix4[idx][3]]
    return np.dot(rotmat, W)
