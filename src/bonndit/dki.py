import logging
import math

import cvxopt
import numpy as np
import numpy.linalg as la
from dipy.reconst.dki import Wrotate

from bonndit.base import ReconstModel, ReconstFit
from bonndit.multivoxel import MultiVoxel, MultiVoxelFitter


class DkiModel(ReconstModel):

    def __init__(self, gtab, constraint=False):
        self.gtab = gtab

        self.dki_matrix = self.get_dki_matrix()
        cond_number = np.linalg.cond(self.dki_matrix)
        if cond_number > 1e6:
            logging.error('Refusing to create DkiModel. '
                          'Condition number of DKI matrix is to bit ({}). '
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

        # Parameters in this dict are needed to reinitalize the model
        self._model_params = {'bvals': gtab.bvals, 'bvecs': gtab.bvecs,
                              'constraint': constraint}

    def _fit_helper(self, data):
        """

        :param data:
        :return:
        """
        solver = {False: self._solve, True: self._solve_c}

        data = data.astype(float)
        try:
            func = solver[self.constraint]
            coeffs = func(data)
        except KeyError:
            raise ValueError(('"{}" is not supported as a constraint, please' +
                              ' choose from [True, False]').format(
                self.constraint))

        return DkiFit(coeffs)

    def fit(self, data, mask=None, **kwargs):
        """

        :param data:
        :param mask:
        :param kwargs:
        :return:
        """
        per_voxel_data = {}
        return MultiVoxelFitter(self, **kwargs).fit(self._fit_helper, data,
                                                    per_voxel_data, mask)

    def _solve(self, data):
        """

        :param data:
        :return:
        """
        dki_tensor = np.zeros(22)
        data = data[~self.gtab.b0s_mask]
        dki_tensor[0] = 1
        dki_tensor[1:] = la.lstsq(self.dki_matrix, data, rcond=None)[0]
        return dki_tensor

    def _solve_c(self, data):
        """

        :param data:
        :return:
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
            logging.warning('The average b0 measurement is 0 or smaller.')
            raise ValueError()
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

        :return:
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
        (maps DKI params to log signal ratio)

        :return:
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
        self.coeffs = coeffs
        self.dti_tensor = np.array(
            [[self.coeffs[1], self.coeffs[2], self.coeffs[3]],
             [self.coeffs[2], self.coeffs[4], self.coeffs[5]],
             [self.coeffs[3], self.coeffs[5], self.coeffs[6]]])

        # lambdas are in *ascending* order
        (self.lambdas, self.evecs) = np.linalg.eigh(self.dti_tensor)
        # clamp lambdas to avoid numerical trouble
        self.lambdas[self.lambdas < 1e-10] = 1e-10

        self.rot_kurtosis_tensor = Wrotate(self.coeffs[7:], self.evecs)

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
        return MultiVoxel.load(filepath, model_class=DkiModel,
                               fit_class=cls)

    def predict(self, gtab):
        super().predict(gtab)

    # Diffusivity properties
    @property
    def diffusivity_axial(self):
        return self._diffusivity_axial

    @diffusivity_axial.getter
    def diffusivity_axial(self):
        if self._diffusivity_axial is None:
            self._diffusivity_axial = self.lambdas[2]
        else:
            return self._diffusivity_axial

    @property
    def diffusivity_radial(self):
        return self._diffusivity_axial

    @diffusivity_radial.getter
    def diffusivity_radial(self):
        if self._diffusivity_radial is None:
            self._diffusivity_radial = 0.5 * (self.lambdas[0] +
                                              self.lambdas[1])
        else:
            return self._diffusivity_radial

    @property
    def diffusivity_mean(self):
        return self._diffusivity_mean

    @diffusivity_mean.getter
    def diffusivity_mean(self):
        if self._diffusivity_mean is None:
            self._diffusivity_mean = np.mean(self.lambdas)
        else:
            return self._diffusivity_mean

    # Fractional anisotropy
    @property
    def fractional_anisotropy(self):
        return self._fractional_anisotropy

    @fractional_anisotropy.getter
    def fractional_anisotropy(self):
        if self._fractional_anisotropy is None:
            self._fractional_anisotropy = math.sqrt(
                ((self.lambdas[0] - self.lambdas[1]) ** 2
                 + (self.lambdas[1] - self.lambdas[2]) ** 2
                 + (self.lambdas[2] - self.lambdas[0]) ** 2)
                / (self.lambdas[0] ** 2
                   + self.lambdas[1] ** 2
                   + self.lambdas[2] ** 2) / 2)
        else:
            return self._fractional_anisotropy

    # Kurtosis properties
    @property
    def kurtosis_axial(self):
        return self._kurtosis_axial

    @kurtosis_axial.getter
    def kurtosis_axial(self):
        if self._kurtosis_axial is None:
            self._kurtosis_axial = axial_kurtosis(self.lambdas,
                                                  self.rot_kurtosis_tensor)
        else:
            return self._kurtosis_axial

    @property
    def kurtosis_radial(self):
        return self._kurtosis_axial

    @kurtosis_radial.getter
    def kurtosis_radial(self):
        if self._kurtosis_radial is None:
            self._kurtosis_radial = radial_kurtosis(self.lambdas,
                                                    self.rot_kurtosis_tensor)
        else:
            return self._kurtosis_radial

    @property
    def kurtosis_mean(self):
        return self._kurtosis_mean

    @kurtosis_mean.getter
    def kurtosis_mean(self):
        if self._kurtosis_mean is None:
            self._kurtosis_mean = mean_kurtosis(self.lambdas,
                                                self.rot_kurtosis_tensor)
        else:
            return self._kurtosis_mean

    # Kurtosis properties
    @property
    def kappa_axial(self):
        return self._kappa_axial

    @kappa_axial.getter
    def kappa_axial(self):
        if self._kappa_axial is None:
            self._kappa_axial = axial_kappa(self._diffusivity_mean,
                                            self.rot_kurtosis_tensor)
        else:
            return self._kappa_axial

    @property
    def kappa_radial(self):
        return self._kappa_axial

    @kappa_radial.getter
    def kappa_radial(self):
        if self._kappa_radial is None:
            self._kappa_radial = radial_kappa(self._diffusivity_mean,
                                              self.rot_kurtosis_tensor)
        else:
            return self._kappa_radial

    @property
    def kappa_diamond(self):
        return self._kappa_diamond

    @kappa_diamond.getter
    def kappa_diamond(self):
        if self._kappa_diamond is None:
            self._kappa_diamond = diamond_kappa(self._diffusivity_mean,
                                                self.rot_kurtosis_tensor)
        else:
            return self._kappa_diamond


def radial_kappa(lambda_mean, kurtosis_tensor):
    """

    :param lambda_mean:
    :param kurtosis_tensor:
    :return:
    """
    return lambda_mean ** 2 * (kurtosis_tensor[0]
                               + kurtosis_tensor[10]
                               + 3 * kurtosis_tensor[3]) / 3


def axial_kappa(lambda_mean, kurtosis_tensor):
    """

    :param lambda_mean:
    :param kurtosis_tensor:
    :return:
    """
    return lambda_mean ** 2 * kurtosis_tensor[14]


def diamond_kappa(lambda_mean, kurtosis_tensor):
    """

    :param lambda_mean:
    :param kurtosis_tensor:
    :return:
    """
    return 6 * lambda_mean ** 2 * (kurtosis_tensor[5]
                                   + kurtosis_tensor[12]) / 2


def radial_kurtosis(lambdas, kurtosis_tensor):
    """

    :param lambdas:
    :param kurtosis_tensor:
    :return:
    """
    pass


def axial_kurtosis(lambdas, kurtosis_tensor):
    """

    :param lambdas:
    :param kurtosis_tensor:
    :return:
    """
    pass


def mean_kurtosis(lambdas, kurtosis_tensor):
    """

    :param lambdas:
    :param kurtosis_tensor:
    :return:
    """
    pass
