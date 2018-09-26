import logging

import cvxopt
import numpy as np
import numpy.linalg as la

from bonndit.base import ReconstModel, ReconstFit, multi_voxel_method


class DkiModel(ReconstModel):

    def __init__(self, gtab, constraints=False):
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
        self.constraints = constraints
        if self.constraints:
            self.constraint_matrix = self.c_matrix()
        else:
            self.constraint_matrix = None

    @multi_voxel_method
    def fit(self, data, constraint=False):
        """

        :param data:
        :return:
        """

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        solver = {False: self._solve, True: self._solve_c}

        try:
            func = solver[constraint]
            coeffs = func(data)
        except KeyError:
            raise ValueError(('"{}" is not supported as a constraint, please' +
                              ' choose from [True, False]').format(constraint))

        return DkiFit(self, coeffs)

    def _solve(self, data):
        """

        :param data:
        :return:
        """
        dki_tensors = la.lstsq(self.dki_matrix, data, rcond=None)[0]
        return dki_tensors

    def _solve_c(self, data):
        """

        :param data:
        :return:
        """
        dki_tensor = np.zeros(22)

        bvals = self.gtab.bvals[~self.gtab.b0s_mask]
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

        S0 = np.mean(data[~self.gtab.b0s_mask])
        if S0 <= 0:
            logging.warning('The average b0 measurement is 0 or smaller.')
            raise ValueError()

        S = data[self.gtab.b0s_mask]
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
        bvals = self.gtab.bvals[~self.gtab.b0s_mask]

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
        bvals = self.gtab.bvals[~self.gtab.b0s_mask]
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
    def __init__(self, model, coeffs):
        pass

    def predict(self, gtab):
        super().predict(gtab)
