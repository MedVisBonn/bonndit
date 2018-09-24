class DkiModel(object):

    def __init__(self, gtab, constraints=None):
        self.gtab = gtab
        self.dki_matrix = self.dki_matrix()

        # Let the user choose
        self.constraints = constraints
        self.constraint_matrix = self.c_matrix()

    def _fit_helper(self, data):
        """

        :param data:
        :return:
        """
        d = np.zeros((nk * 2 + 9, 1))
        # impose minimum diffusivity
        d[2 * nk] = -0.1
        d[2 * nk + 4] = -0.1
        d[2 * nk + 8] = -0.1
        dims = {'l': 2 * nk, 'q': [], 's': [3]}

        # set up QP problem from normal equations
        cvxopt.solvers.options['show_progress'] = False
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(A.T, A)))
        G = cvxopt.matrix(np.ascontiguousarray(C))
        h = cvxopt.matrix(np.ascontiguousarray(d))




    def fit(self, data, verbose=False, cpus=1, desc=''):
        """

        :param data:
        :param verbose:
        :param cpus:
        :param desc:
        :return:
        """
        cond_number = np.linalg.cond(self.dki_matrix())
        if  cond_number > 1e6:
            logging.error('Refusing to fit DKI with condition number {}. Are '
                            'you trying to estimate kurtosis from single-shell '
                            'data?'.format(cond_number))
            raise InputError('Condition Number to high.')
        else:
            logging.info('Condition number of A: {}'.format(cond_number))

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3 or cpus == 1:
            dki_coeff = list(tqdm(imap(self._fit_helper,data),
                                    total=np.prod(data.shape[:-1]),
                                    disable=not verbose,
                                    desc=desc))
        else:
            with mp.Pool(cpus) as p:
                dki_coeff = list(tqdm(p.imap(self._fit_helper, data,
                                               chunksize),
                                        total=np.prod(data.shape[:-1]),
                                        disable=not verbose,
                                        desc=desc))

        return DkiFit(model, dki_coeff)

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


    def dki_matrix(self):
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

        if np.linalg.cond(A) > 1e6:
            print('Refusing to fit DKI with condition number ',
                  np.linalg.cond(A))
            print(
                'Are you trying to estimate kurtosis from single-shell data?')
            sys.exit(1)
        elif args.verbose:
            print('Condition number of A: ', np.linalg.cond(A))

class DkiFit(object):
    def __init__(self):
        pass
