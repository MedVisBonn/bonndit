class DkiModel(object):

    def __init__(self, gtab):
        self.gtab = gtab

    def fit(self):
        pass

    def get_matrix_A(self):
        """ Build matrix A (maps DKI params to log signal ratio)

        :return:
        """

        grads = gtab[~gtab.b0s_mask]
        bvecs[bvals > bval_eps]
        dwibvals = bvals[bvals > bval_eps]
        bmax = np.max(dwibvals)
        nk = len(dwibvals)
        A = np.zeros((nk, 21))
        for i in range(nk):
            # note: the order at this point deviates from Tabesh et al.
            # so as to agree with teem conventions
            A[i, 0] = -dwibvals[i] * grads[i, 0] * grads[i, 0]
            A[i, 1] = -dwibvals[i] * 2 * grads[i, 0] * grads[i, 1]
            A[i, 2] = -dwibvals[i] * 2 * grads[i, 0] * grads[i, 2]
            A[i, 3] = -dwibvals[i] * grads[i, 1] * grads[i, 1]
            A[i, 4] = -dwibvals[i] * 2 * grads[i, 1] * grads[i, 2]
            A[i, 5] = -dwibvals[i] * grads[i, 2] * grads[i, 2]
            A[i, 6] = dwibvals[i] ** 2 / 6.0 * grads[i, 0] * grads[i, 0] * \
                      grads[i, 0] * grads[i, 0]
            A[i, 7] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 0] * \
                      grads[i, 0] * grads[i, 1]
            A[i, 8] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 0] * \
                      grads[i, 0] * grads[i, 2]
            A[i, 9] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 0] * grads[i, 0] * \
                      grads[i, 1] * grads[i, 1]
            A[i, 10] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[
                i, 0] * grads[i, 1] * grads[i, 2]
            A[i, 11] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 0] * grads[i, 0] * \
                       grads[i, 2] * grads[i, 2]
            A[i, 12] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 1] * \
                       grads[i, 1] * grads[i, 1]
            A[i, 13] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[
                i, 1] * grads[i, 1] * grads[i, 2]
            A[i, 14] = dwibvals[i] ** 2 / 6.0 * 12 * grads[i, 0] * grads[
                i, 1] * grads[i, 2] * grads[i, 2]
            A[i, 15] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 0] * grads[i, 2] * \
                       grads[i, 2] * grads[i, 2]
            A[i, 16] = dwibvals[i] ** 2 / 6.0 * grads[i, 1] * grads[i, 1] * \
                       grads[i, 1] * grads[i, 1]
            A[i, 17] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 1] * grads[i, 1] * \
                       grads[i, 1] * grads[i, 2]
            A[i, 18] = dwibvals[i] ** 2 / 6.0 * 6 * grads[i, 1] * grads[i, 1] * \
                       grads[i, 2] * grads[i, 2]
            A[i, 19] = dwibvals[i] ** 2 / 6.0 * 4 * grads[i, 1] * grads[i, 2] * \
                       grads[i, 2] * grads[i, 2]
            A[i, 20] = dwibvals[i] ** 2 / 6.0 * grads[i, 2] * grads[i, 2] * \
                       grads[i, 2] * grads[i, 2]

        if np.linalg.cond(A) > 1e6:
            print('Refusing to fit DKI with condition number ',
                  np.linalg.cond(A))
            print(
                'Are you trying to estimate kurtosis from single-shell data?')
            sys.exit(1)
        elif args.verbose:
            print('Condition number of A: ', np.linalg.cond(A))
