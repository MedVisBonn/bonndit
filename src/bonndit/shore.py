from __future__ import division

import errno
import os
import pickle

import cvxopt
import numpy as np
import numpy.linalg as la
from bonndit.constants import LOTS_OF_DIRECTIONS
from bonndit.michi import shore, esh, tensor
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.reconst.shore import shore_matrix
from tqdm import tqdm


class ShoreModel:
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        self.gtab = gtab
        self.order = order
        self.zeta = zeta
        self.tau = tau

    def fit(self, data, wm_mask, gm_mask, csf_mask, dti_mask, dti_fa, dti_vecs, fawm=0.7, verbose=False):
        # Load DTI fa map
        fa = dti_fa.get_data()

        # Load DTI vecs
        vecs = dti_vecs.get_data()

        # Load DTI mask if available
        if dti_mask is None:
            NX, NY, NZ = fa.shape
            mask = np.ones((NX, NY, NZ))
        else:
            mask = dti_mask.get_data()

        # Create masks
        # CSF
        csf = csf_mask.get_data()
        mask_csf = np.logical_and(mask, np.logical_and(csf > 0.95, fa < 0.2)).astype('int')
        # GM
        gm = gm_mask.get_data()
        mask_gm = np.logical_and(mask, np.logical_and(gm > 0.95, fa < 0.2)).astype('int')
        # WM
        wm = wm_mask.get_data()
        mask_wm = np.logical_and(mask, np.logical_and(wm > 0.95, fa > float(fawm))).astype('int')

        # Load data
        data = data.get_data()

        # Do not show divide by zero warnings
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate csf response
        shore_coeff = self._get_response(data, mask_csf, verbose)
        signal_csf = self._shore_compress(shore_coeff)

        # Calculate gm response
        shore_coeff = self._get_response(data, mask_gm, verbose)
        signal_gm = self._shore_compress(shore_coeff)

        # Calculate wm response
        shore_coeff = self._get_response_reorient(data, mask_wm, vecs, verbose)
        signal_wm = self._shore_compress(shore_coeff)

        return ShoreFit(self, [signal_csf, signal_gm, signal_wm])

    def _shore_compress(self, s):
        """ "kernel": only use z-rotational part

        :param s:
        :return:
        """
        r = np.zeros(get_kernel_size(self.order))
        counter = 0
        ccounter = 0
        for l in range(0, self.order + 1, 2):
            for n in range(l, (self.order - l) // 2 + 1):
                r[ccounter] = s[counter + l]
                counter += 2 * l + 1
                ccounter += 1
        return r

    def _accumulate_shore(self, shore_coeff, mask):
        """Average over all shore coefficients
        """

        shore_accum = np.zeros(shore_get_size(self.order))
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
        return shore_accum / accum_count

    def _get_response(self, data, mask, verbose=False):
        """

        """
        shore_coeff = np.zeros(data.shape[:3] + (shore_get_size(self.order),))
        M = shore_matrix(self.order, self.zeta, self.gtab, self.tau)

        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:3]),
                      total=np.prod(data.shape[:3]),
                      disable=not verbose):
            if mask[i] == 0:
                continue

            r = la.lstsq(M, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return self._accumulate_shore(shore_coeff, mask)

    def _get_response_reorient(self, data, mask, vecs, verbose=False):
        """
        vecs: the first principal direction of diffusion for every voxel
        """
        shore_coeff = np.zeros(data.shape[:3] + (shore_get_size(self.order),))

        # Iterate over the data indices; show progress with tqdm
        for i in tqdm(np.ndindex(*data.shape[:3]),
                      total=np.prod(data.shape[:3]),
                      disable=not verbose):
            if mask[i] == 0:
                continue

            gtab2 = gtab_reorient(self.gtab, vecs[i])
            M = shore_matrix(self.order, self.zeta, gtab2, self.tau)
            r = la.lstsq(M, data[i], rcond=-1)
            shore_coeff[i] = r[0]

        return self._accumulate_shore(shore_coeff, mask)


class ShoreFit:
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
        with open(filepath, 'rb') as in_file:
            return pickle.load(in_file)

    def save(self, output):
        with open(output, 'wb') as out_file:
            pickle.dump(self, out_file, -1)

    @classmethod
    def old_load(cls, filepath):
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = ShoreModel(gtab, response['order'], response['zeta'], response['tau'])
        return cls(model, (response['csf'], response['gm'], response['wm']))

    def old_save(self, outdir):
        try:
            os.makedirs(outdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(outdir + 'response.npz', csf=self.signal_csf, gm=self.signal_gm, wm=self.signal_wm,
                 zeta=self.zeta, tau=self.tau, order=self.order, bvals=self.gtab.bvals, bvecs=self.gtab.bvecs)

    def fodf(self, img, pos='hpsd', verbose=False):
        """ Deconvolve the signal with the 3 response functions

        :param img:
        :param pos:
        :param verbose:
        :return:
        """

        #data = img.get_data().astype('float32')
        #gtab = reorient_bvecs(self.gtab, [img.affine for x in self.gtab.bvals if x > 0])
        from bonndit.michi import dwmri
        data, gtab, meta = dwmri.load(img)


        space = data.shape[:3]

        mask = np.ones(space)
        # TODO: Add possibility to add mask

        # Kernel_ln
        kernel_csf = shore.signal_to_kernel(self.signal_csf, self.order, self.order)
        kernel_gm = shore.signal_to_kernel(self.signal_gm, self.order, self.order)
        kernel_wm = shore.signal_to_kernel(self.signal_wm, self.order, self.order)

        # Build matrix that maps ODF+volume fractions to signal
        # in two steps: First, SHORE matrix
        M_shore = shore.matrix(self.order, self.order, self.zeta, gtab, self.tau)

        # then, convolution
        M_wm = shore.matrix_kernel(kernel_wm, self.order, self.order)
        M_gm = shore.matrix_kernel(kernel_gm, self.order, self.order)
        M_csf = shore.matrix_kernel(kernel_csf, self.order, self.order)
        M = np.hstack((M_wm, M_gm[:, :1], M_csf[:, :1]))

        # now, multiply them together
        M = np.dot(M_shore, M)
        print('Condition number of M:', np.linalg.cond(M))

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
                G[:NC, :NN] = esh.matrix(angular_order, LOTS_OF_DIRECTIONS)
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

        for i in tqdm(np.ndindex(*data.shape[:3]),
                      total=np.prod(data.shape[:3]),
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
                    c = deconvolve_hpsd(P, q, G, h, init, self.order, NN)
            else:
                c = la.lstsq(M, S)[0]
            out[i] = esh.esh_to_sym(c[:NN])
            f = kernel_csf[0][0] / max(self.signal_csf[0], 1e-10)
            wmout[i] = c[0] * f
            gmout[i] = c[NN] * f
        csfout[i] = c[NN + 1] * f

        return out, wmout, gmout, csfout


def gtab_rotate(gtab, rot_matrix):
    N = len(gtab.bvals)
    rot_bvecs = np.zeros((N, 3))
    for i in range(N):
        rot_bvecs[i, :] = np.dot(rot_matrix, gtab.bvecs[i, :])
    return gradient_table(gtab.bvals, rot_bvecs)


def gtab_reorient(gtab, old_vec, new_vec=np.array((0, 0, 1))):
    # rotate gradients to align 1st eigenvector specified direction - default is (0,0,1)
    rot_matrix = vec2vec_rotmat(old_vec, new_vec)
    return gtab_rotate(gtab, rot_matrix)


MAX_ORDER = 12

SIZES = [1, 0, 7, 0, 22, 0, 50, 0, 95, 0, 161, 0, 252]


def shore_get_size(order):
    try:
        return SIZES[order]
    except IndexError:
        raise ValueError('Please specify an order <= 12')


KERNEL_SIZES = [1, 0, 3, 0, 6, 0, 10, 0, 15, 0, 21, 0, 28]


def get_kernel_size(order):
    try:
        return KERNEL_SIZES[order]
    except IndexError:
        raise ValueError('Please specify an order <= 12')


ESH_LENGTH = [1, 0, 6, 0, 15, 0, 28, 0, 45, 0, 66, 0, 91]


# now uses a Quadratic Cone Program to do it in one shot
def deconvolve_hpsd(P, q, G, h, init, order, NN):
    # NS = len(np.array(T{4,6,8}.TT).reshape(-1))
    NS = tensor.LENGTH[order // 2]

    # first two are orthant constraints, rest positive definiteness
    dims = {'l': 2, 'q': [], 's': [NS]}

    # This init stuff is a HACK. It empirically removes some isolated failure cases
    # first, allow it to use its own initialization
    try:
        sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
    except Exception as e:
        print("error-----------", e)
        return np.zeros(NN + 2)
    if sol['status'] != 'optimal':
        # try again with our initialization
        try:
            sol = cvxopt.solvers.coneqp(P, q, G, h, dims, initvals={'x': init})
        except Exception as e:
            print("error-----------", e)
            return np.zeros(NN + 2)
        if sol['status'] != 'optimal':
            print('Optimization unsuccessful.', sol)
    c = np.array(sol['x'])[:, 0]

    return c
