from __future__ import division

import errno
import multiprocessing as mp
import os
import sys

import numpy as np
import numpy.linalg as la
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import real_sph_harm
from tqdm import tqdm

from bonndit.michi import esh
from .gradients import gtab_reorient

try:
    from itertools import imap
except ImportError:
    # For Python 3 imap was removed as gloabl map now returns an iterator
    imap = map


class SphericalHarmonicsModel(object):
    def __init__(self, gtab, order=4):
        """

        :param gtab:
        :param order:
        """
        self.gtab = gtab
        self.order = order

        # Ignore division by zero warning
        # dipy.core.geometry.cart2sphere -> theta = np.arccos(z / r)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.sh_m = esh_matrix(self.order, self.gtab)

    def _fit_helper(self, data_vecs, rcond=None):
        """

        :param data_vecs:
        :param rcond:
        :return:
        """
        signal, vec = data_vecs[0], data_vecs[1]

        if vec is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                sh_m = esh_matrix(self.order, gtab_reorient(self.gtab, vec))

        else:
            sh_m = self.sh_m

        return la.lstsq(sh_m, signal, rcond)[0]

    def fit(self, data, vecs=None, verbose=False, cpus=1, desc=""):
        """

        :param data:
        :param vecs:
        :param verbose:
        :param cpus:
        :param desc:
        :return:
        """

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

        # If no vectors are specified create array of Nones for iteration.
        if type(vecs) != np.ndarray and vecs is None:
            vecs = np.empty(data.shape[:-1], dtype=object)

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3 or cpus == 1:
            sh_coeff = list(tqdm(imap(self._fit_helper,
                                      zip(list(data), list(vecs))),
                                 total=np.prod(data.shape[:-1]),
                                 disable=not verbose,
                                 desc=desc))
        else:
            with mp.Pool(cpus) as p:
                sh_coeff = list(tqdm(p.imap(self._fit_helper,
                                            zip(list(data), list(vecs)),
                                            chunksize),
                                     total=np.prod(data.shape[:-1]),
                                     disable=not verbose,
                                     desc=desc))

        return SphericalHarmonicsFit(self, np.array(sh_coeff))


class SphericalHarmonicsFit(object):
    def __init__(self, model, coefs, kernel="rank1"):
        """

        :param model:
        :param coefs:
        :param kernel:
        """
        self.model = model
        self.coefs = coefs
        self.kernel_type = kernel

        self.order = model.order
        self.gtab = model.gtab

    @classmethod
    def load(cls, filepath):
        """ Load a precalculated SphericalHarmonicsFit object from a file.

        :param filepath: path to the saved SphericalHarmonicsFit object
        :return: SphericalHarmonicsFit object which contains wm response
        function
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = SphericalHarmonicsModel(gtab, response['order'])

        return cls(model, response['coefs'])

    def save(self, filepath):
        """ Save a mtShoreFit object to a file.

        :param filepath: path to the file
        """
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(filepath, coefs=self.coefs,
                 order=self.order, bvals=self.gtab.bvals,
                 bvecs=self.gtab.bvecs)


class ShResponseEstimator(object):
    def __init__(self, gtab, order=4):
        """

        :param gtab:
        :param order:
        """
        self.gtab = gtab
        self.order = order

    def fit(self, data, dti_vecs, wm_mask, verbose=False, cpus=1):
        """

        :param data:
        :param dti_vecs:
        :param wm_mask:
        :param verbose:
        :param cpus:
        :return:
        """
        # Check if tissue masks give at least a single voxel
        if np.sum(wm_mask.get_data()) < 1:
            raise ValueError('No white matter voxels specified by wm_mask. '
                             'A corresponding response can not be computed.')

        # Calculate wm response
        wm_voxels = data.get_data()[wm_mask.get_data() == 1]
        wm_vecs = dti_vecs.get_data()[wm_mask.get_data() == 1]
        wmshore_coeffs = SphericalHarmonicsModel(
            self.gtab, self.order).fit(wm_voxels, wm_vecs, verbose=verbose,
                                       cpus=cpus,
                                       desc='WM response').coefs
        wmshore_coeff = self.sh_accumulate(wmshore_coeffs)
        signal_wm = self.sh_compress(wmshore_coeff)

        return ShResponse(self, signal_wm)

    def sh_accumulate(self, sh_coefs):
        """

        :param sh_coefs:
        :return:
        """
        sh_accum = np.zeros_like(sh_coefs[0])
        accum_count = 0

        # Iterate over the data indices
        for i in np.ndindex(*sh_coefs.shape[:-1]):
            sh_accum += sh_coefs[i]
            accum_count += 1
        if accum_count == 0:
            return sh_accum

        # Do not show divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            return sh_accum / accum_count

    def sh_compress(self, coefs):
        """ Compress the shore coefficients

        An axial symetric response function aligned to the z-axis can be
        described fully using only the z-rotational part of the shore
        coefficients.

        :param coefs: shore coefficients
        :return: z-rotational part of the shore coefficients
        """
        r = np.zeros(esh.get_kernel_size(self.order))

        # counter = 0
        # for l in range(0, self.order):
        #    counter = counter+l
        #    if l % 2 == 0:
        #        r[int(l/2)] = coefs[counter]
        r[0] = coefs[0]
        r[1] = coefs[3]
        r[2] = coefs[10]

        return r


class ShResponse(object):
    def __init__(self, model, sh_coef, kernel="rank1"):
        """

        :param model:
        :param sh_coef:
        :param kernel:
        """
        self.model = model
        self.gtab = model.gtab
        self.order = model.order

        self.wm_response = sh_coef

        # The deconvolution kernels are computed in set_kernel
        self.kernel_type = kernel
        self.kernel_wm = None
        # self.set_kernel(kernel)

    @classmethod
    def load(cls, filepath):
        """ Load a precalculated mtShoreFit object from a file.

        :param filepath: path to the saved mtShoreFit object
        :return: mtShoreFit object which contains response functions for white
        matter, gray matter and CSF
        """
        response = np.load(filepath)

        gtab = gradient_table(response['bvals'], response['bvecs'])
        model = ShResponseEstimator(gtab, response['order'])

        return cls(model, response['wm_resp'])

    def save(self, filepath):
        """ Save a mtShoreFit object to a file.

        :param filepath: path to the file
        """
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savez(filepath, wm_resp=self.wm_response, order=self.order,
                 bvals=self.gtab.bvals, bvecs=self.gtab.bvecs)


def esh_matrix(order, gtab):
    """ Matrix that evaluates SH coeffs in the given directions

    :param order:
    :param gtab:
    :return:
    """
    bvecs = gtab.bvecs
    r, theta, phi = cart2sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])
    theta[np.isnan(theta)] = 0
    M = np.zeros((bvecs.shape[0], esh.LENGTH[order]))
    counter = 0
    for l in range(0, order + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = real_sph_harm(m, l, theta, phi)
            counter += 1
    return M
