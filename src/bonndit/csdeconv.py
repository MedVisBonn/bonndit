import multiprocessing as mp
import sys

import numpy as np
import numpy.linalg as la
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm
from tqdm import tqdm

from bonndit.michi import esh
from .gradients import gtab_reorient

try:
    from itertools import imap
except ImportError:
    # For Python 3 imap was removed as gloabl map now returns an iterator
    imap = map


class ConstrainedSphericalDeconvModel(object):
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

        return ConstrainedSphericalDeconvFit(self, np.array(sh_coeff))


class ConstrainedSphericalDeconvFit(object):
    def __init__(self, model, coeff):
        pass


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
