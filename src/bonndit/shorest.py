from __future__ import division

try:
    from itertools import imap
except ImportError:
    # For Python 3 imap was removed as global map now returns an iterator
    imap = map

import multiprocessing as mp

import sys

import numpy as np
import numpy.linalg as la
from dipy.reconst.shore import shore_matrix
from tqdm import tqdm

from .gradients import gtab_reorient


class ShoreModel(object):
    def __init__(self, gtab, order=4, zeta=700, tau=1 / (4 * np.pi ** 2)):
        """

        :param gtab:
        :param order:
        :param zeta:
        :param tau:
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

    def _fit_helper(self, data_vecs, rcond=None):
        """ Fit shore coefficients to diffusion weighted imaging data.

        This is a helper function for parallelizing the fitting of shore
        coefficients. First it checks whether the default shore matrix can be
        used or if a vector is specified to first rotate the gradient table
        and compute a custom shore matrix for the voxel.

        :param data_vecs: tuple with DWI signal as first entry and a vector or
        None as second entry
        :return: fitted shore coefficients
        """

        signal, vec = data_vecs[0], data_vecs[1]

        if vec is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                shore_m = shore_matrix(self.order, self.zeta,
                                       gtab_reorient(self.gtab, vec),
                                       self.tau)

        else:
            shore_m = self.shore_m

        return la.lstsq(shore_m, signal, rcond)[0]

    def fit(self, data, vecs=None, verbose=False, cpus=1, desc=''):
        """ Fit shore coefficients to diffusion weighted imaging data.

        If an array of vectors is specified (vecs), the gradient table is
        rotated with an affine matrix which would align the vector to the
        z-axis. This can be used to compute comparable shore coefficients for
        white matter regions of different orientation. Use the first
        eigenvectors of precomputed diffusion tensors as vectors and use only
        regions with high fractional anisotropy to ensure working only with
        single fiber voxels.

        :param data: ndarray with DWI data
        :param vecs: ndarray which specifies for every data point the main
        direction of diffusion (e.g. first eigenvector of the diffusion tensor)
        :param verbose: set to true to show a progress bar
        :param cpus: Number of cpu workers to use
        :param desc: description for the progress bar
        :return:  array of per voxel shore coefficients
        """
        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

        # If no vectors are specified create array of Nones for iteration.
        if type(vecs) != np.ndarray and vecs is None:
            vecs = np.empty(data.shape[:-1], dtype=object)

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3 or cpus == 1:
            shore_coeff = list(tqdm(imap(self._fit_helper,
                                         zip(list(data), list(vecs))),
                                    total=np.prod(data.shape[:-1]),
                                    disable=not verbose,
                                    desc=desc))
        else:
            with mp.Pool(cpus) as p:
                shore_coeff = list(tqdm(p.imap(self._fit_helper,
                                               zip(list(data), list(vecs)),
                                               chunksize),
                                        total=np.prod(data.shape[:-1]),
                                        disable=not verbose,
                                        desc=desc))
        return ShoreFit(self, np.array(shore_coeff))


class ShoreFit(object):
    def __init__(self, model, coefs):
        self.model = model
        self.coefs = coefs
