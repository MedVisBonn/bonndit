try:
    from itertools import imap
except ImportError:
    # For Python 3 imap was removed as gloabl map now returns an iterator
    imap = map

import multiprocessing as mp
import sys

import numpy as np
from tqdm import tqdm


class CylKurtosisModel(object):

    def __init__(self, gtab):
        self.gtab = gtab

    def fit(self, data):
        # return CylKurtosisFit()
        pass

    def _predict_helper(self, kurt_dir):
        """

        :param kurt_params:
        :param direction:
        :return:
        """
        kurt_params, direction = kurt_dir
        ax_dif, ra_dif, ax_kap, ra_kap, rue_kap = np.rollaxis(kurt_params,
                                                              -1)
        v = direction
        signal = np.zeros((len(self.gtab.bvals),))
        for i in range(len(self.gtab.bvals)):
            signal[i] = (- self.gtab.bvals[i] / 1000
                         * (ra_dif + (ax_dif - ra_dif)
                            * np.dot(v, self.gtab.bvecs[i]) ** 2)
                         + (((self.gtab.bvals[i] / 1000) ** 2) / 6)
                         * (ra_kap + (rue_kap - 2 * ra_kap)
                            * np.dot(v, self.gtab.bvecs[i]) ** 2
                            + (ax_kap - rue_kap + ra_kap)
                            * np.dot(v, self.gtab.bvecs[i]) ** 4))

        return np.exp(np.array(signal))

    def predict(self, kurt_params, directions=None, S0=None, verbose=False,
                cpus=None, desc=""):
        """Predict a diffusion weighted signal.

        Given the parameters of a

        :param kurt_params:
        :param directions:
        :param S0:
        :param verbose:
        :param cpus:
        :param desc:
        :return:
        """
        # If no fiber directions are given, assume that fibers have been
        # aligned to the z-axis
        if directions is None:
            directions = np.full(kurt_params.shape[:-1] + (3,),
                                 np.array([0, 0, 1]))

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(directions.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3:
            signal = list(tqdm(imap(self._predict_helper,
                                    zip(list(kurt_params),
                                        list(directions))),
                               total=np.prod(directions.shape[:-1]),
                               disable=not verbose,
                               desc=desc))
        else:
            with mp.Pool(cpus) as p:
                signal = list(tqdm(p.imap(self._predict_helper,
                                          zip(list(kurt_params),
                                              list(directions)),
                                          chunksize),
                                   total=np.prod(
                                       directions.shape[:-1]),
                                   disable=not verbose,
                                   desc=desc))

        # TODO: Should there be the possibility of per signal S0 values?
        if S0:
            signal = np.array(signal) * S0
        else:
            signal = np.array(signal)

        return signal

class CylKurtosisFit(object):

    def __init__(self, model, kurt_params, directions=None):
        self.model = model
        self.gtab = self.model.gtab
        self.kurt_params = kurt_params
        self.n = np.prod(self.kurt_params.shape[:-1])


