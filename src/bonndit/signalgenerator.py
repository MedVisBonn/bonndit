import numpy as np
import pandas as pd

import bonndit as bd


class SignalGenerator(object):
    def __init__(self, modelname, gtab):
        self.gtab = gtab
        self.model = bd.dwi_models[modelname](gtab)
        self.signals = None

    def generate(self, parameters, directions=None, S0=None, cpus=None,
                 kwargs={}):
        self.signals = self.model.predict(parameters, directions, S0, cpus,
                                          **kwargs)

        columns = list(zip([str(i) for i in self.gtab.bvals],
                           [str(i) for i in self.gtab.bvecs]))
        df = pd.DataFrame(self.signals.reshape(-1, self.signals.shape[-1]),
                          columns=columns, )
        df.columns = pd.MultiIndex.from_tuples(df.columns,
                                               names=['bvals', 'bvecs'])
        return df


# TODO: make this class more general to enable use of different frameworks applying direcional information to single fiber signals.
class SignalConvolver(object):
    def __init__(self, signals, directions, frameworkname, gtab, kwargs):
        self.signals = signals
        self.directions = directions
        self.model = bd.conv_frameworks[frameworkname](gtab, **kwargs)
        self.gtab = gtab
        self.kwargs = kwargs

    def convolve(self, combination=(1, 0), cpus=None, verbose=False, desc=""):
        """

        :param combination: This parameter specifies how to build pairs of
        signal and direction for the convolution.

        - (1,0): The default, to combine every signal with a random direction
        - (0,1): Combine every direction with a random signal
        - (1,n): Combine every signal with n random directions
        - (n,1): Combine every direction with n random signals

        If you pass a ndarray of shape (n,2) each row needs to hold the index
        of a  signal and a direction (signal_ix, direction_ix) which will be
        used in the convolution.
        :return:
        """

        if type(combination) == np.ndarray:
            signal_indices = combination[:, 0]
            fODf_indices = combination[:, 1]

        elif type(combination) == tuple:
            if combination == (1, 0):
                signal_indices = np.array(range(self.signals.shape[0]))
                fODF_indices = np.random.randint(0, self.directions.shape[0],
                                                 signal_indices.shape)

            elif combination == (0, 1):
                fODF_indices = np.array(range(self.directions.shape[0]))
                signal_indices = np.random.randint(0, self.signals.shape[0],
                                                   fODF_indices.shape)

            elif combination[0] == 1 and combination[1] > 1:
                signal_indices = np.array(range(self.signals.shape[0])) \
                    .repeat(combination[1])
                fODF_indices = np.random.randint(0, self.directions.shape[0],
                                                 signal_indices.shape)

            elif combination[0] > 1 and combination[1] == 1:
                fODF_indices = np.array(range(self.directions.shape[0])) \
                    .repeat(combination[0])
                signal_indices = np.random.randint(0, self.directions.shape[0],
                                                   fODF_indices.shape)

            else:
                raise ValueError("The provided combination can not be "
                                 "evaluated. Possible combinations are: "
                                 "(1,0), (0,1), (1,n) and (n,1)")
        else:
            raise ValueError("The parameter combination needs to be a tuple "
                             "of length 2 or an 2D array where the second "
                             "dimension has length 2")

        signal = self.convolve_pairs(signal_indices, fODF_indices, cpus,
                                     verbose, desc)
        return signal

    def convolve_pairs(self, signal_indices, fODF_indices, cpus=None,
                       verbose=False, desc=""):
        """

        :param signal_indices:
        :param fODF_indices:
        :return:
        """
        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(directions.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3:
            signal = list(tqdm(imap(self._convolve_helper,
                                    zip(self.signals[signal_indices],
                                        self.directions[fODF_indices])),
                               total=signal_indices.shape[0],
                               disable=not verbose,
                               desc=desc))
        else:
            with mp.Pool(cpus) as p:
                signal = list(tqdm(p.imap(self._convolve_helper,
                                          zip(self.signals[signal_indices],
                                              self.directions[fODF_indices]),
                                          chunksize=chunksize),
                                   total=signal_indices.shape[0],
                                   disable=not verbose,
                                   desc=desc))

        return signal

    def _convolve_helper(self, signal_direction):
        """

        :param signal_fODF:
        :return:
        """
        signal_index, direction_index = signal_direction
        signal = self.signals[signal_index]
        fODf = self.directions[direction_index]
