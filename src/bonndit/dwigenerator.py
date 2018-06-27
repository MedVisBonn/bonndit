import numpy as np

import bonndit as bd


class ParametersDKI:
    """
    This Class is initialized with a large number of values for every DKI parameter. The parameters are split into bins
    such that every bin contains the same number of values but the bins may have a different size if for a value range
    there is a smaller probability to yield parameters.

    Here we assume that the DKI parameters are completely uncorrelated
    """

    def __init__(self, radial_diff, axial_diff, radial_kappa, axial_kappa, rue_kappa):
        # TODO: Randomly create data which follows the same distribution instead of drawing from given data
        self.radial_diff = radial_diff
        self.axial_diff = axial_diff
        self.radial_kappa = radial_kappa
        self.axial_kappa = axial_kappa
        self.rue_kappa = rue_kappa

    def generate(self):
        radial_diff = np.random.choice(self.radial_diff)
        axial_diff = np.random.choice(self.axial_diff)
        radial_kappa = np.random.choice(self.radial_kappa)
        axial_kappa = np.random.choice(self.axial_kappa)
        rue_kappa = np.random.choice(self.rue_kappa)
        return (radial_diff, axial_diff, radial_kappa, axial_kappa, rue_kappa)

    def _chunkify(self, array, bin_size):
        return np.array([array[i:i + bin_size] for i in range(len(array), bin_size)])


def signal_from_DKIparams(gtab, DKI_params, evector=np.array([0, 0, 1])):
    """ Compute signal attenuation for given gradients and kurtosis parameters.

    """
    radial_diff, axial_diff, radial_kappa, axial_kappa, rue_kappa = DKI_params

    v = evector

    signal = np.zeros((len(gtab.bvals),))
    for i in range(len(gtab.bvals)):
        signal[i] = (- gtab.bvals[i] / 1000 * (radial_diff + (axial_diff - radial_diff) * np.dot(v, gtab.bvecs[i]) ** 2)
                     + ((gtab.bvals[i] / 1000 ** 2) / 6) * (radial_kappa
                                                            + (rue_kappa - 2 * radial_kappa) * np.dot(v, gtab.bvecs[
                    i]) ** 2
                                                            + (axial_kappa - rue_kappa + radial_kappa) * np.dot(v,
                                                                                                                gtab.bvecs[
                                                                                                                    i]) ** 4))
    return np.array(signal)


def make_data(radial_diff, axial_diff, radial_kappa, axial_kappa, rue_kappa, gtab, fODFs, amount):
    dki_generator = ParametersDKI(radial_diff, axial_diff, radial_kappa, axial_kappa, rue_kappa)

    param_sets = [dki_generator.generate() for _ in range(amount)]
    simple_signals = [signal_from_DKIparams(gtab, params) for params in param_sets]
    simple_response = signals_to_responses(gtab, simple_signals)


def signals_to_responses(gtab, signals):
    """

    """
    mtsm = bd.mtShoreModel(gtab)
    responses = mtsm.fit_shore(signals)

    return responses


def average_responses(gtab, responses):
    """

    """
    mtsm = bd.mtShoreModel(gtab)
    return mtsm.shore_accumulate(responses)


def compress_responses(gtab, responses):
    """

    """
    mtsm = bd.mtShoreModel(gtab)
    return np.array([mtsm.shore_compress(r) for r in responses])
