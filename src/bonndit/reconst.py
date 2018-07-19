models = {"mtshore" = bd.mtShoreModel}

class Reconstructor(object):

    def __init__(self, data, gtab):
        self.data = data
        self.gtab = gtab

    def reconstruct(self, model=None, mask=None, verbose=False, cpus=None,
                    kwargs=[]):
        """Reconstruct DWI data with the specified model.

        For the specified model, this method returns a ModelFit object which
        contains the corresponding model parameters for every non-masked voxel
        in the input data. The specified model needs to provide a fit function,
        which returns the parameters for a single voxel

        :param model:
        :param mask:
        :param verbose:
        :param cpus:
        :param kwargs:
        :return:
        """

        # Validation, does data/gtab fits to model

        # Multiprocessing, fit model to data, return parameters

        # 1000 chunks for the progressbar to run smoother
        chunksize = max(1, int(np.prod(fODFs.shape[:-1]) / 1000))

        # Iterate over the data indices; show progress with tqdm
        # multiple processes for python > 3
        if sys.version_info[0] < 3 or cpus == 1:
            params = list(tqdm(imap(shorefit_convolver_helper,
                                    zip(list(shore_fits),
                                        list(fODF_sets))),
                               total=len(shore_fits),
                               disable=not verbose,
                               desc=desc))
        else:
            with mp.Pool(cpus) as p:
                params = list(tqdm(p.imap(shorefit_convolver_helper,
                                          zip(list(shore_fits),
                                              list(fODF_sets)),
                                          chunksize),
                                   total=len(shore_fits),
                                   disable=not verbose,
                                   desc=desc))


class Predictor(object):

    def __init(self, model=None):
        self.model = model

    def predict(self, parameters, kwargs=[]):
        pass
