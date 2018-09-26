import functools
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
from dipy.reconst.multi_voxel import MultiVoxelFit
from tqdm import tqdm


class ReconstModel(ABC):
    def __init__(self, gtab):
        """

        :param gtab:
        """
        self.gtab = gtab

    @abstractmethod
    def fit(self, data, mask, **kwargs):
        """

        :param data:
        :param mask:
        :param kwargs:
        :return:
        """
        raise NotImplementedError(
            "{} does not implement 'fit()' yet".format(self.__name__))


class ReconstFit(ABC):
    def __init__(self, model, coeffs):
        """

        :param model:
        :param coeffs:
        """
        self.model = model
        self.coeffs = coeffs

    @abstractmethod
    def predict(self, gtab):
        """

        :param gtab:
        :return:
        """
        raise NotImplementedError(
            "{} does not implement 'predict()' yet".format(self.__name__))


def fit_helper(args_kwargs):
    args = args_kwargs[1:-1]
    kwargs = args_kwargs[-1]
    func = args_kwargs[0]
    return func(*args, **kwargs)


def multi_voxel_method(per_voxel_data=[]):
    def decorator_mulit_voxel_method(single_voxel_method):
        """

        :param single_voxel_method:
        :return:
        """

        @functools.wraps(single_voxel_method)
        def new_method(self, data, mask=None, verbose=False, cpus=1, desc='',
                       **kwargs):
            """

            :param self:
            :param data:
            :param mask:
            :param verbose:
            :param cpus:
            :param desc:
            :param args:
            :param kwargs:
            :return:
            """
            space = data.shape[:-1]

            if mask is None:
                mask = np.ones(space)
            if mask.shape != space:
                raise ValueError("mask and data shape do not match")

            # Convert integer to boolean mask
            mask = np.ma.make_mask(mask)

            # 1000 chunks for the progressbar to run smoother
            chunksize = max(1, int(np.prod(data.shape[:-1]) / 1000))

            # collect kwargs which are the same for all voxels
            general_kwargs = {key: kwargs[key] for key in kwargs.keys()
                              if key not in per_voxel_data}
            args_kwargs = []

            for ijk in np.ndindex(*data.shape[:-1]):
                if mask[ijk]:
                    # collect kwargs per voxel if specified in the kwargs
                    per_voxel_kwargs = {key: kwargs[key][ijk]
                                        for key in per_voxel_data
                                        if key in kwargs.keys()}

                    new_kwargs = {**general_kwargs, **per_voxel_kwargs}

                    args_kwargs.append((single_voxel_method, self, data[ijk],
                                        new_kwargs))

            if cpus == 1:
                coeffs = list(tqdm(map(fit_helper, args_kwargs),
                                   total=np.prod(data.shape[:-1]),
                                   disable=not verbose,
                                   desc=desc))
            else:
                with mp.Pool(cpus) as p:
                    coeffs = list(tqdm(p.imap(fit_helper, args_kwargs,
                                              chunksize),
                                       total=np.prod(data.shape[:-1]),
                                       disable=not verbose,
                                       desc=desc))

            fit_array = np.empty(data.shape[:-1], dtype=object)
            fit_array[mask] = coeffs
            return MultiVoxelFit(self, fit_array, mask)

        return new_method

    return decorator_mulit_voxel_method
