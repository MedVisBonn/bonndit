import errno
import functools
import multiprocessing as mp
import os

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.multi_voxel import MultiVoxelFit
from tqdm import tqdm


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
            :param verbose: set to true to show a progress bar
            :param cpus: Number of cpu workers to use
            :param desc: description for the progress bar
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
            return MultiVoxel(self, fit_array, mask)

        return new_method

    return decorator_mulit_voxel_method


class MultiVoxel(MultiVoxelFit):
    def __init__(self, model, fit_array, mask):
        super().__init__(model, fit_array, mask)

    @classmethod
    def load(cls, filepath, model_class, fit_class):
        filecontent = np.load(filepath)

        gtab = gradient_table(filecontent['bvals'], filecontent['bvecs'])
        model_params = {key: filecontent[key] for key in filecontent.keys()
                        if key not in ['data', 'mask', 'bvals', 'bvecs']}
        model = model_class(gtab, **model_params)

        data = filecontent['data']
        mask = filecontent['mask']

        fit_array = np.empty(data.shape[:-1], dtype=object)
        for ijk in np.ndindex(*data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = fit_class(data[ijk])

        return cls(model, fit_array, mask)

    def save(self, filepath, affine=None, type='npz'):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if affine is None:
            affine = np.zeros((4, 4))

        data = self.fit_array.coeffs
        mask = self.mask
        if type == 'npz':
            np.savez(filepath, data=data, mask=mask, **self._model_params)


        elif type == 'nii':
            img = nib.Nifti1Image(data, affine=affine)
            nib.save(img, filepath)
