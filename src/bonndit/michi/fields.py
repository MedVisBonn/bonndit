#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from . import storage
from .storage import is_nrrd, is_nifti, filetype, FileType, AxisType, default_meta
from . import tensor as T

# treat arrays in the program as world space...
# automatically try to convert between world/file-space while loading/saving
auto_convert_world_space = True


######################################################################
# mid level load/save
#   (with axis rotation)

def _basic_process_load(filename, data, meta):
    if is_nrrd(filename):
        if len(data.shape) == 4:
            # permute axis
            return np.rollaxis(data, 0, 4)
        elif len(data.shape) == 5:
            # permute axis
            data = np.rollaxis(data, 1, 5)
            return np.rollaxis(data, 0, 4)
        elif len(data.shape) > 5:
            raise Exception("can't load nrrd with more than 5 axes")
    elif is_nifti(filename):
        meta.check_nifti_compatible()
    return data


def load_basic(filename, dtype=None):
    data, meta = storage.load(filename, dtype)
    return _basic_process_load(filename, data, meta), meta


def save_basic(filename, data, meta=default_meta, dtype=None):
    if is_nrrd(filename):
        data2 = data
        if len(data.shape) == 4:
            # permute axis
            data2 = np.rollaxis(data, 3, 0)
        elif len(data.shape) == 5:
            # permute axis
            data2 = np.rollaxis(data, 3, 0)
            data2 = np.rollaxis(data2, 4, 1)
        elif len(data.shape) > 5:
            raise Exception("can't save nrrd with more than 5 axes")
        storage.save(filename, data2, meta, dtype)
    elif is_nifti(filename):
        meta.check_nifti_compatible()
        storage.save(filename, data, meta, dtype)
    else:
        storage.save(filename, data, meta, dtype)


######################################################################
# scalar

def load_scalar(filename, dtype=None):
    data, meta = load_basic(filename, dtype)
    assert (len(data.shape) == 3)
    return data, meta._with([])


def save_scalar(filename, data, meta=default_meta, dtype=None):
    assert (len(data.shape) == 3)
    meta.clear_axis()
    save_basic(filename, data, meta._with([]), dtype)


def make_scalar(s, ref):
    if filetype(s) == FileType.UNKNOWN:
        # not a filename... try to create a new field using it as a value
        r = np.zeros(ref.shape[:3])
        r += float(s)
        return r
    else:
        # filename...
        return load_scalar(s, dtype=float)[0]


######################################################################
# list

def load_list(filename, dtype=None):
    data, meta = load_basic(filename, dtype)
    return data, meta


def save_list(filename, data, meta=default_meta, dtype=None):
    assert (len(data.shape) == 4)
    save_basic(filename, data, meta._with(AxisType.LIST), dtype)


def load_list2(filename, dtype=None):
    data, meta = load_basic(filename, dtype)
    assert (len(data.shape) == 5)
    return data, meta


def save_list2(filename, data, meta=default_meta, dtype=None):
    assert (len(data.shape) == 5)
    save_basic(filename, data, meta._with([AxisType.LIST, AxisType.LIST]), dtype)


######################################################################
# vector

def __nifti_flip_signs_vec(data, meta):
    out = data.copy()
    # flip the signs of vector coords as needed
    affine = meta.affine()
    for i in range(3):
        if affine[i, i] < 0:
            out[:, :, :, i] = -out[:, :, :, i]
    return out


def _is_vector(filename, data):
    if len(data.shape) != 4:
        return False
    return data.shape[3] == 3


def _vector_process_load(filename, data, meta):
    if is_nifti(filename):
        if auto_convert_world_space:
            return __nifti_flip_signs_vec(data, meta)
    return data


def load_vector(filename, dtype='d'):
    data, meta = load_basic(filename, dtype=dtype)
    assert (data.shape[3] == 3)
    return _vector_process_load(filename, data, meta), meta


def save_vector(filename, data, meta=default_meta, dtype='f'):
    assert (len(data.shape) == 4)
    assert (data.shape[3] == 3)
    out = data
    if is_nifti(filename):
        if auto_convert_world_space:
            out = __nifti_flip_signs_vec(data, meta)
    save_basic(filename, out, meta._with(AxisType.VECTOR), dtype=dtype)


######################################################################
# vector list

# [x,y,z,#vector,components]

def load_vector_list(filename, dtype='d'):
    data, meta = load_basic(filename, dtype=dtype)
    assert (len(data.shape) == 5)
    assert (data.shape[4] == 3)
    if is_nifti(filename):
        raise Exception("not a good idea...")
    return data, meta


def save_vector_list(filename, data, meta=default_meta, dtype='f'):
    assert (len(data.shape) == 5)
    assert (data.shape[4] == 3)
    out = data
    if is_nifti(filename):
        raise Exception("not a good idea...")
    save_basic(filename, out, meta._with([AxisType.LIST, AxisType.VECTOR]), dtype=dtype)


######################################################################
# tensor

def __nifti_flip_signs_t2(data, meta):
    out = data.copy()
    # flip the signs of vector coords as needed
    axes = T.INDEX[2]
    affine = meta.affine()
    for i, a in enumerate(axes):
        if affine[a[0], a[0]] * affine[a[1], a[1]] < 0:
            out[:, :, :, i] = -out[:, :, :, i]
    return out


def _is_tensor(filename, data):
    if len(data.shape) != 4:
        return False
    if is_nrrd(filename):
        return data.shape[3] in [7, 16, 29, 46, 67, 92]
    return data.shape[3] in [6]  # ,15,28,45]


def _tensor_process_load(filename, data, meta):
    if is_nrrd(filename):
        return data[:, :, :, 1:]
    elif is_nifti(filename):
        assert (data.shape[3] in [6])  # ,15,28,45])
        tensors = data
        if auto_convert_world_space:
            if data.shape[3] == 6:  # order 2
                return __nifti_flip_signs_t2(tensors, meta)
            else:
                raise Exception("can't load nifti tensor of order > 2 into world-coordinates yet...")
        return data
    raise Exception("unknown file type: " + filename)


def _tensor_extract_mask(filename, data):
    if is_nrrd(filename):
        return data[:, :, :, 0]
    return np.ones(data.shape[:3])


# return: tensors, mask, meta
def load_tensor(filename, dtype='d'):
    data, meta = load_basic(filename, dtype)
    assert (len(data.shape) == 4)
    mask = _tensor_extract_mask(filename, data)
    tensors = _tensor_process_load(filename, data, meta)
    return tensors, mask, meta


def save_tensor(filename, data, mask=None, meta=default_meta, dtype='d'):
    assert (len(data.shape) == 4)
    assert (data.shape[-1] in [6, 15, 28, 45, 66, 91])
    if mask is not None:
        assert (len(mask.shape) == 3)
        assert (mask.shape == data.shape[:-1])
    order = T.get_order(data)

    if is_nrrd(filename):
        if mask is None:
            mask = np.ones(data.shape[:3])
        mmask = mask.reshape(mask.shape + (1,))
        out = np.concatenate((mmask, data), axis=3)
    elif is_nifti(filename):
        out = data
        if auto_convert_world_space:
            if order == 2:
                out = __nifti_flip_signs_t2(data, meta)
            else:
                raise Exception("can't save nifti tensor of order > 2 from world-corrdinates yet...")
    else:
        raise Exception("unknown file type: " + filename)

    save_basic(filename, out, meta._with(AxisType.TENSOR[order]), dtype)


######################################################################
# ....

def load_guess(filename):
    data, meta = load_basic(filename)
    if len(data.shape) == 3:  # is_scalar(filename, data):
        return data, meta, 'scalar'
    if len(data.shape) == 4:
        if _is_vector(filename, data):
            return _vector_process_load(filename, data, meta), meta, 'vector'
        if _is_tensor(filename, data):
            return _tensor_process_load(filename, data, meta), meta, 'tensor'
    return data, meta, '?'
