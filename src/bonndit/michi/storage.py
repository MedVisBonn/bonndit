#!/usr/bin/python
# -*- coding: utf-8 -*-


import nibabel as NIB
import numpy as np

from . import nrrd as NRRD

# treat arrays in the program as world space...
# automatically try to convert between world/file-space while loading/saving
auto_convert_world_space = True


class AxisType:
    SPACE = 1
    LIST = 2
    DWMRI_GRADIENTS = 3
    VECTOR = 8
    TENSOR2 = 10
    TENSOR4 = 11
    TENSOR6 = 12
    TENSOR8 = 13
    TENSOR10 = 14
    TENSOR12 = 15
    TENSOR = [None, None, TENSOR2, None, TENSOR4, None, TENSOR6, None, TENSOR8, None, TENSOR10, None, TENSOR12]

    @classmethod
    def label(cls, _type):
        if _type == cls.TENSOR2:
            return 'tijk_mask_2o3d_sym'
        if _type == cls.TENSOR4:
            return 'tijk_mask_4o3d_sym'
        if _type == cls.TENSOR6:
            return 'tijk_mask_6o3d_sym'
        if _type == cls.TENSOR8:
            return 'tijk_mask_8o3d_sym'
        if _type == cls.TENSOR10:
            return 'tijk_mask_10o3d_sym'
        if _type == cls.TENSOR12:
            return 'tijk_mask_12o3d_sym'
        return ''

    @classmethod
    def kind(cls, _type):
        if _type == cls.SPACE:
            return 'space'
        if _type == cls.LIST:
            return 'list'
        if _type == cls.DWMRI_GRADIENTS:
            return 'list'
        if _type == cls.VECTOR:
            return 'list'
        #		if _type == cls.TENSOR2:
        #			return '3D-masked-symmetric-matrix'
        return '???'

    @classmethod
    def size(cls, _type):
        if _type == cls.VECTOR:
            return 3
        if _type == cls.TENSOR2:
            return 6
        if _type == cls.TENSOR4:
            return 15
        if _type == cls.TENSOR6:
            return 28
        if _type == cls.TENSOR8:
            return 45
        if _type == cls.TENSOR10:
            return 66
        if _type == cls.TENSOR12:
            return 91
        return 0


class Meta:
    # frame : 3x3 [component, axis]
    def __init__(self):
        self.origin = np.zeros(3)
        self.frame = np.eye(3)
        self.space_type = 'right-anterior-superior'
        self.key_value_pairs = {}
        self.data_axis = []

    def clear_axis(self):
        self.data_axis = []

    def add_axis(self, _type):
        self.data_axis += [_type]
        return self

    def _with(self, types):
        out = self.copy()
        out.clear_axis()
        if isinstance(types, int):
            out.add_axis(types)
        else:  # list
            for t in types:
                out.add_axis(t)
        return out

    def to_nrrd(self):
        m = {}
        m['labels'] = []
        m['kinds'] = []
        m['space directions'] = []

        # data axes
        for a in self.data_axis:
            m['labels'] += ['"' + AxisType.label(a) + '"']
            m['kinds'] += [AxisType.kind(a)]
            m['space directions'] += ['none']

        # 3d space
        for i in range(3):
            m['labels'] += ['""']
            m['kinds'] += ['space']
            m['space directions'] += [tuple(self.frame[:, i])]
        # file order: data first, then space...

        m['space'] = self.space_type
        m['space origin'] = tuple(self.origin)
        m['keyvaluepairs'] = self.key_value_pairs
        return m

    def affine(self):
        a = np.zeros((4, 4))
        a[:3, 3] = self.origin
        for i in range(3):
            a[:3, i] = self.frame[:, i]
        a[3, 3] = 1
        return a

    def copy(self):
        c = Meta()
        c.origin = np.copy(self.origin)
        c.frame = np.copy(self.frame)
        c.space_type = self.space_type
        c.key_value_pairs = self.key_value_pairs.copy()
        return c

    def check_nifti_compatible(self):
        # is the frame mostly diagonal?
        # and scaling isotropic
        dia = [abs(self.frame[i, i]) for i in range(3)]
        if max(dia) - min(dia) < 0.1:
            return
        if abs(self.frame[0, 1]) + abs(self.frame[0, 2]) + abs(
            self.frame[1, 2]) < 0.1:
            return
        print('-----------------------------------------------------------')
        print('  ...might have problems with this frame in nifti...')
        print(self.frame)
        print('-----------------------------------------------------------')


default_meta = Meta()


def nrrd_to_meta3d(meta):
    m = Meta()
    try:
        m.space_type = meta['space']
    except:
        pass
    try:
        frame = [v for v in meta['space directions'] if v != 'none']
        if len(frame) == 3:
            m.frame = np.array(frame, dtype=float)
    except:
        pass
    try:
        origin = meta['space origin']
        if len(origin) == 3:
            m.origin = np.array(origin, dtype=float)
    except:
        pass
    try:
        m.key_value_pairs = meta['keyvaluepairs']
    except:
        pass
    return m


def affine_to_meta3d(affine):
    m = Meta()

    # affine is really just an affine matrix
    affine = np.array(affine)
    m.frame = affine[:3, :3]
    m.origin = affine[:3, 3]
    return m


class FileType:
    UNKNOWN = 0
    NRRD = 1
    NIFTI = 2
    NPZ = 3


def is_nrrd(filename):
    return filename[-5:] == '.nrrd' or filename[-5:] == '.nhdr'


def is_nifti(filename):
    return filename[-4:] == '.nii' or filename[-7:] == '.nii.gz'


def is_npz(filename):
    return filename[-4:] == '.npz'


def filetype(_filename):
    if is_nrrd(_filename):
        return FileType.NRRD
    if is_nifti(_filename):
        return FileType.NIFTI
    if is_npz(_filename):
        return FileType.NPZ
    return FileType.UNKNOWN


######################################################################
# general naive load/save
#   (will not rotate axis correctly)

def apply_dtype(a, dtype):
    if dtype is None:
        return a
    return np.array(a, dtype=dtype)


def load(filename, dtype=None):
    if is_nrrd(filename):
        data, meta = NRRD.read(filename)
        data = apply_dtype(data, dtype)
        meta = nrrd_to_meta3d(meta)
        return data, meta
    elif is_nifti(filename):
        img = NIB.load(filename)
        data = img.get_data()
        data = apply_dtype(data, dtype)
        affine = img.get_affine()
        meta = affine_to_meta3d(affine)

        return data, meta
    #	elif is_npz(filename):
    #		return
    raise Exception("unknown file type: " + filename)


def save(filename, data, meta=default_meta, dtype=None):
    data = apply_dtype(data, dtype)
    if is_nrrd(filename):
        if len(data.shape) >= 3:
            # correct field
            NRRD.write(filename, data, meta.to_nrrd())
        else:
            # dirty hack...
            NRRD.write(filename, data)
    elif is_nifti(filename):
        img = NIB.Nifti1Image(data, meta.affine())
        NIB.save(img, filename)
    else:
        raise Exception("unknown file type: " + filename)


# axis can be int or [int, int, int...]
def zeros(space_dimensions, axis, dtype='f'):
    if isinstance(axis, int):
        axis = [axis]
    size = space_dimensions + tuple([AxisType.size(a) for a in axis])
    return np.zeros(size, dtype=dtype)
