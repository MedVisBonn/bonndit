#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os

# from __future__ import print_function
import numpy as np
import numpy.linalg as la
# import nibabel as nib
from dipy.io import read_bvals_bvecs

from . import fields
from . import storage
from . import vector
from .storage import is_nrrd, is_nifti


class GradientTable:
    # bvals: [N]
    # bvecs: unit vectors [N,3]
    def __init__(self):
        self.bvals = []
        self.bvecs = []

    def __init__(self, bvals, bvecs):
        N = len(bvals)
        self.bvals = np.array(bvals)
        self.bvecs = np.array(bvecs)
        assert (self.bvecs.shape == (N, 3))

    def n(self):
        return len(self.bvals)

    def find_shells(self):
        NGRADS = self.n()
        index = np.zeros((NGRADS,), dtype=int) - 1

        # b0
        index[self.bvals < max(self.bvals) / 100] = 0
        shell_bvals = [0.0]
        shells = 1

        # higher shells
        while sum(index < 0) > 0:
            bmin = min(self.bvals[index < 0])
            shell_bvals += [bmin]
            index[np.logical_and(index < 0, self.bvals < (bmin * 1.2))] = shells
            shells += 1

        # index contains the shell-index for each bvec
        return index, np.array(shell_bvals)

    def angles(self):
        return vector.cart_to_sphere(self.bvecs)[:, 1:]

    def rotate(self, R):
        N = self.n()
        rot_bvecs = np.zeros((N, 3))
        for i in range(N):
            rot_bvecs[i, :] = np.dot(R, self.bvecs[i, :])
        return GradientTable(self.bvals, rot_bvecs)


def nifti_get_specific_bvals_bvecs_filenames(filename):
    if filename[-4:] == '.nii':
        return filename[:-4] + '_bvals', filename[:-4] + '_bvecs'
    if filename[-7:] == '.nii.gz':
        return filename[:-7] + '_bvals', filename[:-7] + '_bvecs'
    raise Exception('...')

def nifti_get_global_bvals_bvecs_filenames(filename):
    _dir = os.path.dirname(filename)
    if len(_dir) > 0:
        _dir += '/'
    return _dir + 'bvals', _dir + 'bvecs'

def nifti_read_gtab(filename, meta):
    fsbvals, fsbvecs = nifti_get_specific_bvals_bvecs_filenames(filename)
    fgbvals, fgbvecs = nifti_get_global_bvals_bvecs_filenames(filename)

    # where to find the files?
    if os.path.isfile(fsbvals) and os.path.isfile(fsbvecs):
        # specific bvals/bvecs for this nifti file?
        bvals, bvecs = read_bvals_bvecs(fsbvals, fsbvecs)
    elif os.path.isfile(fgbvals) and os.path.isfile(fgbvecs):
        # general bvals/bvecs in the same directory?
        bvals, bvecs = read_bvals_bvecs(fgbvals, fgbvecs)
    else:
        raise Exception('no bvals/bvecs files found for ' + filename)

    if fields.auto_convert_world_space:
        bvecs = _nifti_bvecs_to_worldspace(bvecs, meta)
    return GradientTable(bvals, bvecs)

def _nifti_write_gtab(filename, gtab, meta):
    sfbvals, sfbvecs = nifti_get_specific_bvals_bvecs_filenames(filename)
    bvecs = gtab.bvecs
    if fields.auto_convert_world_space:
        bvecs = _nifti_bvecs_to_filespace(bvecs, meta)

    # bvals
    with open(sfbvals, 'w') as h:
        for i, b in enumerate(gtab.bvals):
            h.write('{}'.format(b))
            h.write('  ')
        h.write('\n')

    # bvecs
    with open(sfbvecs, 'w') as h:
        for i in range(bvecs.shape[1]):
            for j in range(bvecs.shape[0]):
                h.write('{}'.format(bvecs[j, i]))
                h.write('  ')
            h.write('\n')

def _nifti_bvecs_to_worldspace(bvecs, meta):
    bvecs = np.dot(bvecs, meta.frame.T)
    norm = la.norm(bvecs, axis=1)
    norm[norm == 0] = 1.0
    bvecs = bvecs / norm[:, None]
    bvecs[norm == 0] = np.array((0, 0, 1))
    return bvecs


def _nifti_bvecs_to_worldspace_new(bvecs, meta):
    from scipy.linalg import polar
    from scipy.linalg import inv
    R, S = polar(meta.frame)
    for i in range(len(bvecs)):
        bvecs[i] = np.dot(inv(R), bvecs[i])
    #norm = la.norm(bvecs, axis=1)
    #norm[norm == 0] = 1.0
    #bvecs = bvecs / norm[:, None]
    #bvecs[norm == 0] = np.array((0, 0, 1))
    return bvecs


def _nifti_bvecs_to_filespace(bvecs, meta):
    bvecs = np.dot(bvecs, meta.frame)
    norm = la.norm(bvecs, axis=1)
    norm[norm == 0] = 1.0
    bvecs = bvecs / norm[:, None]
    return bvecs



def load(filename, dtype='f'):
    data, meta = fields.load_basic(filename, dtype)
    data = np.array(data, dtype=dtype)
    if is_nrrd(filename):
        #		print "loading nrrd"

        # gradients
        gtab = _nrrd_parse_gtab(data, meta)
        meta = meta_del_gtab(meta)
        del (meta.key_value_pairs['modality'])

    elif is_nifti(filename):
        #		print "loading nifti"

        # gradients
        gtab = nifti_read_gtab(filename, meta)

    else:
        raise Exception("unknown file type: " + filename)

    assert data.shape[-1] == gtab.n()

    return data, gtab, meta


def save(filename, data, gtab, meta, dtype='f'):
    if is_nrrd(filename):
        meta = meta_set_gtab(meta, gtab)
        meta.key_value_pairs['modality'] = 'DWMRI'
        meta.clear_axis()
        meta.add_axis(storage.AxisType.DWMRI_GRADIENTS)

        fields.save_basic(filename, data, meta, dtype)

    elif is_nifti(filename):
        fields.save_basic(filename, data, meta, dtype)

        # save bvals/bvecs
        _nifti_write_gtab(filename, gtab, meta)

    else:
        raise Exception("unknown file type: " + filename)

def _nrrd_parse_gtab(data, meta):
    N = data.shape[-1]
    bmax = float(meta.key_value_pairs['DWMRI_b-value'])
    B = np.zeros(N)
    G = np.zeros((N, 3))
    # print meta['keyvaluepairs']['DWMRI_gradient_0000']

    for i in range(N):
        key = 'DWMRI_gradient_{:04}'.format(i)
        try:
            g = meta.key_value_pairs[key]
            g = np.array([float(x) for x in g.split(' ')])
            l2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2
            B[i] = bmax * l2
            if l2 > 0:
                G[i, :] = g / math.sqrt(l2)
            else:
                G[i, :] = np.array((0, 0, 1))
        except Exception as e:
            print(e)
            pass
    return GradientTable(B, G)

def meta_del_gtab(meta):
    meta = meta.copy()

    #	del(meta.key_value_pairs['DWMRI_b-value'])

    # remove old gradients
    to_del = []
    for k in meta.key_value_pairs:
        if k[:15] == 'DWMRI_gradient_':
            to_del.append(k)
    for k in to_del:
        del meta.key_value_pairs[k]

    return meta

def meta_set_gtab(meta, gtab):
    meta = meta_del_gtab(meta)

    bmax = max(gtab.bvals)
    meta.key_value_pairs['DWMRI_b-value'] = '{:6f}'.format(bmax)
    n = len(gtab.bvals)

    # create new gradients
    for i in range(n):
        key = 'DWMRI_gradient_{:04}'.format(i)
        g = gtab.bvecs[i, :] * math.sqrt(gtab.bvals[i] / bmax)
        meta.key_value_pairs[key] = '{:.6f} {:.6f} {:.6f}'.format(g[0], g[1], g[2])
    return meta

def get_s0(data, gtab):
    s0 = np.zeros(data.shape[:-1])
    n = 0
    for i, b in enumerate(gtab.bvals):
        if b > 20:
            continue
        for j, x in np.ndenumerate(s0):
            s0[j] += data[j + (i,)]
        n += 1
    if n == 0:
        raise Exception("no B=0 image found (B<20)")
    return s0 / n

def apply_measurement_frame(G, meta):
    frame = nrrd_get_measurement_frame(meta)
    for g in G:
        g = np.dot(frame, g)
    return G

def nrrd_get_measurement_frame(meta):
    frame = []
    for d in meta['space directions']:
        if d != 'none':
            frame += [d]
    if len(frame) != 3:
        raise Exception('invalid measurement frame: ' + meta['space directions'])
    frame = np.array(frame)
    return frame.T

def nrrd_get_origin(meta):
    return np.array(meta['space origin'])

def compute_response(order, bval, md, delta):
    response = np.zeros(order / 2 + 1)

    # these analytical expressions have been derived using sage
    exp1 = math.exp(bval * (delta - md))
    exp2 = math.exp(3 * bval * delta)
    exp3 = math.exp(bval * (-md - 2 * delta))
    erf1 = math.erf(math.sqrt(3 * bval * delta))
    sqrtbd = math.sqrt(bval * delta)
    response[0] = math.pi / math.sqrt(3.0) * exp1 * erf1 / sqrtbd
    if order < 2:
        return response
    response[1] = -0.186338998125 * math.sqrt(math.pi) * \
                  (2.0 * math.sqrt(3.0 * math.pi) * exp1 * erf1 / sqrtbd +
                   (-math.sqrt(math.pi) * exp2 * erf1 + 2.0 * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta))
    if order < 4:
        return response
    response[2] = 0.0104166666667 * math.sqrt(math.pi) * \
                  (36.0 * math.sqrt(3 * math.pi) * exp1 * erf1 / sqrtbd -
                   60 * (math.sqrt(math.pi) * exp2 * erf1 - 2 * sqrtbd * math.sqrt(3.0)) *
                   math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta) +
                   35.0 * (
                       math.sqrt(math.pi) * exp2 * erf1 - 2.0 * (2.0 * bval * delta + 1.0) * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta * bval * delta))
    if order < 6:
        return response
    response[3] = -0.00312981881551 * math.sqrt(math.pi) * \
                  (120.0 * math.sqrt(3.0 * math.pi) * exp1 * erf1 / sqrtbd -
                   420.0 * (math.sqrt(math.pi) * exp2 * erf1 - 2 * sqrtbd * math.sqrt(3.0)) *
                   math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta) +
                   630.0 * (math.sqrt(math.pi) * exp2 * erf1 - 2.0 * (2.0 * bval * delta + 1.0)
                            * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * bval * delta * delta) -
                   77.0 * (5.0 * math.sqrt(math.pi) * exp2 * erf1 -
                           2.0 * (12.0 * bval * delta * bval * delta + 10.0 * bval * delta + 5.0)
                           * sqrtbd * math.sqrt(3.0)) *
                   math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta * bval * delta * bval * delta))
    if order < 8:
        return response
    response[4] = 0.000223692796529 * math.sqrt(math.pi) * \
                  (1680.0 * math.sqrt(3.0 * math.pi) * exp1 * erf1 / sqrtbd -
                   10080.0 * (math.sqrt(math.pi) * exp2 * erf1 - 2.0 * sqrtbd * math.sqrt(3.0)) *
                   math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta) +
                   27720.0 * (math.sqrt(math.pi) * exp2 * erf1 -
                              2.0 * (2.0 * bval * delta + 1.0) * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta * bval * delta) -
                   8008.0 * (5.0 * math.sqrt(math.pi) * exp2 * erf1 -
                             2.0 * (12.0 * bval * delta * bval * delta +
                                    10.0 * bval * delta + 5.0) * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta * bval * delta * bval * delta) +
                   715.0 * (35.0 * math.sqrt(math.pi) * exp2 * erf1 -
                            2.0 * (72.0 * bval * delta * bval * delta * bval * delta +
                                   84.0 * bval * delta * bval * delta +
                                   70.0 * bval * delta + 35.0) * sqrtbd * math.sqrt(3.0))
                   * math.sqrt(3.0) * exp3 / (sqrtbd * bval * delta * bval * delta * bval * delta * bval * delta))
    #  if (!AIR_EXISTS(response[1])) {
    #    fprintf(stderr, "WARNING: "
    #            "bval=%f md=%f delta=%f\n"
    #            "exp1=%f exp2=%f exp3=%f erf1=%f sqrtbd=%f\n",
    #            bval, md, delta,
    #            exp1, exp2, exp3, erf1, sqrtbd)
    #  }
    # }
    return response
