#!/usr/bin/python
# -*- coding: utf-8 -*-

# we're using lexicographical ordering of indices:
#   xx,xy,xz,yy,yz,zz

# but some operations are easier using 'counted indices'
#   200,110,101,020,011,002

from math import factorial
from .mmath import binom, multinom
from .vector import project
from itertools import product
import numpy as np
import math


class ZeroTensorError(Exception):
    pass


class InvalidSizeError(Exception):
    def __init__(self, size):
        super().__init__('unknown tensor size: {}'.format(size))


MAX_ORDER = 16

LENGTH = [((d + 2) * (d + 1)) // 2 for d in range(MAX_ORDER + 1)]


# (0,0,0,1,1,1,1,1,2,2) -> [3,5,2]
def index_count(index):
    n = [0, 0, 0]
    for i in index:
        n[i] += 1
    return n


# create the INDEX list for a specific order
def _create_index(order):
    # combinations_with_replacement([0,1,2], order)...guaranteed to be sorted?
    if order == 0:
        return [()]
    index = []
    i = [0 for k in range(order)]
    while i[0] <= 2:
        index += [tuple(i)]
        i[order - 1] += 1
        for k in range(order - 1, 0, -1):
            if i[k] > 2:
                i[k - 1] += 1
                for j in range(k, order):
                    i[j] = i[k - 1]
    return index


# print(_create_index(8))

INDEX = [_create_index(i) for i in range(MAX_ORDER + 1)]

# counted indiced xxyz -> [2,1,1]
CINDEX = [[index_count(i) for i in I] for I in INDEX]


def _create_multiplier(order):
    m = []
    nn = factorial(order)
    for I in CINDEX[order]:
        m += [nn / factorial(I[0]) / factorial(I[1]) / factorial(I[2])]
    return m


MULTIPLIER = [_create_multiplier(i) for i in range(MAX_ORDER + 1)]


def order(t):
    try:
        s = t.shape[-1]
    except:
        # in case, we're not given a np.array
        s = len(t)
    try:
        return LENGTH.index(s)
    except:
        raise InvalidSizeError(s)


get_order = order


def zero(order):
    return np.zeros(LENGTH[order])


def iso(order):
    r = zero(order)
    for iz in range(0, order + 1, 2):
        for ix in range(0, (order - iz) + 1, 2):
            I = [ix, order - ix - iz, iz]
            ii = CINDEX[order].index(I)
            r[ii] = 1.0 * binom((order - iz) // 2, ix // 2) * binom(order // 2, iz // 2) / MULTIPLIER[order][ii]
    return r


# compute the tensor poduct  v⊗v⊗v... for a vector v
def power(v, order):
    return np.array([v[0] ** I[0] * v[1] ** I[1] * v[2] ** I[2] for I in CINDEX[order]])


# <a, b>
def dot(a, b):
    assert (len(a) == len(b))
    order = get_order(a)
    return np.dot(a, b * MULTIPLIER[order])


# Frobenius norm ||T||
def norm(a):
    return math.sqrt(dot(a, a))


# T(v) = T(v,v,v..)
def eval(t, v):
    order = get_order(t)
    return dot(t, power(v, order))


# return the vector w = T(v,v,...,v,-)
def s_form(t, v):
    o = order(t)
    vv = power(v, o - 1)
    w = np.zeros(3)
    for i, I in enumerate(CINDEX[o - 1]):
        ix = CINDEX[order].index([I[0] + 1, I[1], I[2]])
        iy = CINDEX[order].index([I[0], I[1] + 1, I[2]])
        iz = CINDEX[order].index([I[0], I[1], I[2] + 1])
        m = MULTIPLIER[o - 1][i]
        w[0] += t[ix] * vv[i] * m
        w[1] += t[iy] * vv[i] * m
        w[2] += t[iz] * vv[i] * m
    return w


# grad(T(v)) at v
#  ...but projected onto the sphere (normal to v)
#   (because we don't want to optimize radially)
def grad_eval_project(t, v):
    o = order(t)
    w = s_form(t, v) * o
    return w - project(w, v)


# sum_a T_{aaijkl...}
# will always return an array, even for order2 -> order0
def trace(t):
    order = get_order(t)
    assert order >= 2
    r = np.zeros(LENGTH[order - 2])
    for i, I in enumerate(CINDEX[order - 2]):
        ixx = CINDEX[order].index([I[0] + 2, I[1], I[2]])
        iyy = CINDEX[order].index([I[0], I[1] + 2, I[2]])
        izz = CINDEX[order].index([I[0], I[1], I[2] + 2])
        r[i] = t[ixx] + t[iyy] + t[izz]
    return r


# R⊗R⊗R⊗... for transforming tensors via np.dot()
def matrix_power(R, order):
    M = np.zeros((LENGTH[order], LENGTH[order]))
    for i, I in enumerate(INDEX[order]):
        for p in product([0, 1, 2], repeat=order):
            x = 1
            for j, pp in enumerate(p):
                x *= R[I[j], pp]
            k = CINDEX[order].index(index_count(p))
            M[i, k] += x
    return M


# ---------------------------------------------------------------
#        H-matrix
# ---------------------------------------------------------------

# H-matrixification-matrices :P
_H_index_matrix = [None for i in range(MAX_ORDER + 1)]


def _create_H_index_matrix(order):
    assert (order % 2) == 0
    s = order // 2
    N = LENGTH[s]
    H = np.zeros((N, N), dtype=int)
    for i, I in enumerate(CINDEX[s]):
        for j, J in enumerate(CINDEX[s]):
            c = [a + b for a, b in zip(I, J)]
            H[i, j] = CINDEX[order].index(c)
    return H


def H_index_matrix(order):
    if _H_index_matrix[order] is None:
        _H_index_matrix[order] = _create_H_index_matrix(order)
    return _H_index_matrix[order]


# H-matrix
def matrixify(t):
    order = get_order(t)
    TT = H_index_matrix(order)
    M = np.zeros(TT.shape)
    for i, a in np.ndenumerate(TT):
        M[i] = t[a]
    return M


# restore tensor from its H-matrix
def unmatrixify(M):
    s = LENGTH.index(M.shape[0])
    order = s * 2
    TT = H_index_matrix(order)
    t = np.zeros(LENGTH[order])
    for i in range(LENGTH[order]):
        t[i] = np.mean(M[TT == i])
    return t
