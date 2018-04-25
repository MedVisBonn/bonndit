#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `bonndit.gradients` module."""

from bonndit.gradients import gtab_reorient, gtab_rotate
from dipy.core.gradients import GradientTable
import numpy as np

GRADIENTS = np.array([[2500, 0, 0],
                      [0, 2500, 0],
                      [0, 0, 2500],
                      [0, 0, 0]])
gtab = GradientTable(GRADIENTS)


def test_gtab_rotate():
    rot_matrix = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    assert (gtab_rotate(gtab, rot_matrix).gradients == GRADIENTS).all()


def test_gtab_rotate2():
    rot_matrix = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [0, 0, 0]])
    assert (gtab_rotate(gtab, rot_matrix).gradients == np.array([[2500, 0, 0],
                                                                 [0, 2500, 0],
                                                                 [2500, 0, 0],
                                                                 [0, 0, 0]])).all()


def test_gtab_reorient():
    old_vec = np.array((0, 0, 1))
    new_vec = np.array((0, 0, 1))
    assert (gtab_reorient(gtab, old_vec, new_vec).gradients == GRADIENTS).all()


def test_gtab_reorient2():
    old_vec = np.array((0, 0, 1))
    new_vec = np.array((1, 0, 0))
    assert (gtab_reorient(gtab, old_vec, new_vec).gradients == np.array([[0, 0, -2500],
                                                                         [0, 2500, 0],
                                                                         [2500, 0, 0],
                                                                         [0, 0, 0]])).all()
