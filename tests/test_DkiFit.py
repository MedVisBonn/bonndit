#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nib

from bonndit import DkiFit
from .constants import KURTOSIS_RESULTS_DIR, DKI_TENSOR

fit = DkiFit.load(DKI_TENSOR)

ref_axial_diffusivity = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'da.nii')).get_data()
ref_radial_diffusivity = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'dr.nii')).get_data()
ref_mean_diffusivity = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'dm.nii')).get_data()

ref_axial_kurtosis = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'ka.nii')).get_data()
ref_radial_kurtosis = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'kr.nii')).get_data()
ref_mean_kurtosis = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'km.nii')).get_data()

ref_axial_kappa = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'kappaAxial.nii')).get_data()
ref_radial_kappa = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'kappaRadial.nii')).get_data()
ref_diamond_kappa = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'kappaDiamond.nii')).get_data()

ref_fractional_anisotropy = nib.load(
    os.path.join(KURTOSIS_RESULTS_DIR, 'fa.nii')).get_data()

ALLOWED_ERROR = 1e-10


# Test diffusivity measures
def test_DkiFit_axial_diff():
    assert ((fit.diffusivity_axial - ref_axial_diffusivity)
            / ref_axial_diffusivity < ALLOWED_ERROR).all()


def test_DkiFit_radial_diff():
    assert ((fit.diffusivity_radial - ref_radial_diffusivity)
            / ref_radial_diffusivity < ALLOWED_ERROR).all()


def test_DkiFit_mean_diff():
    assert ((fit.diffusivity_mean - ref_mean_diffusivity)
            / ref_mean_diffusivity < ALLOWED_ERROR).all()


# Test kurtosis measures
def test_DkiFit_axial_kurt():
    assert ((fit.kurtosis_axial - ref_axial_kurtosis)
            / ref_axial_kurtosis < ALLOWED_ERROR).all()


def test_DkiFit_radial_kurt():
    assert ((fit.kurtosis_radial - ref_radial_kurtosis)
            / ref_radial_kurtosis < ALLOWED_ERROR).all()


def test_DkiFit_mean_kurt():
    print((fit.kurtosis_mean - ref_mean_kurtosis) / ref_mean_kurtosis)
    assert ((fit.kurtosis_mean - ref_mean_kurtosis)
            / ref_mean_kurtosis < ALLOWED_ERROR).all()


# Test fractional anisotropy
def test_DkiFit_fractional_anisotropy():
    assert ((fit.fractional_anisotropy - ref_fractional_anisotropy)
            / ref_fractional_anisotropy < ALLOWED_ERROR).all()


# Test kappa measures
def test_DkiFit_axial_kappa():
    assert ((fit.kappa_axial - ref_axial_kappa)
            / ref_axial_kappa < ALLOWED_ERROR).all()


def test_DkiFit_radial_kappa():
    assert ((fit.kappa_radial - ref_radial_kappa)
            / ref_radial_kappa < ALLOWED_ERROR).all()


def test_DkiFit_mean_kappa():
    print((fit.kappa_diamond - ref_diamond_kappa) / ref_diamond_kappa)
    assert ((fit.kappa_diamond - ref_diamond_kappa)
            / ref_diamond_kappa < ALLOWED_ERROR).all()
