import logging
from os.path import isfile

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table


def load(filename, **kwargs):
    """ Load NIFTI files based on the base of the filename.

    Using this function it does not matter whether your data is in .nii or in
    .nii.gz format as long as the base of the filename is correct the data is
    going to be loaded.

    Parameters
    ----------
    filename : str
        Name of the file
    kwargs : dict
        Keyword arguments for nib.load()

    Returns
    -------
    nibabel Image Object

    """
    base_filename = filename.rstrip(".gz").rstrip(".nii")

    if isfile(base_filename + '.nii') and isfile(base_filename + '.nii.gz'):
        logging.warning("There are two files with the same base in the "
                        "input folder."
                        " {} is loaded.".format(filename))
        return nib.load(filename, **kwargs)

    else:
        try:
            return nib.load(base_filename + '.nii.gz', **kwargs)
        except FileNotFoundError:
            try:
                return nib.load(base_filename + '.nii', **kwargs)
            except FileNotFoundError:
                msg = 'No such file: "{}"'.format(filename)
                raise FileNotFoundError(msg)


def vector_norm(vectors):
    """ Calculate the norm for a given vector and return it.

    Parameters
    ----------
    vectors : ndarray
        Vector to be normed

    Returns
    -------
    ndarray
        Vector of length 1

    """
    vecnorm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vecnorm[vecnorm == 0] = 1.0  # avoid division by zero

    return vectors / vecnorm


def fsl_flip_sign(vectors, affine):
    """ Flip the sign of the x-axis if the affines determinant is larger than 0

    Parameters
    ----------
    vectors : ndarray
        Vectors on which to apply the sign flip
    affine : ndarray
        Linear transformation part of the affine (3x3 matrix)

    Returns
    -------
    ndarray
    """
    # Flip sign according to FSL documentation
    if np.linalg.det(affine) > 0:
        # On the last axis flip the sign of the first value (x-coordinate)
        vectors[..., 0] = -vectors[..., 0]

    return vectors


def fsl_gtab_to_worldspace(gtab, affine):
    """ Rotate bvecs into world coordinate system for data saved by FSL.

    According to the FSL documentation the sign of the x-coordinate has to be
    flipped if the determinant of the linear transformation matrix is positive.

    Parameters
    ----------
    gtab : dipy.data.GradientTable
        An object holding information about the applied Gradients including
        b-values and b-vectors
    affine : ndarray
        The 4x4 affine matrix belonging to the provided gtabs data.

    Returns
    -------
    dipy.data.GradientTable
        GradientTable rotated to world coordinates
    """
    # Get 3x3 linear transformation part of the affine matrix
    linear = affine[0:3, 0:3]
    # Flip sign according to FSL documentation
    bvecs = gtab.bvecs
    bvecs = fsl_flip_sign(bvecs, linear)
    # Apply linear mapping to bvecs
    bvecs = np.dot(bvecs, np.transpose(linear))

    # Renormalize and return gtab
    return gradient_table(gtab.bvals, vector_norm(bvecs),
                          b0_threshold=gtab.b0_threshold)


def fsl_vectors_to_worldspace(vectors):
    """ Rotate vectors into world coordinate system for data saved by FSL

    Parameters
    ----------
    vectors : ``Spatial Image``
        nibabel Image object holding an affine and vectors to be transformed

    Returns
    -------
    ``Spatial Image``
        Image object holding the transformed vectors and the original affine

    """
    affine = vectors.affine
    vecs = vectors.get_data()

    # Get 3x3 linear transformation part of the affine matrix
    linear = affine[0:3, 0:3]
    vecs = fsl_flip_sign(vecs, linear)

    # Apply linear mapping to vecs
    vecs = np.dot(vecs, np.transpose(linear))

    # Renormalize and return as nifti
    return nib.Nifti1Image(vector_norm(vecs), affine)
