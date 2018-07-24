import logging
from os.path import isfile

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table


def load(filename, kwargs={}):
    """ This function loads NIFTI files based on the base of the filename. You
    do not need to know wether the ending is .nii or .nii.gz.

    :param filename:
    :return:
    """
    base_filename = filename.rstrip(".gz").rstrip(".nii")

    if isfile(base_filename + '.nii') and isfile(base_filename + '.nii.gz'):
        logging.warning("There are two files with the same base in the "
                        "input folder."
                        " {} is loaded.".format(filename))
        return nib.load(filename, **kwargs)

    else:
        try:
            return nib.load(base_filename + '.nii', **kwargs)
        except FileNotFoundError:
            return nib.load(base_filename + '.nii.gz', **kwargs)


def vector_norm(vectors):
    """

    :param vectors:
    :return:
    """
    vecnorm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vecnorm[vecnorm == 0] = 1.0  # avoid division by zero

    return vectors / vecnorm


def fsl_flip_sign(vectors, affine):
    """

    :param vectors:
    :return:
    """
    # Flip sign according to FSL documentation
    if np.linalg.det(affine) > 0:
        # On the last axis flip the sign of the first value (x-coordinate)
        vectors[..., 0] = -vectors[..., 0]

    return vectors


def fsl_gtab_to_worldspace(gtab, affine):
    """ Rotate bvecs into world coordinate system for data saved by FSL. According
    to the FSL documentation the sign of the x-coordinate has to be flipped if the
    determinant of the linear transformation matrix is positive.

    :param affine: The 4x4 affine matrix belonging to the provided gtabs data.
    :param gtab: A dipy GradientTable object holding gradients belonging to the given data
    :return: A dipy GradientTable object rotated to world coordinates
    """
    # Get 3x3 linear transformation part of the affine matrix
    linear = affine[0:3, 0:3]
    # Flip sign according to FSL documentation
    bvecs = gtab.bvecs
    bvecs = fsl_flip_sign(bvecs, linear)
    # Apply linear mapping to bvecs
    bvecs = np.dot(bvecs, np.transpose(linear))

    # Renormalize and return gtab
    return gradient_table(gtab.bvals, vector_norm(bvecs))


def fsl_vectors_to_worldspace(vectors):
    """

    :param vectors:
    :return:
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
