from dipy.core.gradients import gradient_table
import numpy as np
import nibabel as nib

def fsl_to_worldspace(affine, gtab):
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
    if np.linalg.det(linear) > 0:
        bvecs[:, 0] = -bvecs[:, 0]
    # Apply linear mapping to bvecs and re-normalize
    bvecs = np.dot(bvecs, np.transpose(linear))
    bvecnorm = np.linalg.norm(bvecs, axis=1)
    bvecnorm[bvecnorm == 0] = 1.0  # avoid division by zero
    bvecs = bvecs / bvecnorm[:, None]
    return gradient_table(gtab.bvals, bvecs)

def fsl_flip_signs_vec(vecs):
    """

    :param vecs:
    :return:
    """
    affine = vecs.affine
    out = vecs.get_data().copy()
    # flip the signs of vector coordinates as needed
    for i in range(3):
        if affine[i, i] < 0:
            out[:, :, :, i] = -out[:, :, :, i]

    return nib.Nifti1Image(out, vecs.affine)
