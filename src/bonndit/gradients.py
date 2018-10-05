import numpy as np
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table


def gtab_rotate(gtab, rot_matrix):
    """ Rotate gradients with a rotation matrix (3,3)

    Parameters
    ----------
    gtab : dipy.data.GradientTable
        An object holding information about the applied Gradients including
        b-values and b-vectors
    rot_matrix : ndarray (3,3)
        3x3 rotation matrix

    Returns
    -------
    dipy.data.GradientTable
        Rotated GradientTable

    """
    length = len(gtab.bvals)
    rot_bvecs = np.zeros((length, 3))
    for i in range(length):
        rot_bvecs[i, :] = np.dot(rot_matrix, gtab.bvecs[i, :])
    return gradient_table(gtab.bvals, rot_bvecs)


def gtab_reorient(gtab, old_vec, new_vec=np.array((0, 0, 1))):
    """ Rotate gradients the same way you would rotate old_vec to get new_vec

    Parameters
    ----------
     gtab : dipy.data.GradientTable
        An object holding information about the applied Gradients including
        b-values and b-vectors
    old_vec : ndarray (3)
        Vector before rotation
    new_vec : ndarray (3)
        Vector after rotation. Default is the z-axis (0, 0, 1)

    Returns
    -------
    dipy.data.GradientTable
        Rotated GradientTable

    """
    rot_matrix = vec2vec_rotmat(old_vec, new_vec)
    return gtab_rotate(gtab, rot_matrix)
