import numpy as np
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table


def gtab_rotate(gtab, rot_matrix):
    """ Rotate gradients with a rotation matrix (3,3)

    :param gtab: dipy GradientTable
    :param rot_matrix: 3x3 rotation matrix
    :return: rotated dipy GradientTable
    """
    length = len(gtab.bvals)
    rot_bvecs = np.zeros((length, 3))
    for i in range(length):
        rot_bvecs[i, :] = np.dot(rot_matrix, gtab.bvecs[i, :])
    return gradient_table(gtab.bvals, rot_bvecs)


def gtab_reorient(gtab, old_vec, new_vec=np.array((0, 0, 1))):
    """ Rotate gradients such that a given vector will be mapped to another given vector, by default the z-axis

    :param gtab: dipy GradientTable
    :param old_vec: vector before rotation
    :param new_vec: vector after rotation
    :return: rotated dipy GradientTable
    """
    rot_matrix = vec2vec_rotmat(old_vec, new_vec)
    return gtab_rotate(gtab, rot_matrix)
