from dipy.core.geometry import vec2vec_rotmat
import numpy as np
from dipy.core.gradients import gradient_table

def gtab_rotate(gtab, rot_matrix):
    """ Rotate gradients with a rotation matrix (3,3)

    :param gtab:
    :param rot_matrix:
    :return:
    """
    length = len(gtab.bvals)
    rot_bvecs = np.zeros((length, 3))
    for i in range(length):
        rot_bvecs[i, :] = np.dot(rot_matrix, gtab.bvecs[i, :])
    return gradient_table(gtab.bvals, rot_bvecs)


def gtab_reorient(gtab, old_vec, new_vec=np.array((0, 0, 1))):
    """ Rotate gradients to align the 1st eigenvector to the specified direction

    :param gtab:
    :param old_vec:
    :param new_vec:
    :return:
    """
    rot_matrix = vec2vec_rotmat(old_vec, new_vec)
    return gtab_rotate(gtab, rot_matrix)
