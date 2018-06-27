import numpy as np

from bonndit.io import vector_norm, fsl_flip_sign


def test_vector_norm_1_axis():
    """ The vector norm should be calculated for all kinds of input arrays always on the last axis
    """

    vectors = np.array((1, 2, 3))
    results = np.array((1 / np.sqrt(14), 2 / np.sqrt(14), 3 / np.sqrt(14)))

    assert (vector_norm(vectors) == results).all()


def test_vector_norm_2_axis():
    """ The vector norm should be calculated for all kinds of input arrays always on the last axis
    """

    vectors = np.array((1, 2, 3))
    results = np.array((1 / np.sqrt(14), 2 / np.sqrt(14), 3 / np.sqrt(14)))

    assert (vector_norm(vectors) == results).all()


def test_vector_norm_3_axis():
    """ The vector norm should be calculated for all kinds of input arrays always on the last axis
    """

    vectors = np.array((1, 2, 3))
    results = np.array((1 / np.sqrt(14), 2 / np.sqrt(14), 3 / np.sqrt(14)))

    assert (vector_norm(vectors) == results).all()


def test_fsl_flip_sign_2_axis():
    """
    """

    vectors = np.array([1, 1, 1, 1, 1, 1]).reshape((2, 3))
    affine = np.array([np.array([1, 0, 0]),
                       np.array([0, 1, 0]),
                       np.array([0, 0, 1])])
    results = np.array([-1, 1, 1, -1, 1, 1]).reshape((2, 3))
    assert np.all(fsl_flip_sign(vectors, affine) == results)


def test_fsl_flip_sign_3_axis():
    """
    """

    vectors = np.array([1] * 18).reshape((2, 3, 3))
    affine = np.array([np.array([1, 0, 0]),
                       np.array([0, 1, 0]),
                       np.array([0, 0, 1])])
    results = np.array([-1, 1, 1] * 6).reshape((2, 3, 3))
    assert np.all(fsl_flip_sign(vectors, affine) == results)
