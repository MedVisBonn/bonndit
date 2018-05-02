from dipy.core.gradients import gradient_table
import numpy as np
import nibabel as nib

def fsl_to_worldspace(data, gtab):
    # we have to bring b vectors into world coordinate system
    # we will use the 3x3 linear transformation part of the affine matrix for this
    linear = data.affine[0:3, 0:3]
    # according to FSL documentation, we first have to flip the sign of the
    # x coordinate if the matrix determinant is positive
    bvecs = gtab.bvecs
    if np.linalg.det(linear) > 0:
        bvecs[:, 0] = -bvecs[:, 0]
    # now, apply the linear mapping to bvecs and re-normalize
    bvecs = np.dot(bvecs, np.transpose(linear))
    bvecnorm = np.linalg.norm(bvecs, axis=1)
    bvecnorm[bvecnorm == 0] = 1.0  # avoid division by zero
    bvecs = bvecs / bvecnorm[:, None]
    return gradient_table(gtab.bvals, bvecs)

def fsl_flip_signs_vec(vecs):
    affine = vecs.affine
    out = vecs.get_data().copy()
    # flip the signs of vector coords as needed
    for i in range(3):
        if affine[i, i] < 0:
            out[:, :, :, i] = -out[:, :, :, i]

    return nib.Nifti1Image(out, vecs.affine)
