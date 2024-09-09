
import numpy as np
import cupy as cp


class Trafo:
    def __init__(self, affine) -> None:
        # Invert the image affine to map world positions into voxel space.
        self.affine = np.linalg.inv(affine)

    def __call__(self, position):
        # Use homogeneous points so the affine translation is included.
        position[..., 3] = 1
        return cp.array(self.affine @ position.T)[:3].T
