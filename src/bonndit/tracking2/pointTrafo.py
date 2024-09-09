
import numpy as np 
import cupy as cp 

class Trafo:
    def __init__(self, affine) -> None:
        self.affine = np.linalg.inv(affine)

    def __call__(self, position):
        position[..., 3] = 1
        return cp.array(self.affine @ position.T)[:3].T