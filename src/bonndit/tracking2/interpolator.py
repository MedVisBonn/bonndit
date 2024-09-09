import re
import cupy as cp
import numpy as np 
import torch as T
from cupyx.scipy.interpolate import RegularGridInterpolator
from bonndit.utils.tensor import MULTIPLIER
mult = T.tensor(MULTIPLIER[4])**0.5
class Interpolator:
    def __init__(self, data, trafo, method='linear') -> None:
        self.trafo = trafo
        data = cp.array(data)
        x = cp.linspace(0, data.shape[0] - 1, data.shape[0])
        y = cp.linspace(0, data.shape[1] - 1, data.shape[1])
        z = cp.linspace(0, data.shape[2] - 1, data.shape[2])
        self.Interpolator = RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0, method=method)
    
    def __call__(self, position):
        position = self.trafo(position)
        ret = cp.array(self.Interpolator(position))
        return ret
    
class fODFInterpolator(Interpolator):
    def __init__(self, data, trafo, method='linear') -> None:
        super().__init__(data, trafo, method)
            
        def __call__(self, position):
            position = self.trafo(position)
            ret = cp.array(self.Interpolator(position))
            #ret /= T.norm(mult[:,None] * ret, dim=1)[:, None]
            return ret
        
        
class RegularizedInterpolator(Interpolator):
    def __init__(self, data, trafo, method='linear', ) -> None:
        super().__init__(data, trafo, method)
        self.sigma_1 = 0.45
        self.sigma_2 = 1.56
        nearest = 3
        self.coordinates = cp.array([[i, j, k] for i in range(-nearest, nearest + 1) \
                                        for j in range(-nearest, nearest + 1) for k in
         range(-nearest, nearest + 1) if np.linalg.norm(np.array([i,j,k])) <= nearest]).T
        # only for scaling
        affine = cp.linalg.inv(cp.array(trafo.affine))
        affine[:3, 3] = 0
        self.dist =cp.linalg.norm((affine @ cp.vstack([self.coordinates, cp.ones([1, self.coordinates.shape[1]])]))[:3], axis=0)    
        self.coordinates = self.coordinates[:, self.dist < 3].T
        self.dist = self.dist[self.dist < 3]
        
    def __call__(self, position):
        position = self.trafo(position)
        center = cp.array(self.Interpolator(position))
        
        ret = None 
        ## Das hier broadcasten! Und evt. Interpolation ersetzen durch nearest neighbor
        for index in range(len(self.coordinates)):
            position1 = cp.array(position) + self.coordinates[index]
            inter = self.Interpolator(position1)
            dist1 = cp.linalg.norm(mult.to("cuda") * (center - inter), axis=1)
            dist = cp.exp(-(dist1**2)/self.sigma_1 - self.dist[index]**2/self.sigma_2)
            
            if ret is not None:
                ret += dist[:, None] *cp.array(self.Interpolator(position1))
                ret_dist += dist
            else: 
                ret =  dist[:, None] *cp.array(self.Interpolator(position1))
                ret_dist = dist
        ret = T.as_tensor(ret)/T.as_tensor(ret_dist[:, None])
        #ret /= T.norm(mult[None, :]*ret, dim=1)[:, None]
        return ret