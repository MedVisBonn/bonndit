import re
import torch as T
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
import numpy as np 

class DirectionSelector:
    def __init__(self, min_len) -> None:
        self.min_len = 0.05 #min_len
    
    def select(self, possibleDirs, oldDirs):
        pass
    
class DirectionSelectorNext(DirectionSelector):    
    def _select(self, possibleDirs, oldDirs):
        oldDirs /= T.norm(oldDirs, dim=-1)[:, None]
        angles = T.zeros_like(oldDirs).to('cuda')
        nrm = T.zeros_like(oldDirs).to('cuda')
        for i in range(3):
            ## TODO gefährlich: Kann einfach alle 0 sein. Was dann?
            nrm[:, i] = T.norm(possibleDirs[:, i*3:(i+1)*3], dim=-1)
            possibleDirs[:, i*3:(i+1)*3] /= nrm[:, i][:, None]
            angles[nrm[:, i] > self.min_len, i] = T.sum(possibleDirs[nrm[:, i] > self.min_len, i*3:(i+1)*3] * oldDirs[nrm[:, i] > self.min_len], dim=-1)
        ## Nan values become value 0, which is equivalent to an angle of 90 degrees 

        angles = T.nan_to_num(angles)
        return angles, nrm 
       
    
    def select(self, possibleDirs, oldDirs, position):
        retVals = T.zeros_like(oldDirs).to('cuda')
        retNrm = T.zeros(oldDirs.shape[0]).to('cuda')
        angles, nrm =  self._select(possibleDirs, oldDirs)
        retValsIdx = T.argmax(T.abs(angles), dim=-1)
    
        for i in range(3):
            retVals[retValsIdx == i] = T.sign(angles[retValsIdx ==i , i])[:, None] * possibleDirs[retValsIdx == i, i*3:(i+1)*3]
            retNrm[retValsIdx == i] = nrm[retValsIdx == i, i]
    
        return retVals, retNrm 
     
class DirectionSelectorProbabilistic(DirectionSelectorNext):
    def __init__(self, min_len) -> None:
        super().__init__(min_len)
        self.last_dir = None
    
    def select(self, possibleDirs, oldDirs, position):
        retVals = T.zeros_like(oldDirs).to('cuda')
        retNrm = T.zeros(oldDirs.shape[0]).to('cuda')
        angles, nrm =  self._select(possibleDirs, oldDirs)
        if self.last_dir is None:
            self.last_nrm = nrm
        probs = T.exp(-T.abs(nrm  - self.last_nrm)/2)*T.cos((3/np.sqrt(2*T.pi)*T.arccos(abs(angles)-1e-7))**2)**6*(abs(angles) > 0.5)
        #probs = T.cos((3/np.sqrt(2*T.pi)*T.arccos(abs(angles)-1e-7))**2)**6*(abs(angles) > 0.5)
        probs /= T.sum(probs, dim=-1)[:, None]
        probs = T.cumsum(T.nan_to_num(probs), dim=1)
        probs = T.hstack([T.zeros([probs.shape[0], 1]).to('cuda'), probs])
        rand = T.rand(probs.shape[0], device='cuda')
        for i in range(3):
            idx = (probs[:, i] <= rand) * (probs[:, i+1] >= rand)
            retVals[idx] = T.sign(angles[idx , i])[:, None] *possibleDirs[idx, i*3:(i+1)*3]
            retNrm[idx] = nrm[idx, i]
        self.last_nrm = retNrm
        return retVals, retNrm 
        
    
    
    
    
class DirectionSelectorRegularized(DirectionSelectorNext):
    def __init__(self, ref, min_len) -> None:
        super().__init__(min_len)
        self.ref = ref

    
    def select(self, possibleDirs, oldDirs, position):
        retVals = T.zeros_like(oldDirs).to('cuda')
        refDirs = T.as_tensor(self.ref(position), dtype=T.float32, device='cuda')
        refDirs[T.norm(refDirs, dim=-1) == 0] = oldDirs[T.norm(refDirs, dim=-1) == 0]
        refDirs /= T.norm(refDirs, dim=-1)[:, None]
         
        retNrm = T.zeros(oldDirs.shape[0]).to('cuda')
        angles, nrm =  self._select(possibleDirs, refDirs)
        anglesIdx = T.argmax(T.abs(angles), dim=-1) 
        for i in range(3):
            sign = T.sign(T.sum(possibleDirs[anglesIdx == i, i*3:(i+1)*3] * oldDirs[anglesIdx == i], dim=-1))
            retVals[anglesIdx == i] = sign[:, None] * possibleDirs[anglesIdx == i, i*3:(i+1)*3]
            retNrm[anglesIdx == i] = nrm[anglesIdx == i, i]
        return retVals, retNrm 
    
 
        
            
            