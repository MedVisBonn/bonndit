import torch as T
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator

class DirectionSelector:
    def __init__(self) -> None:
        pass
    
    def select(self, possibleDirs, oldDirs):
        pass
    
class DirectionSelectorNext(DirectionSelector):
    def __init__(self, minLen) -> None:
        self.minLen = minLen
    
    def _select(self, possibleDirs, oldDirs):
        retVals = T.zeros_like(oldDirs).to('cuda')
        retNrm = T.zeros(oldDirs.shape[0]).to('cuda')
        angles = T.zeros_like(oldDirs).to('cuda')
        nrm = T.zeros_like(oldDirs).to('cuda')
        for i in range(possibleDirs.shape[1]//3):
            ## TODO gefährlich: Kann einfach alle 0 sein. Was dann?
            nrm[:, i] = T.norm(possibleDirs[:, i*3:(i+1)*3], dim=-1)
            possibleDirs[:, i*3:(i+1)*3] /= nrm[:, i][:, None]
            angles[nrm[:, i] > self.minLen, i] = T.sum(possibleDirs[nrm[:, i] > self.minLen, i*3:(i+1)*3] * oldDirs[nrm[:, i] > self.minLen], dim=-1)
        
        retValsIdx = T.argmax(T.abs(angles), dim=-1)
        
        for i in range(3):
            ## Ist der TODO von oben damit gefixt?
            retVals[retValsIdx == i] = T.nan_to_num(T.sign(angles[retValsIdx ==i , i])[:, None] *possibleDirs[retValsIdx == i, i*3:(i+1)*3])
            retNrm[retValsIdx == i] = nrm[retValsIdx == i, i]
        return retVals, retNrm
    
    def select(self, possibleDirs, oldDirs):
        return self._select(possibleDirs, oldDirs)
    
    
    
class DirectionSelectorRegularized(DirectionSelectorNext):
    def __init__(self, minLen, ref) -> None:
        self.minLen = minLen
        x = cp.linspace(0, ref.shape[0] - 1, ref.shape[0])
        y = cp.linspace(0, ref.shape[1] - 1, ref.shape[1])
        z = cp.linspace(0, ref.shape[2] - 1, ref.shape[2])
        self.refInterpolator = RegularGridInterpolator((x, y, z), ref, bounds_error=False, fill_value=0)

    
    def select(self, possibleDirs, oldDirs, position):
        refDirs = T.tensor(self.refInterpolator(position), dtype=T.float32).to('cuda')
        refDirs[T.norm(refDirs, dim=-1) == 0] = oldDirs[T.norm(refDirs, dim=-1) == 0]
        refDirs /= T.norm(refDirs, dim=-1)[:, None]
        return self._select(possibleDirs, refDirs)
        
            
            