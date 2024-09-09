import torch as T
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
from bonndit.utils.tensor import MULTIPLIER
mult = T.tensor(MULTIPLIER[4], device="cuda")**0.5
class Validator:
    def __init__(self, seeds, wmregion, **kwargs) -> None:
        self._valid = T.ones(seeds.shape[0], dtype=T.bool)
        self.wm = wmregion
        self.DirValidator = DirectionValidator(kwargs['fodfs']) if kwargs['DirValidator'] else NoneValidator()
        self.TractSegValidator = StoppingTractSegValidator(seeds, kwargs['StoppingRegions']) if kwargs['TractSegValidator'] else NoneValidator()
        
    
    def validate(self, position, currentDirs, nextDirs, length, j):
        self._valid = self.DirValidator.validate(**dict(
                currentDirs=currentDirs, 
                nextDirs=nextDirs, 
                position=position,
                length=length, 
                _valid=self._valid))
       # self._valid = self.TractSegValidator.validate(**dict(position=position, _valid=self._valid, j=j))
        self._valid *= T.as_tensor(self.wm(position) > 0.15, device="cuda")
        self._valid = self._valid
        pass
    
    @property
    def valid(self):
        return self._valid.to('cpu')
    
    @valid.getter
    def valid(self):
        return self._valid.to('cpu')
    
    @valid.setter
    def valid(self, value):
        self._valid = value
        
    def done(self):
        return self._valid.sum() == 0
    
class NoneValidator():
    def __init__(self) -> None:
        pass
    
    def validate(self, **kwargs):
        return kwargs['_valid']
    
    
class DirectionValidator():
    def __init__(self, fodfs) -> None:
        self.maxAngle = 0.5
        self.minLen = 0.05
        self.fodfs = fodfs
        #self.angles = T.zeros((seed_shape[0],30), device="cuda")
        self.i = 0
        
    def validate(self, **kwargs):
        currentDirs = kwargs['currentDirs']
        position = kwargs['position']
        
        nextDirs = kwargs['nextDirs']
        _valid = T.as_tensor(kwargs['_valid'], device="cuda")
        _valid = self.validateDir(nextDirs, _valid)
        _valid = self.validateAngles(currentDirs, nextDirs, _valid)
     #   _valid = self.validFODF(position, _valid)
     #   _valid = self.validateLength(length, _valid)
        #self._valid = _valid
        return _valid
        
    
    def validateAngles(self, currentDirs, nextDirs, _valid):
       # """Checks the current angle and the angle over the last 30 steps. If this angle is smaller than the maxAngle, the streamline is invalid"""
       # self.angles[:, self.i%30] = T.abs(T.sum(currentDirs/T.norm(currentDirs, dim=1)[:, None] * nextDirs/T.norm(nextDirs, dim=1)[:, None], dim=-1))
        _valid *= T.abs(T.sum(currentDirs/T.norm(currentDirs, dim=1)[:, None] * nextDirs/T.norm(nextDirs, dim=1)[:, None], dim=-1)) > self.maxAngle
        #self.i += 1
        #_valid *= T.sum(self.angles, dim=-1) >  130
        return _valid
    
    def validateDir(self, nextDirs, _valid):
        _valid *= ((T.norm(nextDirs, dim=-1) - 1) < 1e-7)
        return _valid
    
    def validFODF(self, position, _valid):
        fodfs = T.as_tensor(self.fodfs(position), device='cuda')
        _valid *= (T.norm(mult[None] * fodfs, dim=-1) > 0.1)
        return _valid
    
    def validateLength(self, length, _valid):
        _valid *= (length > self.minLen).to('cpu')
        return _valid

class StoppingTractSegValidator(NoneValidator):
    """Streamline is valid if it enters all  regions. If it exits after entering, it tractography should be stopped
    """
    def __init__(self, seeds, stoppingRegions) -> None:
        
        
        self.stoppingRegions = stoppingRegions
        self._stoppingRegions = T.zeros(seeds.shape[0])
        self._prev = []

     
    
    def validate(self,  **kwargs):
        return kwargs['_valid']

    def stopping_valid(self, streamline):
        endings = T.vstack([T.as_tensor(self.stoppingRegions[i](streamline)) for i in range(len(self.stoppingRegions))])
        nz = [np.nonzero(endings[0]), np.nonzero(endings[1])]
        if len(nz[0]) == 0 or len(nz[1]) == 0:
            return None       
        return streamline[min(nz[1].min(), nz[0].min()): max(nz[1].max(), nz[0].max()) + 1, :3]
   #     return streamline[:,:3]

    #def stopping_global(self, tractogram):
    #    tractogram = np.vstack(tractogram)
    #    endings = np.vstack([np.array(self.stoppingRegions[i](tractogram).get()) for i in range(len(self.stoppingRegions))])
    #    nz = np.nonzero(endings)
    #    include = np.zeros(tractogram.shape[0], dtype=bool)
    #    unique = np.unique(nz[])
    #    for i in range(len(unique)):
    #        include[nz] = True
    #    include[(nz[0], nz[1])] = True
    #    return include 

