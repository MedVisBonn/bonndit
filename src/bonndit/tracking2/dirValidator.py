import torch as T
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator

class Validator:
    def __init__(self, seeds, **kwargs) -> None:
        self._valid = T.ones(seeds.shape[0], dtype=T.bool)
        self.DirValidator = DirectionValidator() if kwargs['DirValidator'] else NoneValidator()
        self.TractSegValidator = StoppingTractSegValidator(seeds, kwargs['StoppingRegions']) if kwargs['TractSegValidator'] else NoneValidator()
        
    
    def validate(self, position, currentDirs, nextDirs, length):
        self._valid = self.DirValidator.validate(**dict(
                currentDirs=currentDirs, 
                nextDirs=nextDirs, 
                length=length, 
                _valid=self._valid))
        self._valid = self.TractSegValidator.validate(**dict(position=position, _valid=self._valid))
        
        pass
    
    @property
    def valid(self):
        return self._valid
    
    @valid.getter
    def valid(self):
        return self._valid
    
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
    def __init__(self) -> None:
        self.maxAngle = 1/np.sqrt(2)
        self.minLen = 0.1
        
    def validate(self, **kwargs):
        currentDirs = kwargs['currentDirs']
        nextDirs = kwargs['nextDirs']
        length = kwargs['length']
        _valid = kwargs['_valid']
        _valid = self.validateAngles(currentDirs, nextDirs, _valid)
        _valid = self.validFODF(nextDirs, _valid)
        _valid = self.validateLength(length, _valid)
        return _valid
        
    
    def validateAngles(self, currentDirs, nextDirs, _valid):
        valids = T.sum(currentDirs * nextDirs, dim=-1) > self.maxAngle
        _valid *= valids.to('cpu')
        return _valid
    
    def validFODF(self, fodfs, _valid):
        _valid *= (T.norm(fodfs, dim=-1) > 0).to('cpu')
        return _valid
    
    def validateLength(self, length, _valid):
        _valid *= (length > self.minLen).to('cpu')
        return _valid

class StoppingTractSegValidator(NoneValidator):
    """Streamline is valid if it enters all  regions. If it exits after entering, it tractography should be stopped
    """
    def __init__(self, seeds, stoppingRegions) -> None:
        
        x = cp.linspace(0, stoppingRegions.shape[1] - 1, stoppingRegions.shape[1])
        y = cp.linspace(0, stoppingRegions.shape[2] - 1, stoppingRegions.shape[2])
        z = cp.linspace(0, stoppingRegions.shape[3] - 1, stoppingRegions.shape[3])
        self.stoppingRegions = [RegularGridInterpolator((x,y,z), stoppingRegions[i], bounds_error=False, fill_value=0) for i in range(stoppingRegions.shape[0])]
        self._stoppingRegions = T.zeros((seeds.shape[0], stoppingRegions.shape[0]), dtype=T.bool)
     
    
    def validate(self,  **kwargs):
        position = kwargs['position']
        _valid = kwargs['_valid']
        for i in range(self._stoppingRegions.shape[1]):
            self._stoppingRegions[self._stoppingRegions[:, i] == 0, i] = T.tensor(self.stoppingRegions[i](position[self._stoppingRegions[:, i] == 0]) > 0, dtype=T.bool)
        ## How to implement this? 
            _valid[self._stoppingRegions[:, i] == 1] = T.tensor(self.stoppingRegions[i](position[self._stoppingRegions[:, i] == 1]) < 0.1, dtype=T.bool)    
        return _valid
        