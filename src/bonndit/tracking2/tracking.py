import numpy as np 
import torch as T
import cupy as cp
from bonndit.utils.tck_io import Tck
import numpy as np 

## Todo handle to many seeds! Currently, will just lead to memory overflows

class Tracking:
    def __init__(self, dirGetter, dirSelector,  Validator, fodfs, affine,  seeds, outputPath) -> None:
        self.dirGetter =  dirGetter
        self.validator = Validator
        self.dirSelector = dirSelector
        self.affine = affine
        
        self.maxSteps = 300
        self.tractogram = np.zeros((2, self.maxSteps, seeds.shape[0], 4), dtype=np.float32)
        # for affine transformation
        self.tractogram[..., 3] = 1
        self._init_tractoraphy(seeds)
        self.writter = Tck(outputPath)
        self.writter.force = True
        self.writter.write()
        self.step_size = 0.5
        
    def _position_getter(self, position):
        position[..., 3] = 1
        return cp.array(np.linalg.inv(self.affine) @ position.T)[:3].T
    
    def _init_tractoraphy(self, seeds):
        self.tractogram[0, 0, ..., :3] = seeds[:,:3]
        self.tractogram[1, 0, ..., :3] = -seeds[:, :3]
        self.ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32).to('cuda')
        self.currentDirs = self.dirGetter.init_dirs(self._position_getter(seeds[:, :4])).to('cuda')
        self.nextDirs = self.currentDirs.clone()
    
    def create_tractogram(self):
        for i in range(2):
            self.validator.valid = T.ones(self.tractogram.shape[2], dtype=T.bool)
            for j in range(1, self.maxSteps):
                positions = self._position_getter(self.tractogram[i, j - 1, :])
                self.currentDirs = self.nextDirs.clone()
                self.nextPossibleDirs = self.dirGetter.get_directions(positions)
                self.nextDirs, nrm = self.dirSelector.select(self.nextPossibleDirs, self.currentDirs, positions)
                self.nextDirs = T.nan_to_num(self.nextDirs)
          #      self.validator.validate(positions, self.currentDirs, self.nextDirs, nrm)
                valid = self.validator.valid
                self.tractogram[i, j, valid, :3] = self.tractogram[i, j - 1, valid, :3] + (self.nextDirs[valid] * self.step_size).to('cpu').numpy()
                if self.validator.done():
                    break
    
    def save(self):
        ## Todo optimize this! 
        tractogram = np.vstack((self.tractogram[0, ::-1], self.tractogram[1, 1:]))[..., :3]
        for i in range(tractogram.shape[1]):
            nz = np.nonzero(tractogram[:, i])
            self.writter.append(tractogram[nz[0].min(): nz[0].max() + 1, i])
        
                
                
                
        
    
        
    
    
        