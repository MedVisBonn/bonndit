import numpy as np 
import torch as T
import cupy as cp
from bonndit.utils.tck_io import Tck
import numpy as np 
from tqdm import tqdm

## Todo handle to many seeds! Currently, will just lead to memory overflows

class Tracking:
    def __init__(self, dirGetter, dirSelector,  Validator, seeds, outputPath) -> None:
        self.dirGetter =  dirGetter
        self.validator = Validator
        self.dirSelector = dirSelector
        
        self.maxSteps = 600
        self.tractogram = np.zeros((2, self.maxSteps, seeds.shape[0], 4), dtype=np.float32)
        # for affine transformation
        self.tractogram[..., 3] = 1
        self.seeds = seeds
        self.writter = Tck(outputPath)
        self.writter.force = True
        self.min_length = 50
        self.writter.write()
        self.step_size = 0.9
        
    
    def _init_tractoraphy(self, i=0):
        self.tractogram[i, 0, ..., :3] = self.seeds[:,:3]
        ## If we supply initial directions, take them! Else use the dirGetter to get largest direction
        if self.seeds.shape[1] == 6:
            self.currentDirs = T.as_tensor(self.seeds[:, 3:], device='cuda')
        else:
            self.currentDirs = self.dirGetter.init_dirs(self.seeds[:, :4]).to('cuda')
        if i==1:
            self.currentDirs *= -1
        self.nextDirs = self.currentDirs.clone()
    
    def create_tractogram(self):
        for i in range(2):
            ## this should be a reset class? 
            self.validator.valid = T.ones(self.tractogram.shape[2], dtype=T.bool)
            self._init_tractoraphy(i)
            for j in tqdm(range(1, self.maxSteps)):
                positions = self.tractogram[i, j - 1, :]
                self.currentDirs = self.nextDirs.clone()
                self.nextPossibleDirs = self.dirGetter.get_directions(positions)
                self.nextDirs, nrm = self.dirSelector.select(self.nextPossibleDirs, self.currentDirs, positions)
                self.nextDirs = T.nan_to_num(self.nextDirs)
                self.validator.validate(positions, self.currentDirs, self.nextDirs, nrm, i+1)
          
                valid = self.validator.valid
                self.tractogram[i, j, valid, :3] = self.tractogram[i, j - 1, valid, :3] + (self.nextDirs[valid] * self.step_size).to('cpu').numpy()
                if self.validator.done():
                    break
    
    def save(self):
        ## Todo optimize this! 
        tractogram = np.vstack((self.tractogram[0, ::-1], self.tractogram[1, 1:]))[..., :3]
        tractogram = np.vstack((tractogram, np.ones((1, *tractogram.shape[1:]))*np.inf))
        tractogram = np.moveaxis(tractogram, 0, 1)
        tractogram = tractogram.reshape(-1, 3)
        tractogram = tractogram[np.linalg.norm(tractogram, axis=-1) > 0]
        tractogram[np.isinf(tractogram)] = np.nan
        idx = np.hstack([[-1], np.where(np.isnan(tractogram[:, 0]))[0]])
        length = idx[1:] - idx[:-1]
        # Create a boolean array based on the condition
        condition = length > self.min_length

        # Set the slices of 'include' to True where the condition is False
        include =  np.repeat(condition, length)
      #  include = np.ones(len(tractogram), dtype=bool)
      #  for i in range(1,len(idx)):
      #      include[idx[i-1]:idx[i]] = False if length[i-1] < self.min_length else True
        tractogram = tractogram[include]
        
    
        self.writter.append(tractogram, None, True)

        self.writter.close() 
                
                
                
        
    
        
    
    
        