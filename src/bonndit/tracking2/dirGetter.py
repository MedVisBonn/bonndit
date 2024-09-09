from weakref import ref
import torch as T
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
from bonndit.models.LowRank import LowRankModel
from bonndit.utils import tensor as bdt
mult = T.tensor(bdt.MULTIPLIER[4], device='cuda')**0.5
class DirectionGetter:
    def __init__(self, regularized_model, ref):
        pass
    
    def get_directions(self, fodfs):
        pass
    
class DirectionGetterLowRank(DirectionGetter):
    def __init__(self, model, data):
        self.model = LowRankModel([15, 1000, 1000, 1000, 1000, 9], 0.01)
        self.model.load_state_dict(T.load(model))
        self.model.eval()
        self.model.cuda()
        self.data = data 
    
    @T.no_grad()
    def get_directions(self, position):
        fodfs = T.as_tensor(self.data(position), dtype=T.float32, device='cuda')
        #l = T.norm(mult * fodfs, dim=-1)
        #fodfs /= l[:, None]
        output =  self.model(fodfs)
        #output *= l[:, None]
        output[T.norm(fodfs, dim=-1) == 0] = 0
        return output
    
    def _init_dir(self, ret_vals, seeds, where):
        """ Takes seeds, interpolates the fodfs and selects the largest direction as the initial direction, where
        "where" is True.

        Args:
            ret_vals (pytorch array): n x 3 array of directions
            seeds (pytorch array): n x 3 array of seeds in index space
            where (bools): n x 1 array of bools
        """
        ret = ret_vals[where.get()].to('cuda')
        current_fodfs = self.data(seeds[where.get(), :4])
        current_fodfs = T.as_tensor(current_fodfs, dtype=T.float32, device='cuda')
        #l = T.norm(mult * current_fodfs, dim=-1)
        #current_fodfs /= l[:, None]
        output = self.model(current_fodfs)
        #output *= l[:, None]
        nrm = T.zeros((output.shape[0], 3), dtype=T.float32).to('cuda')
        for i in range(3):
            nrm[:, i] = T.norm(output[:, i*3:(i+1)*3], dim=-1) 
        nrm = T.argmax(nrm, dim=-1)
        for i in range(3):
            ret[nrm ==i] = output[nrm == i, i*3:(i+1)*3]
        ret_vals[where.get()] = ret.to('cpu').detach()
        return ret_vals
    
    def init_dirs(self, seeds):
        ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32)
        where = cp.ones(seeds.shape[0], dtype=bool)
        return self._init_dir(ret_vals, seeds, where)
    
    ## TODO this has to be implemented!
  #  def nnls(self, position, dirs):
  #      """Takes the position and the directions and returns the nnls solution of the directions at the position.

  #      Args:
  #          position (pytorch array): n x 3 array of positions in index space
  #          dirs (pytorch array): n x 3 array of directions

  #     """
  #      fodfs = self.dataInterpolator(position)
  #      fodfs = T.tensor(fodfs.get(), dtype=T.float32).to('cuda')
  #      output = self.model(fodfs)
  #      output[T.norm(fodfs, dim=-1) == 0] = 0
  #      return T.nnls(output, dirs)
    
class DirectionGetterLowRankReg(DirectionGetterLowRank):
    def __init__(self, regularized_model, ref, model, data):
        super().__init__(model, data)
        self.regModel = LowRankModel([18, 1000, 1000, 1000, 9], 0.1)
        self.regModel.load_state_dict(T.load(regularized_model))
        self.regModel.cuda()
        self.regModel.eval()
        self.ref = ref
    
    @T.no_grad()
    def get_directions(self, position):
        ## If a reference direction is available, use it and use regularized model, else use the model
        fodfs = self.data(position)
        refs = self.ref(position)
        where = T.as_tensor(cp.linalg.norm(refs, axis=-1) < 0.1, device='cuda')
        refs = T.as_tensor(refs, dtype=T.float32, device='cuda')
        refs /= T.norm(refs, dim=-1)[:, None]
        refs = T.nan_to_num(refs)
        fodfs = T.as_tensor(fodfs, dtype=T.float32, device='cuda')
        fodfs_merged = T.hstack([fodfs[~where], refs[~where]])
        ## TODO Just works with rank 3
        output = T.zeros((fodfs.shape[0], 9), dtype=T.float32).to('cuda') 
        output[~where] =  self.regModel(fodfs_merged)
        output[where] = self.model(fodfs[where])
      #  output = self.model(fodfs)
        output[T.norm(fodfs, dim=-1) == 0] = 0
        return output
    
    def init_dirs(self, seeds):
        """Takes seeds, interopolates the reference directions, if no reference direction is available, uses 
        largest direction of the fodf as reference direction.

        Args:
            seeds (pytorch array): n x 3 array of seeds in index space
        """
        current_ref = self.ref(seeds[:, :4])
        where_ref = cp.linalg.norm(current_ref, axis=-1) > 0
        ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32)
        ret_vals[where_ref.get(), :] = T.tensor(current_ref[where_ref.get()].get(), dtype=T.float32)
        ret_vals[where_ref.get()] /= T.norm(ret_vals[where_ref.get()], dim=-1)[:, None]
        if (~where_ref).sum() != 0:
            ret_vals = self._init_dir(ret_vals, seeds, ~where_ref)
        return ret_vals