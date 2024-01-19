#cython: language_level=3, warn.unused=True, warn.unused_args=True, profile=False
from concurrent.futures import ProcessPoolExecutor
import numpy as np
cimport numpy as np
import nrrd
from bonndit.utilc.hota cimport *
from bonndit.utilc.lowrank cimport init_max_3d
from itertools import repeat
from numpy cimport ndarray
from tqdm import tqdm
from bonndit.utilc.blas_lapack cimport *
from libc.math cimport pow, pi, acos, floor, fabs


cdef class TijkRefineRank1Parm:
    def __init__(self, eps_start=1e-10, eps_impr=1e-6, beta=0.3, gamma=0.9, sigma=0.01, maxTry=200):
        self.eps_start=eps_start
        self.eps_impr=eps_impr
        self.beta=beta
        self.gamma=gamma
        self.sigma=sigma
        self.maxTry=maxTry

cdef class TijkRefineRank:
    def __init__(self, eps_res=1e-10, eps_impr=1e-4, pos=1, rank1_parm=TijkRefineRank1Parm()):
        self.eps_res = eps_res
        self.eps_impr = eps_impr
        self.pos = pos
        self.rank1_parm = rank1_parm


cdef class RegLowRank:
    def __init__(self, tensor, ref, mu, meta):
        self.angles = np.zeros(tensor.shape[1:], dtype= np.float64)
        self.meta = meta
        self._tensor = tensor
        self._ref = ref
        self.index = np.zeros(tensor.shape[1:], dtype= np.int8)
        self._low_rank = np.zeros((4,3, *tensor.shape[1:]))
        self._mu = mu
        self.rank = 3
        self.TijkRefineRank1Parm = TijkRefineRank1Parm()
        self.TijkRefineRank = TijkRefineRank()

    @property
    def ref(self):
        return np.array(self._ref)

    @ref.setter
    def ref(self, ref):
        self._ref = ref

    @ref.getter
    def ref(self):
        return np.array(self._ref)

    @property
    def low_rank(self):
        return np.array(self._low_rank)

    @low_rank.setter
    def low_rank(self, double[:,:,:,:,:] new_low_rank):
        self._low_rank = new_low_rank

    @low_rank.getter
    def low_rank(self):
        return np.array(self._low_rank)
    
    @property
    def tensor(self):
        return np.array(self._tensor)

    @tensor.setter
    def tensor(self, new_tensor):
        self._tensor = new_tensor

    @tensor.getter
    def tensor(self):
        return np.array(self._tensor)
    
    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, new_mu):
        self._mu = new_mu

    @mu.getter
    def mu(self):
        return self._mu

    cpdef write(self, outfile):
        newmeta = {k: self.meta[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], self.meta['space directions']))
        nrrd.write(outfile, np.float32(self._low_rank), newmeta)

    cpdef write_reg(self, outfile):
        newmeta = {k: self.meta[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], self.meta['space directions']))
        idx = np.where(np.array(self.index) >= 0)
        print(idx) 
        out = np.zeros_like(self._low_rank)
        for i,j,k in zip(*idx):
            if np.linalg.norm(self._ref[:, i,j,k]) > 0:
                out[:, 0, i,j,k] = self._low_rank[:, self.index[i,j,k], i,j,k]
        nrrd.write(outfile, np.float32(out), newmeta)



    cpdef void create_min_mapping(self):
        ### Finds the closest low-rank direction in terms of angle to the refenrence direction for optimization.
        # If no reference direction is set: Set the respective index to -1
        results = np.zeros((3, self._low_rank.shape[2], self._low_rank.shape[3], self._low_rank.shape[4]))
        mask_a = np.linalg.norm(self._ref, axis=0) > 0
        mask_b = np.linalg.norm(self._low_rank, axis=(0,1)) > 0
        mask = (mask_a.astype(int) + mask_b.astype(int)) > 1
        for i in range(3):
            results[i] = np.sum(np.asarray(self._low_rank[1:, i]) * np.asarray(self._ref), axis=0)
            results[i] = np.degrees(np.arccos(np.clip(results[i],-1,1), where=mask))
            results[i] = np.abs((results[i] > 90) * 180 - results[i])

        idx = np.argmin(results, axis=0).astype(np.int8)
        self.index = idx
        print(np.sum(idx), np.sum(self.index), np.sum(mask))
        angles = np.min(results, axis=0)
        angles[~mask] = -1
        self.angles = angles 


    cdef grad(self,double[:] res,
                    bint reg,
                    double[:] tensor,
                    double[:] direction,
                    double[:] ref):
        ### Calculate Surface gradient on sphere
        #cdef double[:] res = np.zeros(3)
        cblas_dscal(3, 0, &res[0], 1)
        hota_4o3d_sym_v_form(res, tensor, direction[1:])
        # res *= 4
        cblas_dscal(3, 4*direction[0], &res[0], 1)
        if reg:
            # res += self._mu * ref
            cblas_daxpy(3, self._mu, &ref[0], 1, &res[0], 1)

        #res -= np.dot(res, direction[1:]) * direction[1:]
        #print(np.array(res) - np.dot(res, direction[1:]) * np.array(direction[1:]))
        #print('old scale', np.dot(res, direction[1:])) 
        cdef double scale = cblas_ddot(3, &res[0], 1, &direction[1], 1)
        #print('new scale', scale)
        cblas_daxpy(3, -scale, &direction[1], 1, &res[0], 1)
        #print("new", np.array(res))za
    cpdef return_arr(self):
        return np.asarray(self.index)


    cdef int minimize_single_peak(self, double[:] v,
                                  double[:] tens,
                                  double[:] reference,
                                  bint reg,
                                  ):
        """
        Gradient descent with armijo stepsize
        """

        hota_sym_make_iso = hota_4o3d_sym_make_iso
        hota_sym_norm = hota_4o3d_sym_norm
        _sym_grad = self.grad
        hota_mean = hota_4o3d_mean
        hota_sym_s_form = hota_4o3d_sym_s_form

        cdef int sign = 1 if v[0]>0 else -1
        if reg:
            sign = 1 if  v[0]**2  + 2 *self._mu * cblas_ddot(3, &v[1], 1, &reference[0], 1) > 0 else -1
        cdef double iso = hota_mean(tens)

        cdef int i, armijoct,  k=tens.shape[0]
        cdef double anisonorm, anisonorminv, alpha, beta, oldval, val
        cdef double[:] isoten = np.zeros(15, dtype=np.float64)
        cdef double[:] anisoten = np.zeros(15, dtype=np.float64)
        cdef double[:] testv = np.zeros(3, dtype=np.float64)
        cdef double[:] der = np.zeros(3, dtype=np.float64)


        hota_sym_make_iso(isoten, iso)
        #anisoten = tens - isoten
        cblas_dcopy(15, &tens[0], 1, &anisoten[0], 1)
        #cblas_daxpy(15, -1, &isoten[0], 1,  &anisoten[0], 1)
        #print("old anisoten", np.array(tens) - np.array(isoten))
        #print("new", np.array(anisoten))

        anisonorm = hota_sym_norm(anisoten)
        if anisonorm < self.TijkRefineRank1Parm.eps_start:
            return 1
        else:
            anisonorminv = 1/anisonorm
        alpha = beta = self.TijkRefineRank1Parm.beta*anisonorminv
        if reg:
            oldval =  v[0]**2 + 2 *self._mu * cblas_ddot(3, &v[1], 1, &reference[0], 1)
        else:
            oldval = v[0]
        self.grad(der, reg, anisoten, v, reference)
        while True:
            armijoct = 0
            while True:
                armijoct += 1
                der_len=cblas_dnrm2(3, &der[0], 1)
                if armijoct>self.TijkRefineRank1Parm.maxTry:
                    return 2
                # Gradienten Abstieg testv = v[1:] + sign*alpha * der
                #print("old testv", np.array(v[1:]) + sign*alpha * np.array(der))
                cblas_dcopy(3, &v[1], 1, &testv[0], 1)
                cblas_daxpy(3, sign*alpha, &der[0], 1, &testv[0], 1)
                #print("new testv", np.array(testv))
                # np.linalg.norm(testv)
                dist = cblas_dnrm2(3, &testv[0], 1)
                # testv /= dist
                cblas_dscal(3, 1/dist, &testv[0], 1)

                # dist = 1 - np.linalg.norm(v[1:]*testv)
                dist = 1 - cblas_ddot(3, &v[1], 1, &testv[0], 1)**2
                if reg:
                    val = hota_sym_s_form(anisoten, testv)**2 +  2 *self._mu * cblas_ddot(3, &testv[0], 1, &reference[0], 1)
                    print(val, oldval, cblas_ddot(3, &testv[0], 1, &reference[0], 1))
                    #print(val, oldval, cblas_ddot(3, &testv[0], 1, &reference[0], 1))

                else:
                    val = hota_sym_s_form(anisoten, testv)
                if sign*val >= sign*oldval + self.TijkRefineRank1Parm.sigma*der_len*dist:
                    cblas_dcopy(3, &testv[0], 1, &v[1], 1)
                    v[0] = hota_sym_s_form(anisoten, testv)
                    self.grad(der, reg, anisoten, v, reference)
                    if alpha < beta:
                        alpha /= self.TijkRefineRank1Parm.gamma
                    break
                alpha *= self.TijkRefineRank1Parm.gamma
            if sign*(val-oldval) <= self.TijkRefineRank1Parm.eps_impr *anisonorm:
                break
            oldval = val
        return 1

    cpdef optimize_voxel(self, np.ndarray[np.int16_t, ndim=1] idx):
        # Deactivate regularization

        cdef int i=idx[0], j=idx[1], k=idx[2]
        if self._tensor[0, i,j,k] == 0:
            return
        # Do not reoptimize voxels, which are already optimized
        if self._mu > 0 and np.linalg.norm(self._ref[0,i,j,k]) == 0:
            return

        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] low_rank = np.zeros((4 *  self.rank, ), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] ref = np.zeros((3, ), dtype=np.float64)
        cdef int z=0
        for z in range(3):
            cblas_dcopy(4, &self._low_rank[0,z,i,j,k],self._low_rank.shape[1]* self._low_rank.shape[2] * self._low_rank.shape[3] * self._low_rank.shape[4], &low_rank[4*z], 1)
        cblas_dcopy(3, &self._ref[0,i,j,k], self._ref.shape[1] * self._ref.shape[2] * self._ref.shape[3], &ref[0], 1)
        for z in range(3):
            low_rank[4*z:4*(z+1)] = np.copy(self._low_rank[:,z,i,j,k])
        self.index[i,j,k] = self.optimize_tensor(np.array(self._tensor[1:, i,j,k]), low_rank,
                             self.index[i,j,k],
                             np.array(ref), self._mu if np.linalg.norm(ref) else 0)

        for z in range(self.rank):
            cblas_dcopy(4, &low_rank[4*z], 1, &self._low_rank[0,z,i,j,k], self._low_rank.shape[1] * self._low_rank.shape[2] * self._low_rank.shape[3] * self._low_rank.shape[4])

    cdef min_mapping_voxel(self, double[:] low_rank, double[:] ref):
        index = 0
        v_max = 0
        for i in range(self.rank):
            if low_rank[4*i] > 0:
                v = abs(cblas_ddot(3, &low_rank[4*i + 1], 1, &ref[0], 1))
                if v > v_max: 
                    index = i 
                    v_max = v
        return index

    cdef optimize_tensor(self, np.ndarray[np.float64_t, ndim=1] tensor,
                               np.ndarray[np.float64_t, ndim=1] low_rank,
                               int index,
                               np.ndarray[np.float64_t, ndim=1] ref, float mu):
        """
        Iterate over each rank and perform gradient descent for each rank.

        """
        # Calculate residual first:
            #67971print("new")
        cdef np.ndarray[np.float64_t, ndim=2] ten = np.zeros((self.rank, 15), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] t = np.zeros((15), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros((4), dtype=np.float64)
        cdef float orig_norm = hota_4o3d_sym_norm(tensor) + mu * ( 1- fabs(np.dot(low_rank[index*4+1: index*4 +4],ref)))
        for i in range(self.rank):
            hota_4o3d_sym_eval(ten[i], low_rank[i*4], low_rank[index*4+1: index*4 +4])
            tensor -= ten[i]
        cdef int k = 0
        cdef int index_changed=0
        while True: # and k < 10:
            index_changed=0
            k += 1
            res_norm = hota_4o3d_sym_norm(tensor) + mu * ( 1- fabs(np.dot(low_rank[index*4+1 : index*4 +4],ref)))
            if cblas_ddot(3, &low_rank[index*4+1], 1, &ref[0], 1) < 0 and mu>0:
                cblas_dscal(3, -1, &low_rank[index*4+1], 1)

            for i in range(self.rank):
                ## If the direction is 0, i.e. not initalized, do so
                ## else performe update
                if low_rank[i*4] == 0:
                    init_max_3d(low_rank[i*4: i*4 + 1], low_rank[i*4 + 1: i*4 + 4], tensor)
                    if (index==i and mu>0):
                        index=self.min_mapping_voxel(low_rank, ref)
                        index_changed = 1
                else:
                    tensor += ten[i]
                    low_rank[i*4] = hota_4o3d_sym_s_form(tensor[:], low_rank[i*4 + 1:i*4 + 4])
                    self.minimize_single_peak(low_rank[i*4: i*4 + 4], tensor, ref, index==i and mu > 0)
                if low_rank[i*4] > 0:
                    hota_4o3d_sym_eval(ten[i], low_rank[i*4], low_rank[i*4 +1: i*4 + 4])
                    tensor -= ten[i]
                else:
                     low_rank[i*4] = 0
            new_norm = hota_4o3d_sym_norm(tensor) + mu * (1- fabs(np.dot(low_rank[index*4+1 : index*4 +4],ref)))
            if not index_changed==1 and (new_norm <= self.TijkRefineRank.eps_res or res_norm - new_norm < self.TijkRefineRank.eps_impr*orig_norm):
                break
        return index

    cpdef optimize(self, np.ndarray[np.float64_t, ndim=2] ret,
                         np.ndarray[np.float64_t, ndim=1] tensor,
                         np.ndarray[np.float64_t, ndim=1] reference,
                         int idx):
        self.optimize_tensor(tensor, ret, idx, reference,self._mu)


    cpdef optimize_parallel(self):
        indices = [np.int16(x) for x in np.ndindex(self._tensor.shape[1], self._tensor.shape[2], self._tensor.shape[3]) if self._tensor[0, x[0], x[1], x[2]] > 0]


        for i in tqdm(indices):
          #  print(i)
            self.optimize_voxel(i)
            #executor.map(self.optimize_voxel, indices, repeat(reg))

