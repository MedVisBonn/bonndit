#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, profile=False
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

cdef class TijkRefineRank1Parm:
    def __init__(self, eps_start=1e-10, eps_impr=1e-6, beta=0.3, gamma=0.9, sigma=0.5, maxTry=50):
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
        self.tensor = tensor
        self.ref = ref
        self.index = np.zeros(tensor.shape[1:], dtype= np.int8)
        self.low_rank = np.zeros((4,3, *tensor.shape[1:]))
        self.ret = np.zeros((4,3, *tensor.shape[1:]))

        self.mu = mu
        self.rank = 3
        self.TijkRefineRank1Parm = TijkRefineRank1Parm()
        self.TijkRefineRank = TijkRefineRank()
        self.optimize_parallel(False)

        self.create_min_mapping()

    cpdef write(self, outfile):
        newmeta = {k: self.meta[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], self.meta['space directions']))
        nrrd.write(outfile, np.float32(self.ret), newmeta)



    cdef void create_min_mapping(self):
        ### Finds the closest low-rank direction in terms of angle to the refenrence direction for optimization.
        # If no reference direction is set: Set the respective index to -1
        results = np.zeros((3, self.low_rank.shape[2], self.low_rank.shape[3], self.low_rank.shape[4]))
        for i in range(3):
            results[i] = np.sum(np.asarray(self.low_rank[1:, i]) * np.asarray(self.ref), axis=0)

            results[i] = np.degrees(np.arccos(results[i], where=results[i]!=0))
            results[i] = np.abs((results[i] > 90) * 180 - results[i])
        index_out = results[0] == 0

        self.index = np.argmin(results, axis=0).astype(np.int8)
       # self.index[index_out] = -1
        self.angles = np.min(results, axis=0)
       # self.angles[index_out] = -1


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
        cblas_dscal(3, 4, &res[0], 1)

        if reg:
            # res += self.mu * ref
            cblas_daxpy(3, self.mu, &ref[0], 1, &res[0], 1)

        #res -= np.dot(res, direction[1:]) * direction[1:]
        #print(np.array(res) - np.dot(res, direction[1:]) * np.array(direction[1:]))
        #print('old scale', np.dot(res, direction[1:])) 
        cdef double scale = cblas_ddot(3, &res[0], 1, &direction[1], self.rank)
        #print('new scale', scale)
        cblas_daxpy(3, -scale, &direction[1], self.rank, &res[0], 1)
        #print("new", np.array(res))



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
        cdef double iso = hota_mean(tens)

        cdef int i, armijoct,  k=tens.shape[0]
        cdef double anisonorm, anisonorminv, alpha, beta, oldval, val
        cdef double[:] isoten = np.zeros(15, dtype=np.float64)
        cdef double[:] anisoten = np.zeros(15, dtype=np.float64)
        cdef double[:] testv = np.zeros(3, dtype=np.float64)
        cdef double[:] der = np.zeros(3, dtype=np.float64)


        hota_sym_make_iso(isoten, iso)
        # anisoten = tens - isoten
        cblas_dcopy(15, &tens[0], 1, &anisoten[0], 1)
        cblas_daxpy(15, -1, &isoten[0], 1,  &anisoten[0], 1)
        #print("old anisoten", np.array(tens) - np.array(isoten))
        #print("new", np.array(anisoten))

        anisonorm = hota_sym_norm(anisoten)
        if anisonorm < self.TijkRefineRank1Parm.eps_start:
            return 1
        else:
            anisonorminv = 1/anisonorm
        alpha = beta = self.TijkRefineRank1Parm.beta*anisonorminv
        oldval = v[0] - iso
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
                cblas_dcopy(3, &v[1], self.rank, &testv[0], 1)
                cblas_daxpy(3, sign*alpha, &der[0], 1, &testv[0], 1)
                #print("new testv", np.array(testv))
                # np.linalg.norm(testv)
                dist = cblas_dnrm2(3, &testv[0], 1)
                # testv /= dist
                cblas_dscal(3, 1/dist, &testv[0], 1)

                # dist = 1 - np.linalg.norm(v[1:]*testv)
                dist = 1 - cblas_ddot(3, &v[1], self.rank, &testv[0], 1)**2
                val = hota_sym_s_form(anisoten, testv)
                if sign*val >= sign*oldval + self.TijkRefineRank1Parm.sigma*der_len*dist:
                    cblas_dcopy(3, &testv[0], 1, &v[1], self.rank)
                    v[0] = val + iso
                    self.grad(der, reg, anisoten, v, reference)
                    if alpha < beta:
                        alpha /= self.TijkRefineRank1Parm.gamma
                    break
                alpha *= self.TijkRefineRank1Parm.gamma
            if sign*(val-oldval) <= self.TijkRefineRank1Parm.eps_impr *anisonorm:
                break
            oldval = val
        return 1

    cpdef optimize_voxel(self, np.ndarray[np.int16_t, ndim=1] idx, bint reg):
        # Deactivate regularization
       # if reg:
       #     ref = self.ref
       # else:
       #     ref = np.zeros_like(self.ref) - 1
        cdef int i=idx[0], j=idx[1], k=idx[2]
        if self.tensor[0, i,j,k] == 0:
            return
        cdef np.ndarray[np.float64_t, ndim=2] low_rank = np.zeros((4, self.rank), dtype=np.float64)
        
        self.optimize_tensor(np.array(self.tensor[1:, i,j,k]), low_rank,
                             self.index[i,j,k],
                             np.array(self.ref[:, i,j,k]))
        cblas_dcopy(4*self.rank, &low_rank[0,0], 1, &self.ret[0,0,i,j,k], self.ret.shape[2] * self.ret.shape[3] * self.ret.shape[4])
        #self.ret[:,:, i,j,k] = low_rank

    cdef optimize_tensor(self, np.ndarray[np.float64_t, ndim=1] tensor,
                               np.ndarray[np.float64_t, ndim=2] low_rank,
                               int index,
                               np.ndarray[np.float64_t, ndim=1] ref):
        """
        Iterate over each rank and perform gradient descent for each rank.

        """
        # Calculate residual first:
        cdef np.ndarray[np.float64_t, ndim=2] ten = np.zeros((self.rank, 15), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] t = np.zeros((15), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros((4), dtype=np.float64)
        cdef float orig_norm = hota_4o3d_sym_norm(tensor) + self.mu * np.linalg.norm(low_rank[1:, index] - ref)
        for i in range(self.rank):
            hota_4o3d_sym_eval(ten[i], low_rank[0,i], low_rank[1:, i])
            tensor -= ten[i]
        cdef int k = 0
        while True:# and k < 10:
            k += 1
            res_norm = hota_4o3d_sym_norm(tensor) + self.mu * np.linalg.norm(low_rank[1:, index] - ref)
            for i in range(self.rank):
                if low_rank[0, i] == 0:
                    init_max_3d(low_rank[:1, i], low_rank[1:, i], tensor)
                else:
                    tensor += ten[i]
                    low_rank[0, i] = hota_4o3d_sym_s_form(tensor[:], low_rank[1:, i])
                    if low_rank[0, i] > 0:
                        self.minimize_single_peak(low_rank[:, i], tensor, ref, index==i)
                 #   print(np.array(low_rank[:,i]))
                if low_rank[0, i] > 0.1:
                    hota_4o3d_sym_eval(ten[i], low_rank[0, i], low_rank[1:, i])
                    tensor -= ten[i]
                else:
                    low_rank[0, i] = 0
            new_norm = hota_4o3d_sym_norm(tensor) + self.mu * np.linalg.norm(low_rank[1:, index] - ref)
            if not new_norm > self.TijkRefineRank.eps_res or res_norm - new_norm < self.TijkRefineRank.eps_impr*orig_norm:
                break

    cpdef optimize(self, np.ndarray[np.float64_t, ndim=2] ret,
                         np.ndarray[np.float64_t, ndim=1] tensor,
                         np.ndarray[np.float64_t, ndim=1] reference,
                         int idx):
        self.optimize_tensor(tensor, ret, idx, reference)


    cpdef optimize_parallel(self, bint reg):
        indices = [np.int16(x) for x in np.ndindex(self.tensor.shape[1], self.tensor.shape[2], self.tensor.shape[3]) if self.tensor[0, x[0], x[1], x[2]] > 0]

        #with ProcessPoolExecutor() as executor:
        for i in tqdm(indices):
          #  print(i)
            self.optimize_voxel(i, reg)
            #executor.map(self.optimize_voxel, indices, repeat(reg))
