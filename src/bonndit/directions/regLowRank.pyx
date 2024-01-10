from concurrent.futures import ProcessPoolExecutor
import numpy as np
cimport numpy as np

from bonndit.utilc.hota cimport *
from bonndit.utilc.lowrank cimport init_max_3d
from itertools import repeat
from numpy cimport ndarray

cdef class TijkRefineRank1Parm:
    def __init__(self, eps_start=1e-10, eps_impr=1e-8, beta=0.3, gamma=0.9, sigma=0.5, maxTry=100):
        self.eps_start=eps_start
        self.eps_impr=eps_impr
        self.beta=beta
        self.gamma=gamma
        self.sigma=sigma
        self.maxTry=maxTry

cdef class TijkRefineRank:
    def __init__(self, eps_res=1e-10, eps_impr=1e-8, pos=1, rank1_parm=TijkRefineRank1Parm()):
        self.eps_res = eps_res
        self.eps_impr = eps_impr
        self.pos = pos
        self.rank1_parm = rank1_parm


cdef class RegLowRank:
    def __init__(self, tensor, ref, mu):
        self.angles = np.zeros(tensor.shape[1:], dtype= np.int8)
        self.tensor = tensor
        self.ref = ref
        self.index = np.zeros(tensor.shape[1:], dtype= np.int8)
        self.low_rank = np.zeros((4,3, *tensor.shape[1:]))
        self.optimize_parallel(False)
        self.ret = np.zeros((4,3, *tensor.shape[1:]))
        self.mu = mu
        self.rank = 3
        self.create_min_mapping()
        self.TijkRefineRank1Parm = TijkRefineRank1Parm()
        self.TijkRefineRank = TijkRefineRank()


    cdef void create_min_mapping(self):
        ### Finds the closest low-rank direction in terms of angle to the refenrence direction for optimization.
        # If no reference direction is set: Set the respective index to -1
        results = np.zeros((3, self.low_rank.shape[1], self.low_rank.shape[2], self.low_rank.shape[3]))
        for i in range(3):
            results[i] = np.sum(np.asarray(self.low_rank[1:, i]) * np.asarray(self.ref), axis=0)

            results[i] = np.degrees(np.arccos(results[i], -1, 1, where=results[i]!=0))
            results[i] = np.abs((results[i] > 90) * 180 - results[i])
        index_out = results[0] == 0

        self.index = np.argmin(results, axis=0)
        self.index[index_out] = -1
        self.angles = np.min(results, axis=0)
        self.angles[index_out] = -1


    cdef grad(self,bint reg,
                    np.ndarray[np.float64_t, ndim=1] tensor,
                    np.ndarray[np.float64_t, ndim=1] direction,
                    np.ndarray[np.float64_t, ndim=1] ref):
        ### Calculate Surface gradient on sphere
        res = np.zeros(3)
        hota_4o3d_sym_v_form(res, tensor, direction[1:])
        res *= 4
        if reg:
            res += self.mu * ref
        res -= np.dot(res, direction[1:]) * direction[1:]
        return res


    cdef int minimize_single_peak(self, np.ndarray[np.float64_t, ndim=1] v,
                                  np.ndarray[np.float64_t, ndim=1] tens,
                                  np.ndarray[np.float64_t, ndim=1] reference,
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
        cdef np.ndarray[np.float64_t, ndim=1] isoten = np.zeros(15, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] anisoten = np.zeros(15, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] testv = np.zeros(3, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] der = np.zeros(3, dtype=np.float64)


        hota_sym_make_iso(isoten, iso)
        anisoten = tens - isoten
        anisonorm = hota_sym_norm(anisoten)
        if anisonorm < self.TijkRefineRank1Parm.eps_start:
            return 1
        else:
            anisonorminv = 1/anisonorm
        alpha = beta = self.TijkRefineRank1Parm.beta*anisonorminv
        oldval = v[0] - iso
        der = self.grad(reg, anisoten, v, reference)
        while True:
            armijoct = 0
            while True:
                armijoct += 1
                der_len=np.linalg.norm(der)
                if armijoct>self.TijkRefineRank1Parm.maxTry:
                    return 2
                # Gradienten Abstieg
                testv = v + sign*alpha * der
                dist = np.linalg.norm(testv)
                testv /= dist
                dist = 1 - np.linalg.norm(v - testv)
                val = hota_sym_s_form(anisoten, testv)
                if sign*val >= sign*oldval + self.TijkRefineRank1Parm.sigma*der_len*dist:
                    v[1:] = testv
                    v[0] = val + iso
                    der = self.grad(reg, anisoten, v, reference)
                    if alpha < beta:
                        alpha /= self.TijkRefineRank1Parm.gamma
                    break
                alpha *= self.TijkRefineRank1Parm.gamma
            if sign*(val-oldval) <= self.TijkRefineRank1Parm.eps_impr *anisonorm:
                break
            oldval = val
        return 1

    cpdef optimize_voxel(self, np.ndarray[np.int8_t, ndim=3] idx, bint reg):
        # Deactivate regularization
        if reg:
            ref = self.ref
        else:
            ref = np.zeros_like(self.ref) - 1
        cdef int i=idx[0], j=idx[1], k=idx[2]
        if self.tensor[0, i,j,k] == 0:
            return
        self.optimize_tensor(self.ret[:,:, i,j,k], self.tensor[1:, i,j,k], self.index[i,j,k], ref[:, i,j,k])

    cdef optimize_tensor(self, np.ndarray[np.float64_t, ndim=1] tensor,
                               np.ndarray[np.float64_t, ndim=2] low_rank,
                               int index,
                               np.ndarray[np.float64_t, ndim=1] ref):
        """
        Iterate over each rank and perform gradient descent for each rank.

        """
        # Calculate residual first:
        cdef np.ndarray[np.float64_t, ndim=2] ten = np.zeros((self.rank, 15), dtype=np.float64)
        for i in range(self.rank):
            hota_4o3d_sym_eval(ten[i], low_rank[0,i], low_rank[1:, i])
            tensor -= ten[i]

        while True:
            res_norm = hota_4o3d_sym_norm(tensor) + self.mu * np.linalg.norm(low_rank[1:, index] - ref)
            for i in range(3):
                if low_rank[0, i] == 0:
                    init_max_3d(low_rank[:1, i], low_rank[1:, i], tensor)
                else:
                    tensor += ten[i]
                    low_rank[0, i] = hota_4o3d_sym_s_form(tensor[:], low_rank[1:, i])
                    if low_rank[0, i] == 0:
                        continue
                    self.minimize_single_peak(low_rank[:, i], tensor, ref, index==i)
                    if low_rank[0, i] > 0:
                        hota_4o3d_sym_eval(ten[i], low_rank[0, i], low_rank[1:, i])
                        tensor -= ten[i]
                    else:
                        low_rank[0, i] = 0
            new_norm = hota_4o3d_sym_norm(tensor) + self.mu * np.linalg.norm(low_rank[1:, index] - self.ref)
            if not new_norm > self.TijkRefineRank.eps_res or abs(new_norm - res_norm) > self.TijkRefineRank.eps_impr:
                break

    cpdef optimize(self, np.ndarray[np.float64_t, ndim=2] ret,
                         np.ndarray[np.float64_t, ndim=1] tensor,
                         np.ndarray[np.float64_t, ndim=1] reference,
                         int idx):
        self.optimize_tensor(tensor, ret, idx, reference)


    cpdef optimize_parallel(self, bint reg):
        indices = list(np.ndindex(self.tensor.shape[1], self.tensor.shape[2], self.tensor.shape[3]))
        with ProcessPoolExecutor() as executor:
            executor.map(self.optimize_voxel, indices, repeat(reg))