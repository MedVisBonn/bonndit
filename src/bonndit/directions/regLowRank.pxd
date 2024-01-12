cimport numpy as np

cdef class TijkRefineRank1Parm:
    cdef double eps_start
    cdef double eps_impr
    cdef double beta
    cdef double gamma
    cdef double sigma
    cdef double maxTry

cdef class TijkRefineRank:
    cdef double eps_res
    cdef double eps_impr
    cdef double pos
    cdef TijkRefineRank1Parm rank1_parm


cdef class RegLowRank:
    cdef np.float64_t[:,:,:] angles
    cdef np.float64_t[:,:,:,:] tensor
    cdef np.float64_t[:,:,:,:] ref
    cdef np.int8_t[:,:,:] index
    cdef np.float64_t[:,:,:,:,:] low_rank
    cdef np.float64_t[:,:,:,:,:] ret
    cdef meta
    cdef double mu
    cdef np.int8_t rank
    cdef TijkRefineRank1Parm TijkRefineRank1Parm
    cdef TijkRefineRank TijkRefineRank
    cdef void create_min_mapping(self)
    cdef grad(self,double[:], bint,
                    double[:],
                    double[:],
                    double[:])
    cdef int minimize_single_peak(self, double[:],
                                  double[:],
                                  double[:],
                                  bint)
    cpdef write(self, outfile)
    cdef optimize_tensor(self, np.ndarray[np.float64_t, ndim=1],
                               np.ndarray[np.float64_t, ndim=2],
                               int,
                               np.ndarray[np.float64_t, ndim=1])

    cpdef optimize_voxel(self, np.ndarray[np.int16_t, ndim=1], bint)
    cpdef optimize(self, np.ndarray[np.float64_t, ndim=2],
                         np.ndarray[np.float64_t, ndim=1],
                         np.ndarray[np.float64_t, ndim=1],
                         int)
    cpdef optimize_parallel(self,  bint)
