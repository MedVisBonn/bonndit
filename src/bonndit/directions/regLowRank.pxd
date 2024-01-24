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
    cdef np.float64_t[:,:,:,:] _tensor
    cdef np.float64_t[:,:,:,:] _ref
    cdef np.int8_t[:,:,:] index
    cdef np.float64_t[:,:,:,:,:] _low_rank
    cdef meta
    cdef double _mu
    cdef np.int8_t rank
    cdef TijkRefineRank1Parm TijkRefineRank1Parm
    cdef TijkRefineRank TijkRefineRank
    cdef grad(self,double[:], bint,
                    double[:],
                    double[:],
                    double[:])
    cdef int minimize_single_peak(self, double[:],
                                  double[:],
                                  double[:],
                                  bint)
    cpdef write(self, outfile)
    cpdef write_reg(self, outfile)
    cpdef return_arr(self)
    cdef optimize_tensor(self, np.ndarray[np.float64_t, ndim=1],
                         double[:],
                               int,
                               np.ndarray[np.float64_t, ndim=1], float)
    cdef min_mapping_voxel(self, double[:], double[:])

    cpdef optimize_voxel(self, np.ndarray[np.int16_t, ndim=1])
    cpdef optimize(self, np.ndarray[np.float64_t, ndim=2],
                         np.ndarray[np.float64_t, ndim=1],
                         np.ndarray[np.float64_t, ndim=1],
                         int)
    cpdef optimize_parallel(self)
    cpdef void create_min_mapping(self)
