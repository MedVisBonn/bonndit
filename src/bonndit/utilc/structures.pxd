#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

# These parameters control optimization of rank-1 approximations
ctypedef struct TijkRefineRank1:
    double eps_start
    double eps_impr
    double beta
    double gamma
    double sigma
    int maxTry


ctypedef struct TijkRefineRank:
    double eps_res
    double eps_impr
    double pos
    TijkRefineRank1 rank1_parm


# Holds information about (and functions needed for processing) a
# specific type of tensor
ctypedef struct TijkType:
    int order
    int dim
    int num

cdef TijkRefineRank1 TijkRefineRank1Parm
cdef TijkRefineRank TijkRefineRankkParm
cdef int DIM
cdef double[:] order4_mult
cdef double[:] order8_mult
