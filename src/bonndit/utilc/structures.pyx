import numpy as np

cdef TijkRefineRank1 TijkRefineRank1Parm = [1e-10,1e-6,0.3,0.9,0.5,50]
cdef TijkRefineRank TijkRefineRankkParm = [1e-10, 1e-4, 1, [1e-10,1e-6,0.3,0.9,0.5,50]]
cdef int DIM = 3
cdef double[:] order8_mult =np.array([1, 8, 8, 28, 56, 28, 56, 168, 168, 56, 70, 280, 420, 280, 70, 56, 280, 560, 560,
                                     280, 56,
                                     28, 168, 420, 560, 420, 168, 28, 8, 56, 168, 280, 280, 168, 56, 8, 1, 8, 28, 56,
                                     70, 56,
                                     28, 8, 1], dtype=np.float64)

cdef double[:] order4_mult =np.array([1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
