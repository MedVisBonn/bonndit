import numpy as np

cdef TijkRefineRank1 TijkRefineRank1Parm = [1e-8,1e-6,0.3,0.9,0.5,50]
cdef TijkRefineRank TijkRefineRankkParm = [1e-8, 1e-4, 1, [1e-8,1e-6,0.3,0.9,0.5,50]]
cdef int DIM = 3


cdef double[:,:,:] dj_o4 = np.array([
    [[1.0,0.0, -0.5, -0.0,0.375],
    [0.0, -0.70710678, -0.0,0.4330127,0.],
    [0.0,0.0,0.61237244,0.0, -0.39528471],
    [0.0,0.0,0.0, -0.55901699, -0.],
    [0.0,0.0,0.0,0.0,0.52291252]
    ],
    [[0.0,0.70710678,0.0, -0.4330127, -0.],
    [0.0,0.5, -0.5, -0.125,0.375],
    [0.0,0.0, -0.5,0.39528471,0.1767767],
    [0.0,0.0,0.0,0.48412292, -0.33071891],
    [0.0,0.0,0.0,0.0, -0.46770717]
    ],
    [[0.0,0.0,0.61237244,0.0, -0.39528471],
    [0.0,0.0,0.5, -0.39528471, -0.1767767],
    [0.0,0.0,0.25, -0.5,0.25],
    [0.0,0.0,0.0, -0.30618622,0.46770717],
    [0.0,0.0,0.0,0.0,0.33071891]
    ],
    [[0.0,0.0,0.0,0.55901699,0.],
    [0.0,0.0,0.0,0.48412292, -0.33071891],
    [0.0,0.0,0.0,0.30618622, -0.46770717],
    [0.0,0.0,0.0,0.125, -0.375],
    [0.0,0.0,0.0,0.0, -0.1767767]
    ],
    [[0.0,0.0,0.0,0.0,0.52291252],
    [0.0,0.0,0.0,0.0,0.46770717],
    [0.0,0.0,0.0,0.0,0.33071891],
    [0.0,0.0,0.0,0.0,0.1767767],
    [0.0,0.0,0.0,0.0,0.0625]]])
