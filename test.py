from bonndit.directions import regLowRank
import nrrd
a = nrrd.read('/data/HCP/904044/ankele/fodf.nrrd')

img = a[0]
import numpy as np
b = np.zeros((3, *img.shape[1:]))
regLowRank.RegLowRank(img, b, 0)

