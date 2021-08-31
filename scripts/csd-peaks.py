#!/usr/bin/python
# -*- coding: utf-8 -*-


import nrrd
import numpy as np
fodf, _ = nrrd.read('/home/johannes/UniBonn/HCP/904044/csd/fodf.nrrd')
fodf_sh  = np.reshape(fodf, (46, fodf.shape[1]*fodf.shape[2]*fodf.shape[3]))
del fodf
import bonndit.directions.csd_peaks as cs
a = cs.csd_peaks(fodf_sh, 3, 0,0)
