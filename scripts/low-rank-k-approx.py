#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from bonndit.directions.fodfapprox import approx_all_spherical
import nrrd

def main():
    parser = argparse.ArgumentParser(
        description='This script performs a low-rank approximation of fODFs '
                    'that are given in a higher-order tensor format, as described in '
                    '"Estimating Crossing Fibers: A Tensor Decomposition Approach" '
                    'by Schultz/Seidel (2008).')
    parser.add_argument('infile',
                        help='4D input file containing fODFs in masked higher-order tensor format (1+#fODF coefficients,x,y,z)')

    parser.add_argument('outfile',
                        help='5D output file with the approximation result (4,r,x,y,z)')
    parser.add_argument('-r', help='rank')
    args = parser.parse_args()
    # Load fODF input file
    fodfs, meta = nrrd.read(args.infile)

    if fodfs.shape[0] != 16:
        raise Exception("fodf has to be 4th order tensor.")
    if len(fodfs.shape) != 4:
        raise Exception("fodf have to be in 3d space. Hence, fodf has to be 4d.")

    data = fodfs.reshape((fodfs.shape[0], -1))
    NUM = data.shape[1]
    output = np.zeros((4, int(args.r), NUM))
    # check format:
    approx_all_spherical(output, data, fodfs, np.int(0), np.float32(0), np.int(0), np.int(args.r))

    output = output.reshape((4, int(args.r)) + fodfs.shape[1:])
    newmeta = {k: meta[k] for k in ['space', 'space origin']}
    newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
    newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], meta['space directions']))
    nrrd.write(args.outfile, np.float32(output), newmeta)


if __name__=="__main__":
    main()
