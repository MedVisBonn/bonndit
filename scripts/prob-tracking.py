from bonndit.tracking.tracking_prob import tracking_all
import argparse
import nrrd
import numpy as np
from plyfile import PlyElement, PlyData

path_tree = []


def main():
    parser = argparse.ArgumentParser(
        description='This script performs tracking along a multi vector field as described in '
                    'Reducing Model Uncertainty in Crossing Fiber Tractography, Gruen et al. (2021)')

    parser.add_argument('-i', help='5D (4,3,x,y,z) Multivectorfield, where the first dimension gives the length '
                                   'and the direction of the vector, the second dimension denots different directions')

    parser.add_argument('-wm', help='WM Mask')
    parser.add_argument('-wmmin', help='Minimum WM density befor tracking stops')
    parser.add_argument('-s', help='Seedspointfile: Each row denots a seedpoint 3 columns an initial direction can '
                                   'also be given as 3 additional columns. Columns should be seperated by whitespace.')
    parser.add_argument('-sw', help='Stepwidth for Eulerintegration')
    parser.add_argument('-o', help='Outfile is in ply format.')
    parser.add_argument('-mtlength', help='Maximum track steps.')
    parser.add_argument('-samples', help='Samples per seed.')
    parser.add_argument('-var', help='Variance.')
    parser.add_argument('-exp', help='Expectation')
    parser.add_argument('-interpolation', '--interpolation',
                        help='decide between FACT interpolation and Trilinear interpolation. Default is Trilinear.',
                        default='Trilinear')
    parser.add_argument('-integration', '--integration', help='Decide between Euler integration and Euler integration. '
                                                              'Default is Euler integration.', default='Euler')
    parser.add_argument('-prob', '--prob', help='Decide between Laplacian and Gaussian. '
                                                'Default is Gaussian.', default='Gaussian')
    args = parser.parse_args()
    # Open files and do various error checking
    vector_field, _ = nrrd.read(args.i)
    if vector_field.shape[0] != 4:
        raise Exception("Wrong dimension on first axis. Has to contain 4 values.")
    if vector_field.shape[1] != 3:
        raise Exception("Wrong dimension on second axis. Has to contain 3 values.")
    if len(vector_field.shape) != 5:
        raise Exception("The input multivector field has to have 5 dimensions.")

    wm_mask, meta = nrrd.read(args.wm)
    if vector_field.shape[2:] != wm_mask.shape:
        raise Exception("Vectorfield (x,y,z) and wm mask have to have same dimensions.")
    seeds = open(args.s)
    seeds = [list(map(float, point.split())) for point in seeds]
    if [1 for seed in seeds if len(seed) not in [3,6]]:
        raise Exception("The seedfile is corrupted. Has to have ether 3 or 6 entries per row.")
    seeds = np.float64(seeds)
    paths, paths_len = tracking_all(np.float64(vector_field), meta, np.float64(wm_mask), np.float64(seeds),
                                               args.integration,
                                               args.interpolation, args.prob, args.sw, float(args.var),
                                               int(args.samples),
                                               int(args.mtlength), float(args.wmmin), float(args.exp or 1))

    paths_len = [sum(paths_len[:i + 1]) for i in range(len(paths_len))]

    tracks = PlyElement.describe(np.array(paths, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                        ('seedpoint', 'f4'), ('angle', 'f4')]),
                                 'vertices',
                                 comments=[])
    endindex = PlyElement.describe(np.array(paths_len, dtype=[('endindex', 'i4')]), 'fiber')
    PlyData([tracks, endindex]).write(args.o)
    print("Output file has been written to " + args.o)


if __name__ == "__main__":
    main()
