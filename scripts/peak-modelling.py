import argparse
import nrrd
import numpy as np
from bonndit.pmodels.model_avg import model_avg


def main():
    parser = argparse.ArgumentParser(
        description='This script performs a model selection or averaging according to '
                    'Reducing Model Uncertainty in Crossing Fiber Tractography, Gruen et al. (2021)')

    parser.add_argument('-f', '--fodf', required=True,
                        help='4D input file containing fODFs in masked higher-order tensor format (1+#fODF coefficients,x,y,z)')
    parser.add_argument('-i', '--infile', nargs='+', required=True,
                        help='Three infiles containing low rank approx of rank 1 2 3')
    parser.add_argument('-t', '--type', required=True,
                        help='Selection or average')
    parser.add_argument('-o', '--outfile', required=True,
                        help='5D output file with the approximation result (4,r,x,y,z)')
    parser.add_argument('-x', '--x', default=1,
                        help='Parameter for Distribution')
    parser.add_argument('-y', '--y', default=20,
                        help='Parameter for Distribution')

    args = parser.parse_args()

    # Load fODF input file
    fodf, meta = nrrd.read(args.fodf)
    if fodf.shape[0] != 16:
        raise Exception("fodf has to be 4th order tensor.")
    if len(fodf.shape) != 4:
        raise Exception("fodf have to be in 3d space. Hence, fodf has to be 4d.")

    # Read all low rank approximations and fit them to one array
    ranks = nrrd.read(args.infile[0]), nrrd.read(args.infile[1]), nrrd.read(args.infile[2])
    shape_ranks = [ranks[i][0].shape[1] - 1 for i in range(3)]

    if sum(shape_ranks) != 3 or max(shape_ranks) > 2:
        raise Exception("The given low rank approximations have to be of rank 1, 2, 3.")
    rankk = [0,0,0]
    for i, sh in enumerate(shape_ranks):
        rankk[sh] = ranks[i][0]
    rank_1, rank_2, rank_3 = rankk

    low_rank = np.zeros((3, 4, 3) + rank_3.shape[2:])
    low_rank[0] = rank_3
    low_rank[1, :, :2] = rank_2
    low_rank[2, :, :1] = rank_1
    output = np.zeros(rank_3.shape)
    modelling = np.zeros((3,) + rank_3.shape[2:])

    model_avg(output, low_rank, fodf, args.type, modelling, np.float64(args.x), np.float64(args.y))
    output = output.reshape((4, 3) + fodf.shape[1:])
    # update meta file.
    newmeta = {k: meta[k] for k in ['space', 'space origin']}
    newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
    newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], meta['space directions']))
    nrrd.write(args.outfile, np.float32(output), newmeta)


if __name__ == "__main__":
    main()
