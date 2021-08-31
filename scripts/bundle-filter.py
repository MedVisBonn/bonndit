import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from bonndit.tracking.ItoW import Trafo
from bonndit.filter.filter import plysplitter, streamline_filter, intersection_dict, filter_mask, streamline_itow
import regex as re
import nrrd
from plyfile import PlyElement, PlyData



def main():
    parser = argparse.ArgumentParser(
        description='Given a bundle various checkings on each fiber can be done. All filters are only applied if they '
                    'are set. Filteroptions are exclusion slices, exclusion regions, min length, min density.')

    parser.add_argument('-i', help='Infile should contain a bundle in ply format where the fourth column contains '
                                   'information about the seedpoint.')
    parser.add_argument('-m', help='Metadata from fodf file.')
    parser.add_argument('-o', help='Outfile')
    parser.add_argument('-f', help='Outfile mask')
    parser.add_argument('--mask', help='Min Streamline density. Starting from the seed the streamline is terminated '
                                       'at the first intersection with a low density region')
    parser.add_argument('--exclusion', help='Exclusion slice. Can be set via x<10. Multiple regions can be seperated '
                                            'by whitespace.')
    parser.add_argument('--exclusionq',
                        help='Exclusion region. Can be given via 5<x<10,4<y<20,3<z<30. Multiple regions can be seperated by whitespace.')
    parser.add_argument('--min_len', help='Min length of a streamline in terms of steps.')
    args = parser.parse_args()

    exclusion = []
    if args.exclusion:
        exclusion = re.split(' ', args.exclusion)
        exclusion = [re.split('(<|>)', x) for x in exclusion]
        for ex in exclusion:
            if not ex[0] in ['x', 'y', 'z']:
                raise Exception('Only x,y,z is allowed')
            if not ex[1] in ['<', '>']:
                raise Exception('Only >,< is allowed')
            if not is_integer(ex[2]):
                raise Exception('Only integers are allowed')
    exclusionq = []

    if args.exclusionq:
        exclusionq = re.split(' ', args.exclusionq)
        exclusionq = [re.split(',', x) for x in exclusionq]
        exclusionq = [[re.split('(<|>)', x) for x in exclusionq[index]] for index in range(len(exclusionq))]

    with open(args.i, 'rb') as f:
        try:
            plydata = PlyData.read(f)
        except:
            raise Exception("Plyfile seems to be corrupted.")
        num_verts = plydata['vertices'].count
        num_fiber = plydata['fiber'].count
        vertices = np.zeros(shape=[num_verts, 4], dtype=np.float64)
        endindex = np.zeros(shape=[num_fiber], dtype=np.float64)
        vertices[:, 0] = plydata['vertices'].data['x']
        vertices[:, 1] = plydata['vertices'].data['y']
        vertices[:, 2] = plydata['vertices'].data['z']
        try:
            vertices[:, 3] = plydata['vertices'].data['seedpoint']
        except:
            raise Exception("seedpoint is not in data")
        endindex[:] = plydata['fiber'].data['endindex']

    _, meta = nrrd.read(args.m)
    trafo = Trafo(np.float64(meta['space directions'][2:]), np.float64(meta['space origin']))


    # split streamlines according endindex
    streamlines_test = plysplitter(vertices, endindex)
    # Read meta data and convert the streamlines into subsets in index space
    delete = []
    delete = streamline_filter(streamlines_test, exclusionq, exclusion, trafo)

    streamlines_test = np.array(streamlines_test, dtype=object)[delete]
    endindexes = [streamlines_test[i].shape[0] for i in range(len(streamlines_test))]
    if args.min_len:
        streamlines_test = streamlines_test[np.array(endindexes) > int(args.min_len)]

    # First mask
    mask = np.zeros(meta['sizes'][2:])
    mask2 = np.zeros(meta['sizes'][2:])

    mask = np.array(intersection_dict(mask, streamlines_test))

    # creat ratio between refernce patient and this one

    mask2[mask < float(args.mask)] = 0
    mask2[mask >= float(args.mask)] = 1
    meta['space directions'] = meta['space directions'][2:]
    meta['kinds'] = meta['kinds'][2:]
    if args.f:
        nrrd.write(args.f, mask2, meta)
    mask[mask < float(args.mask)] = 0
    mask = gaussian_filter(mask, sigma=1)
    streamlines_test = filter_mask(streamlines_test, mask)
    endindezes = [True if streamlines_test[i].shape[0] > 1 else False for i in range(len(streamlines_test))]
    streamlines_test = np.array(streamlines_test, dtype=object)[endindezes]

    streamlines_test = streamline_itow(streamlines_test, trafo)

    # filter out short streamlines_test:
    endindexes = [streamlines_test[i].shape[0] for i in range(len(streamlines_test))]

    streamlines_test = np.array(streamlines_test, dtype=object)[np.array(endindexes) > 50]
    streamlines_test = np.concatenate(streamlines_test)
    endindexes = [x for x in endindexes if x > 50]
    endindexes = [sum(endindexes[:i + 1]) for i in range(len(endindexes))]

    # Save the file
    tracks = PlyElement.describe(
        np.array([tuple(x) for x in streamlines_test], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('start', 'f4')]),
        'vertices',
        comments=[])
    endindex = PlyElement.describe(np.array(list(endindexes), dtype=[('endindex', 'i4')]), 'fiber')
    PlyData([tracks, endindex]).write(args.o)


if __name__=="__main__":
    main()


