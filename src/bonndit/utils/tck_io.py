import os
from datetime import datetime
import numpy as np

header = [b'mrtrix tracks\n', b'count: 0000000000\n', b'datatype: Float32LE\n', b'file: . offset\n']


class Tck:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parameters = {}
        self.header = []
        self.length = 0
        self.data = None
        self.feature = {}
        self.force = False

    def add_feature_path(self, feature, path):
        if feature not in self.feature.keys():
            self.feature[feature] = dict(path=path, data=None)
        else:
            print("Feature already existing!")

    def write(self, parameters=None):
        if parameters is None:
            parameters = {}
        if not self.force and os.path.isfile(self.file_path):
            raise Exception('%s exists and shouldnt be overwritten' % self.file_path)
        self.parameters = parameters
        header.append(b'start: %s\n' % bytes(str(datetime.now()), encoding='utf-8'))
        header.append(b'end: %s\n' % bytes(str(datetime.now()), encoding='utf-8'))
        for p in self.parameters.keys():
            header.append(b'%s: %s\n' % (bytes(p, encoding="utf-8"), bytes(self.parameters[p], encoding="utf-8")))
        header.append(b'END\n')
        len_header = len(b''.join(header))
        header[3] = header[3].replace(b'offset', bytes(str(len_header - 6 + len(str(len_header))), encoding='utf-8'))
        with open(self.file_path, 'wb') as f:
            for h in header:
                f.write(h)

        for feat in self.feature.keys():
            with open(self.feature[feat]['path'], 'wb') as f:
                for h in header:
                    f.write(h)
        self.header = header
        if self.data is not None:
            for i in range(self.data.shape[0]):
                self.append(self.data[i], {feat: self.feature[feat]['data'][i] for feat in self.feature.keys()})

    def append(self, path, feature=None):
        v = np.linalg.norm(path)
        if np.isnan(v) or np.isinf(v):
            return
        path = np.ascontiguousarray(np.concatenate([path, [[np.nan, np.nan, np.nan]]]), dtype='<f4')
        # write x,y,z
        with open(self.file_path, 'ab') as f:
            f.write(np.ndarray.tobytes(path))
        # write features in separate files
        if feature:
            for feat in feature.keys():
                path = np.ascontiguousarray(np.concatenate([feature[feat], [np.nan]]), dtype='<f4')
                with open(self.feature[feat]['path'], 'ab') as f:
                    f.write(np.ndarray.tobytes(path))
        self.length += 1

    def close(self):
        with open(self.file_path, "r+b") as f:
            f.seek(21)
            f.write(b'0' * (10 - len(str(self.length))) + bytes(str(self.length), encoding='utf-8'))
        if self.feature:
            for feat in self.feature.keys():
                with open(self.feature[feat]['path'], 'r+b') as f:
                    f.seek(21)
                    f.write(b'0' * (10 - len(str(self.length))) + bytes(str(self.length), encoding='utf-8'))

        with open(self.file_path, 'ab') as f:
            f.write(np.ndarray.tobytes(np.array([np.inf, np.inf, np.inf], dtype='<f4')))
            print('File with %s streamlines saved in %s' % (self.length, self.file_path))
        if self.feature:
            for feat in self.feature.keys():
                with open(self.feature[feat]['path'], 'ab') as f:
                    f.write(np.ndarray.tobytes(np.array([np.inf], dtype='<f4')))
                print('File with feature %s saved in %s' % (feat, self.feature[feat]['path']))

    def read(self):
        with open(self.file_path, 'rb') as f:
            g = f.readlines()
            g = b''.join(g)
            offset = g.find(b'END') + 4
            d = np.frombuffer(g[offset:], '<f4')
            tracts = [d[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(d))]
            keep = [x.shape[0] / 3 == int(x.shape[0] / 3) for x in tracts]
            self.data = np.array(tracts, dtype=object)[np.array(keep)]
            self.data = np.array([x.reshape((x.shape[0] // 3, 3)) for x in self.data], dtype=object)

        if self.feature:
            for feat in self.feature:
                with open(self.feature[feat], 'rb') as f:
                    g = f.readlines()
                    g = b''.join(g)
                    offset = g.find(b'END') + 4
                    d = np.frombuffer(g[offset:], '<f4')
                    tracts = [d[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(d))]
                    self.feature[feat]['data'] = np.array(tracts, dtype=object)[np.array(keep)]

        print("File %s read. Contains %s streamlines and %s features" % (self.file_path, self.data.shape[0], len(self.feature.keys())))
