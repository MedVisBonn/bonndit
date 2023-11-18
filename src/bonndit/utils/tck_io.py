import time

import numpy as np

header = [b'mrtrix tracks\n', b'count: 0000000000\n', b'file: . offset\n']

class Tck:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parameters = {}
        self.header = []
        self.length = 0
        self.data = None

    def write(self, parameters):
        self.parameters = parameters
        header.append(b'start: %s\n' % time.time())
        header.append(b'end: %s\n' % time.time())
        for p in self.parameters.keys():
            header.append(b'%s: %s\n' % (p, self.parameters[p]))
        header.append(b'END\n')
        len_header = len(b''.join(header))
        header[3].replace(b'offset', bytes(str(len_header - 6 + len(str(len_header)))))
        with open(self.file_path, 'rb') as f:
            for h in header:
                f.write(h)
        self.header = header
        if self.data:
            for i in range(self.data.shape[0]):
                self.append(self.data[i])



    def append(self, path):
        v = np.linalg.norm(path)
        if np.isnan(v) or np.isinf(v):
            return
        path = np.ascontiguousarray(np.concatenate([path, [np.nan, np.nan, np.nan]]), dtype='<f4')
        with open(self.file_path, 'ab') as f:
            f.write(np.ndarray.tobytes(path))
        self.length += 1

    def close(self):
        with open(self.file_path, "r+b") as f:
            f.seek(21)
            f.write(b'0'*(10-len(str(self.length))) + bytes(str(self.length)))

        with open(self.file_path, 'ab') as f:
            f.write(np.ndarray.tobytes(np.array([np.inf, np.inf, np.inf])))

        print('File with %s streamlines saved in %s' % (self.length, self.file_path))


    def read(self):
        with open(self.file_path, 'rb') as f:
            g = f.readlines()
            g = b''.join(g)
            offset = g.find(b'END') + 4
            d = np.frombuffer(g[offset:], '<f4')
            tracts = [d[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(d))]
            g = [x.shape[0]/3 == int(x.shape[0]/3) for x in tracts]
            self.data = np.array(tracts, dtype=object)[np.array(g)]
            self.data = np.array([x.reshape((x.shape[0]//3, 3)) for x in self.data], dtype=object)
