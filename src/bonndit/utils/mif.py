import numpy as np


def _to_csv2D(affine, tag):
    ret = ''
    for i in affine:
        ret += tag + ','.join(i) + '\n'
    return ret

def _to_csv(affine, tag):
    ret = ''
    ret += tag + ','.join(affine)
    return ret

def save(filename, data, options={}):
        ''' Save image to MRtix .mif file. '''
        if data is None:
            raise RuntimeError('Image data not set.')
        if not filename.endswith('.mif'):
            raise IOError('only .mif file type supported for writing')
        # write image header

        with open(filename, 'w', encoding='latin-1') as f:
            f.write('mrtrix image\n')
            f.write('dim: ' + _to_csv(data.shape) + '\n')
            f.write('vox: ' + '1,1,1,1' + '\n')

            f.write('layout: ' + options['layout'] if 'layout' in options else 'layout: ' + '-1,-2,+3,0' + '\n')
            f.write('datatype: ' + 'Float32LE' + '\n')
            f.write(_to_csv2D(options['affine'], 'transform: '))
            f.flush()
            offset = f.tell() + 13
            offset += int(np.floor(np.log10(offset))) + 1
            f.write('file: . {:d}\n'.format(offset))
            f.write('END\n')
            f.flush()
        # write image data
        with open(filename, 'ab') as f:
            data.ravel(order='K').tofile(f)

