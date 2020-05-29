import numpy as np
import os
import sys
import zarr

if __name__ == '__main__':

    lut_f = sys.argv[1]

    lut = np.load(lut_f)['fragment_segment_lut'][::-1]

    print(lut, lut.shape)

    # paintera_lut = np.vstack((lut[0], lut[1])).T

    # print(paintera_lut, paintera_lut.shape)

    f_out = zarr.open(sys.argv[2], mode='r+')

    ds_out = f_out.create_dataset(
            'volumes/labels/neuron_ids/fragment-segment-assignment',
            data=lut,
            compressor=zarr.get_codec({'id': 'gzip', 'level': -1}))

