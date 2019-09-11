import cloudvolume
import daisy
import numpy as np
import os
import sys
import zarr

padding = [100, 200, 200]

def replace_all(f, d):
    for i,j in d.items():
        f = f.replace(i,j)
    return f

def fetch_raw(raw_data, gt_volume):

    # assumes all gt_volume are in format: `gt_z<ZSTART>-<ZEND>_y<YSTART>-<YEND>_x<XSTART>-<XEND>.h5`

    print('\t Fetching raw for %s' %gt_volume)

    print('\t Padding raw by %s' %padding)

    rep = {
            "gt_z": "",
            "_y": " ",
            "_x": " ",
            "-": " ",
            ".h5": ""
        }

    l = [int(i) for i in replace_all(gt_volume, rep).split()]

    z_start, z_end = l[0] - padding[0], l[1] + padding[0]
    y_start, y_end = l[2] - padding[1], l[3] + padding[1]
    x_start, x_end = l[4] - padding[2], l[5] + padding[2]

    print('\t Raw Z start, end: %s %s'%(z_start, z_end))
    print('\t Raw Y start, end: %s %s'%(y_start, y_end))
    print('\t Raw X start, end: %s %s'%(x_start, x_end))

    raw = raw_data[x_start:x_end, y_start:y_end, z_start:z_end]
    raw = np.array(np.transpose(raw[...,0], [2,1,0]))

    return raw

if __name__ == '__main__': 

    vol = cloudvolume.CloudVolume(
        'https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata',
        mip=0,bounded=True, progress=False)

    for f in os.listdir(sys.argv[1]):

        if f.endswith('.h5'):

            print('\nConsolidating %s...'%f)

            raw = fetch_raw(vol, f)
            gt = daisy.open_ds(f, 'main').to_ndarray()

            mask = (gt < np.uint64(-10)).astype(np.uint8)

            f_out = zarr.open(f.replace('.h5', '.zarr'), mode='w')

            resolution = [20, 9, 9]

            for ds_name, data in [
                    ('volumes/raw', raw),
                    ('volumes/labels/neuron_ids', gt),
                    ('volumes/labels/mask', mask)]:

                print("\t \t Writing %s..."%ds_name)

                ds_out = f_out.create_dataset(
                        ds_name,
                        data=data,
                        compressor=zarr.get_codec({'id': 'gzip', 'level': 5}))

                ds_out.attrs['resolution'] = resolution

                if ds_name == 'volumes/raw':
                    ds_out.attrs['offset'] = [0, 0, 0]

                else:
                    ds_out.attrs['offset'] = [i*j for i,j in zip(resolution,padding)]

