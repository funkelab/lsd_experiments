import z5py
import numpy as np
from scipy import stats
from peach import Roi, Coordinate

if __name__ == "__main__":

    raw_voxel_size_source = Coordinate((8, 8, 8))
    seg_voxel_size_source = Coordinate((8, 8, 8))

    raw_crop_nm = Roi(
        (10240 + 24576, 3328 + 30280, 31744 + 24576),
        (9920, 9920, 9920))
    seg_crop_nm = Roi(
        (3072, 30280, 46080),
        (9920, 9920, 9920))
    print("Shape in nm: %s"%(raw_crop_nm.get_shape(),))

    raw_crop_vx_source = raw_crop_nm//raw_voxel_size_source
    seg_crop_vx_source = seg_crop_nm//seg_voxel_size_source

    with z5py.File('/nrs/turaga/funkej/fib19/fib19.n5', use_zarr_format=False) as f:

        print("Reading raw from %s"%raw_crop_nm)
        print("Reading seg from %s"%seg_crop_nm)
        print("in voxels for raw: %s"%raw_crop_vx_source)
        print("in voxels for seg: %s"%seg_crop_vx_source)

        raw = f['volumes/raw/s0']
        seg = f['volumes/labels/neuron_ids']
        print("Raw dataset, shape=%s, dtype=%s"%(raw.shape, raw.dtype))
        print("Segmentations, shape=%s, dtype=%s"%(seg.shape, seg.dtype))

        raw_crop = raw[raw_crop_vx_source.to_slices()]
        seg_crop = seg[seg_crop_vx_source.to_slices()]

    print("Writing raw...")
    with z5py.File('cube06.n5', use_zarr_format=False) as f:
        raw_ds = f.create_dataset('volumes/raw', data=raw_crop, compression='gzip')
        raw_ds.attrs['resolution'] = (8, 8, 8)
        raw_ds.attrs['offset'] = raw_crop_nm.get_begin()[::-1]
        raw_ds.attrs['origin'] = 'fib19.n5/volumes/raw/s0'
        seg_ds = f.create_dataset('volumes/labels/segmentation', data=seg_crop, compression='gzip')
        seg_ds.attrs['resolution'] = (8, 8, 8)
        seg_ds.attrs['offset'] = raw_crop_nm.get_begin()[::-1]
        seg_ds.attrs['origin'] = 'fib19.n5/volumes/labels/neuron_ids'
