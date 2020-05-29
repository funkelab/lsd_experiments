import daisy
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np

filename = '/nrs/funke/sheridana/zebrafish_nuclei/130201zf142.zarr'
ds_name = '/volumes/labels/fixed_mask/s0'
interpolated_ds_name = '/volumes/labels/interpolated_mask'

def interpolate(mask, begin, end):

    # begin and end are non-zero sections around a sequence of zero sections
    print(f"Interpolating between {begin} and {end}...")

    # single zero section
    if end - begin == 2:
        mask[begin + 1] = mask[begin]
        return

    # two zero sections
    if end - begin == 3:
        mask[begin + 1] = mask[begin]
        mask[end - 1] = mask[end]
        return

    # more than two sections

    edt_begin = distance_transform_edt(mask[begin]) - \
        distance_transform_edt(1 - mask[begin])
    edt_end = distance_transform_edt(mask[end]) - \
        distance_transform_edt(1 - mask[end])
    for z in range(begin + 1, end):

        alpha = float(end - z)/(end - begin)
        interpolated = alpha*edt_begin + (1.0 - alpha)*edt_end
        mask[z] = interpolated > 0


def interpolate_block(block, ds, ds_out):

    print(f"Reading mask in {block.read_roi}...")
    mask = ds.to_ndarray(block.read_roi, fill_value=0)

    print(f"Finding zero sections...")
    is_nonzero = [np.any(mask[z]) for z in range(mask.shape[0])]

    print(f"Found non-zero sections {is_nonzero}")

    begin = -1
    end = -2
    for z in range(mask.shape[0]):
        if is_nonzero[z]:
            # going from zero to non-zero
            if end == z - 1:
                interpolate(mask, begin, end + 1)
            # the last non-zero seen so far
            begin = z
        else:
            # the last zero seen so far
            end = z

    mask = daisy.Array(mask, roi=block.read_roi, voxel_size=ds.voxel_size)
    ds_out[block.write_roi] = mask[block.write_roi]

if __name__ == "__main__":

    ds = daisy.open_ds(filename, ds_name)
    ds_out = daisy.prepare_ds(
        filename,
        interpolated_ds_name,
        total_roi=ds.roi,
        voxel_size=ds.voxel_size,
        dtype=ds.dtype)

    chunk_size = ds.voxel_size * ds.chunk_shape

    context = ds.voxel_size * (6, 0, 0)
    write_size = daisy.Coordinate((chunk_size[0],) + ds.roi.get_shape()[1:])
    write_roi = daisy.Roi((0, 0, 0), write_size)
    read_roi = write_roi.grow(context, context)

    daisy.run_blockwise(
        ds.roi.grow(context, context),
        read_roi,
        write_roi,
        process_function=lambda b: interpolate_block(b, ds, ds_out),
        fit='shrink',
        num_workers=8)
