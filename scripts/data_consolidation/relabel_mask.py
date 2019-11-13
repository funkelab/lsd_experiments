import zarr
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_holes
import sys

def relabel_mask(mask, label_val, area_threshold):

    to_label = mask == label_val

    ccs = label(to_label)

    mask = ccs == np.argmax(np.bincount(ccs.flat))

    remove_small_holes(mask, area_threshold, in_place=True)

    return mask

if __name__ == '__main__':

    mask_file = zarr.open(sys.argv[1], mode='r+')

    mask_ds = 'volumes/pred_labels/s1'

    in_mask = mask_file[mask_ds][:]

    print(in_mask.shape)

    mask = relabel_mask(in_mask, 1, 100000)
    mask = relabel_mask(mask, 0, 100000)

    out_ds = mask_file.create_dataset(
            'volumes/relabelled_mask',
            data=mask,
            dtype=np.uint8,
            compression='gzip')

    for k, v in mask_file[mask_ds].attrs.items():
        out_ds.attrs[k] = v
