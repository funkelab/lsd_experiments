import dask
import daisy
import sys
import json
import logging
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def renumber(in_file,
             in_dataset,
             out_file,
             out_dataset,
             offset,
             size):
    """
    Renumber a segmentation file by the connected components ONLY in the
    specified ROI.

    Args:

        in_file (``string``):

            The input h5-like file containing a segmentation.

        in_dataset (``string``):

            Name of dataset containing segmentation (should be 1 channel
            integer volume).

        out_file (``string``):

            The out h5-like file to which a renumbered segmentation will
            be written.

        out_dataset (``string``):

            Name of output segmentation dataset.

        offset (``tuple`` of ``int``):

            The offset in world units to start renumbering in.

        size (``tuple`` of ``int``):

            The size in world units of the desired ROI.
    """
    
    total_roi = daisy.Roi(offset, size)
    logging.info("Reading dataset in {0}...".format(total_roi))
    vol = daisy.open_ds(in_file, in_dataset)
    vol = vol[total_roi]
    vol.materialize()
    
    logging.info("Relabelling connected components in dataset...")
    components = vol.data
    dtype = components.dtype
    components = label(vol.data, connectivity=1)
    components = remove_small_objects(components, min_size=2, in_place=True)
    vol.data = components.astype(dtype)

    logging.info("Storing file in {0} as dataset {1}...".format(out_file, out_dataset))
    out = daisy.prepare_ds(out_file,
                           out_dataset,
                           total_roi,
                           vol.voxel_size,
                           dtype)
    out[total_roi] = vol

if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    renumber(**config)
