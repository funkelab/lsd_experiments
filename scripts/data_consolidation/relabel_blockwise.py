import daisy
import sys
import json
import numpy as np
import logging
from funlib.segment.arrays import relabel_connected_components

logging.basicConfig(level=logging.INFO)
logging.getLogger('funlib.segment.arrays.relabel_connected_components').setLevel(logging.DEBUG)

def relabel_cc(
        in_file,
        out_file,
        in_ds,
        out_ds,
        num_workers):

    logging.info('Loading gt data...')

    gt = daisy.open_ds(in_file, in_ds)

    write_roi = daisy.Roi((0, 0, 0), (3600, 3600, 3600))

    gt_ds = daisy.prepare_ds(
            out_file,
            out_ds,
            gt.roi,
            gt.voxel_size,
            gt.data.dtype,
            write_roi=write_roi)
    gt_ds.check_write_chunk_align = True
    relabel_connected_components(gt, gt_ds, write_roi.get_shape(), num_workers)
    out_data = gt_ds.to_ndarray(gt.roi)

if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    relabel_cc(**config)
