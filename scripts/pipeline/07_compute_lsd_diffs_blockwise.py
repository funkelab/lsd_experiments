import json
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time
import pymongo

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)

def compute_lsd_diffs(
        experiment,
        setup,
        iteration,
        lsds_file,
        lsds_dataset,
        fragments_file,
        fragments_dataset,
        lsd_diffs_file,
        lsd_diffs_dataset,
        seg_dataset,
        block_size,
        db_host,
        edges_db_name,
        edges_collection,
        num_workers,
        queue,
        **kwargs):


    logging.info("Opening lsds from %s, %s", lsds_file, lsds_dataset)
    lsds = daisy.open_ds(lsds_file, lsds_dataset, mode='r')

    # center = lsds.roi.get_center()

    # roi = daisy.Roi(center, daisy.Coordinate(block_size))

    roi = lsds.roi

    network_dir = os.path.join(experiment, setup, str(iteration))

    setup_dir = os.path.join('../../', experiment, '02_train', setup)
    compute_lsd_diffs_script = os.path.abspath(os.path.join(
        setup_dir,
        'compute_lsd_diff.py'))

    # prepare lsd_diffs dataset
    lsd_diffs = daisy.prepare_ds(
        lsd_diffs_file,
        lsd_diffs_dataset,
        total_roi=roi,
        voxel_size=lsds.voxel_size,
        dtype=np.uint8,
        write_roi=daisy.Roi((0,)*3,block_size))

    seg = daisy.prepare_ds(
        lsd_diffs_file,
        seg_dataset,
        total_roi=roi,
        voxel_size=lsds.voxel_size,
        dtype=np.uint64,
        write_roi=daisy.Roi((0,)*3,block_size))

    read_roi = daisy.Roi((0,)*roi.dims(), block_size)
    write_roi = daisy.Roi((0,)*roi.dims(), block_size)

    daisy.run_blockwise(
        roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(
            compute_lsd_diffs_script,
            sys.argv[1],
            network_dir,
            queue),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def start_worker(compute_lsd_diffs_script, config_file, network_dir, queue):

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.compute_lsd_diffs_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    log_out = os.path.join(output_dir, 'compute_lsd_diffs_blockwise_%d.out' % worker_id)
    log_err = os.path.join(output_dir, 'compute_lsd_diffs_blockwise_%d.err' % worker_id)

    daisy.call([
        'run_lsf',
        '-c', '1',
        '-g', '0',
        '-q', queue,
        # '-s', 'funkey/lsd:v0.8',
        'python', compute_lsd_diffs_script, config_file],
        log_out=log_out,
        log_err=log_err)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    if len(sys.argv) == 2:
        # run the master
        compute_lsd_diffs(**config)
    else:
        # run a worker
        compute_lsd_diffs_worker(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
