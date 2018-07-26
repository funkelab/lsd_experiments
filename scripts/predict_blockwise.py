from processes import call
from datasets import get_dataset
import hashlib
import json
import logging
import numpy as np
import os
import peach
import sys
import z5py

logging.basicConfig(level=logging.INFO)
# logging.getLogger('peach.blocks').setLevel(logging.DEBUG)

def predict_blockwise(
        experiment,
        setup,
        iteration,
        sample,
        out_dataset,
        out_dims,
        block_size_in_chunks,
        num_workers):
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        sample (``string``):

            Name of the sample to predict in, relative to the experiment's data
            dir. Should be an HDF5 or N5 container with 'volumes/raw'.

        out_dataset (``string``):

            The name of the output dataset (e.g., 'volumes/affs').

        out_dims (``int``):

            The number of components per voxel to predict (e.g., 3 for
            direct neighbor affinities).

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    experiment_dir = '../' + experiment
    data_dir = os.path.join(experiment_dir, '01_data')
    train_dir = os.path.join(experiment_dir, '02_train')
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    # get absolute paths
    setup = os.path.abspath(os.path.join(train_dir, setup))
    in_file = os.path.abspath(os.path.join(data_dir, sample))
    out_file = os.path.abspath(os.path.join(predict_dir, sample))

    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    source, source_offset, voxel_size = get_dataset(in_file, 'volumes/raw')
    source_roi = peach.Roi(
        source_offset,
        voxel_size*source.shape[-3:])

    # get chunk size and context
    with open(os.path.join(setup, 'test_net_config.json')) as f:
        net_config = json.load(f)
    net_input_size = peach.Coordinate(net_config['input_shape'])*voxel_size
    net_output_size = peach.Coordinate(net_config['output_shape'])*voxel_size
    chunk_size = net_output_size
    context = (net_input_size - net_output_size)/2

    print("Following sizes in world untis:")
    print("net input size  = %s"%(net_input_size,))
    print("net output size = %s"%(net_output_size,))
    print("context         = %s"%(net_output_size,))

    # compute sizes of blocks
    block_output_size = chunk_size*tuple(block_size_in_chunks)
    block_input_size = block_output_size + context*2

    # get total input and output ROIs
    input_roi = source_roi.grow(context, context)
    output_roi = source_roi

    # create read and write ROI
    block_read_roi = peach.Roi((0, 0, 0), block_input_size) - context
    block_write_roi = peach.Roi((0, 0, 0), block_output_size)

    print("Following ROIs in world untis:")
    print("Total input ROI  = %s"%input_roi)
    print("Block read  ROI  = %s"%block_read_roi)
    print("Block write ROI  = %s"%block_write_roi)
    print("Total output ROI = %s"%output_roi)

    print("Preparing output datasets...")

    if not os.path.isdir(out_file):
        os.makedirs(out_file)

    out = z5py.File(out_file, use_zarr_format=False)
    if not out_dataset in out:
        ds = out.create_dataset(
            out_dataset,
            shape=(out_dims,) + output_roi.get_shape()/voxel_size,
            chunks=(out_dims,) + tuple(chunk_size/voxel_size),
            dtype='float32',
            compression='gzip')
        ds.attrs['resolution'] = voxel_size
        ds.attrs['offset'] = source_roi.get_begin()[::-1]

    print("Starting block-wise processing...")

    # process block-wise
    peach.run_with_dask(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda r, w: predict_in_block(
            experiment,
            setup,
            iteration,
            in_file,
            out_file,
            out_dataset,
            r,
            w),
        check_function=lambda w: check_block(out_file, out_dataset, w),
        num_workers=num_workers,
        processes=False,
        read_write_conflict=False)

def predict_in_block(
        experiment,
        setup,
        iteration,
        in_file,
        out_file,
        out_dataset,
        read_roi,
        write_roi):

    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    print("Predicting in %s"%write_roi)

    config = {
        'experiment': experiment,
        'setup': setup,
        'iteration': iteration,
        'in_file': in_file,
        'read_begin': read_roi.get_begin(),
        'read_size': read_roi.get_shape(),
        'out_file': out_file,
        'out_dataset': out_dataset,
        'write_begin': write_roi.get_begin(),
        'write_size': write_roi.get_shape()
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    print("Hash for block at %s is %d"%(write_roi, config_hash))

    config_file = '%d.config'%config_hash
    with open(config_file, 'w') as f:
        json.dump(config, f)

    print("Running block with config %s..."%config_file)

    call([
        'run_lsf',
        '-c', '2',
        '-g', '1',
        '-d', 'funkey/lsd:v0.1',
        'python -u %s %s'%(
            predict_script,
            config_file
        )],
        log_out='%d.out'%config_hash,
        log_err='%d.err'%config_hash)

    print("Finished block with config %s..."%config_file)

    # if things went well, remove temporary files
    # os.remove(config_file)
    # os.remove('%d.out'%config_hash)
    # os.remove('%d.err'%config_hash)

def check_block(out_file, out_dataset, write_roi):

    print("Checking if block %s is complete..."%write_roi)

    # TODO: check for empty chunks instead of voxel values

    ds, offset, voxel_size = get_dataset(out_file, out_dataset)

    # convert write_roi to voxels within ds
    write_roi -= offset
    write_roi /= voxel_size

    # get center voxel
    center = write_roi.get_center()
    center_values = ds[:, center[0], center[1], center[2]]

    s = np.sum(center_values)
    print("Sum of center values in %s is %f"%(write_roi, s))

    return s != 0

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    predict_blockwise(**config)
