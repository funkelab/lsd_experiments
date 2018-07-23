from subprocess import check_call, CalledProcessError
import h5py
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

def call(command, log_out, log_err):

    try:
        with open(log_out, 'w') as stdout:
            with open(log_err, 'w') as stderr:
                check_call(
                    ' '.join(command) + ' | sed \'s/\r//g\'',
                    shell=True,
                    stdout=stdout,
                    stderr=stderr)
    except CalledProcessError as exc:
        raise Exception("Calling %s failed with recode %s, stderr in %s"%(
            ' '.join(command),
            exc.returncode,
            stderr.name))

def predict_blockwise(
        setup,
        iteration,
        in_file,
        in_dataset,
        block_input_size,
        block_output_size,
        chunk_size,
        out_file,
        out_dataset,
        out_dims,
        num_workers):

    # get absolute paths
    setup = os.path.abspath(setup)
    in_file = os.path.abspath(in_file)
    out_file = os.path.abspath(out_file)

    # get ROI of source
    source = get_dataset(in_file, in_dataset)
    voxel_size = peach.Coordinate(source.attrs['resolution'])
    source_roi_nm = peach.Roi(
        peach.Coordinate(source.attrs['offset'][::-1]),
        voxel_size*source.shape[-3:])
    source_roi = source_roi_nm/voxel_size

    # get context and total input and output ROI
    block_input_size = peach.Coordinate(block_input_size)
    block_output_size = peach.Coordinate(block_output_size)
    context = (block_input_size - block_output_size)/2
    input_roi = source_roi.grow(context, context)
    output_roi = source_roi

    # create read and write ROI
    block_read_roi = peach.Roi((0, 0, 0), block_input_size)
    block_write_roi = peach.Roi((0, 0, 0), block_output_size)
    block_read_roi -= context

    print("Following ROIs in voxels:")
    print("Input ROI       = %s"%input_roi)
    print("Block read  ROI = %s"%block_read_roi)
    print("Block write ROI = %s"%block_write_roi)
    print("Output ROI      = %s"%output_roi)

    print("Preparing output datasets...")

    out = z5py.File(out_file, use_zarr_format=False)
    if not out_dataset in out:
        lsds = out.create_dataset(
            out_dataset,
            shape=(out_dims,) + output_roi.get_shape(),
            chunks=(out_dims,) + tuple(chunk_size),
            dtype='float32',
            compression='gzip')
        lsds.attrs['resolution'] = voxel_size
        lsds.attrs['offset'] = source_roi_nm.get_begin()[::-1]

    print("Starting block-wise processing...")

    # process block-wise
    peach.run_with_dask(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda r, w: predict_in_block(
            setup,
            iteration,
            in_file,
            out_file,
            r,
            w),
        check_function=lambda w: check_block(out_file, out_dataset, w),
        num_workers=num_workers,
        processes=False,
        read_write_conflict=False)

def get_dataset(filename, dataset):

    if filename.endswith('h5') or filename.endswith('hdf'):
        return h5py.File(filename)[dataset]
    elif filename.endswith('n5'):
        return z5py.File(filename)[dataset]
    else:
        raise RuntimeError("Unknown file format for %s"%filename)

def check_block(out_file, out_dataset, write_roi):

    print("Checking if block %s is complete..."%write_roi)

    ds = get_dataset(out_file, out_dataset)

    voxel_size = peach.Coordinate(ds.attrs['resolution'][::-1])
    offset = peach.Coordinate(ds.attrs['offset'][::-1])
    write_roi -= offset/voxel_size

    s = np.sum(ds[(slice(None),) + write_roi.to_slices()])

    print("Sum of entries in %s is %f"%(write_roi, s))

    return s != 0

def predict_in_block(setup, iteration, in_file, out_file, read_roi, write_roi):

    print("Predicting in %s"%write_roi)

    config = {
        'iteration': iteration,
        'in_file': in_file,
        'read_begin': read_roi.get_begin(),
        'read_shape': read_roi.get_shape(),
        'out_file': out_file,
        'write_begin': write_roi.get_begin(),
        'write_shape': write_roi.get_shape()
    }

    config_str = str(
        setup +
        str(iteration) +
        in_file +
        out_file +
        str(write_roi))
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    print("Hash for block at %s is %d"%(write_roi, config_hash))

    config_file = '%d.config'%config_hash
    with open(config_file, 'w') as f:
        json.dump(config, f)

    try:

        print("Running block with config %s..."%config_file)

        call([
            'run_lsf',
            '-c', '2',
            '-g', '1',
            '-d', 'funkey/lsd:v0.1',
            'python -u %s %s'%(
                os.path.join('../02_train', setup, 'predict.py'),
                config_file
            )],
            log_out='%d.out'%config_hash,
            log_err='%d.err'%config_hash)

        print("Finished block with config %s..."%config_file)

    finally:

        os.remove(config_file)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    predict_blockwise(**config)
