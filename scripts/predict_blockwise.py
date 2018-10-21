import hashlib
import json
import logging
import numpy as np
import os
import daisy
import sys

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)

def predict_blockwise(
        experiment,
        setup,
        iteration,
        in_file,
        in_dataset,
        out_file,
        out_dataset,
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

        in_file (``string``):
        in_dataset (``string``):

            Path to the dataset to predict in.

        out_file (``string``):
        out_dataset (``string``):

            Path to the output datset.

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    experiment_dir = '../' + experiment
    data_dir = os.path.join(experiment_dir, '01_data')
    train_dir = os.path.join(experiment_dir, '02_train')
    network_dir = os.path.join(setup, str(iteration))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    in_file = os.path.abspath(in_file)
    out_file = os.path.abspath(out_file)

    print('Input file path: ', in_file)
    print('Output file path: ', out_file)
    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    source = daisy.open_ds(in_file, in_dataset)
    print("Source dataset has shape %s, ROI %s, voxel size %s"%(
        source.shape, source.roi, source.voxel_size))

    # load config
    with open(os.path.join(setup, 'config.json')) as f:
        net_config = json.load(f)

    out_dims = net_config['out_dims']
    out_dtype = net_config['out_dtype']
    print('Number of dimensions is %i'%out_dims)

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    chunk_size = net_output_size
    context = (net_input_size - net_output_size)/2

    print("Following sizes in world units:")
    print("net input size  = %s"%(net_input_size,))
    print("net output size = %s"%(net_output_size,))
    print("context         = %s"%(context,))

    # compute sizes of blocks
    block_output_size = chunk_size*tuple(block_size_in_chunks)
    block_input_size = block_output_size + context*2

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), block_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), block_output_size)

    print("Following ROIs in world units:")
    print("Total input ROI  = %s"%input_roi)
    print("Block read  ROI  = %s"%block_read_roi)
    print("Block write ROI  = %s"%block_write_roi)
    print("Total output ROI = %s"%output_roi)

    logging.info('Preparing output dataset')
    print("Preparing output dataset...")

    ds = daisy.prepare_ds(
        out_file,
        out_dataset,
        output_roi,
        source.voxel_size,
        out_dtype,
        write_roi=daisy.Roi((0, 0, 0), chunk_size),
        num_channels=out_dims)

    print("Starting block-wise processing...")

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: predict_in_block(
            experiment,
            setup,
            iteration,
            in_file,
            in_dataset,
            out_file,
            out_dataset,
            b),
        check_function=lambda b: check_block(out_file, out_dataset, b),
        num_workers=num_workers,
        processes=False,
        read_write_conflict=False,
        fit='overhang')

def predict_in_block(
        experiment,
        setup,
        iteration,
        in_file,
        in_dataset,
        out_file,
        out_dataset,
        block):

    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    read_roi = block.read_roi
    write_roi = block.write_roi

    print("Predicting in %s"%write_roi)

    if in_file.endswith('.json'):
        with open(in_file, 'r') as f:
            spec = json.load(f)
            in_file = spec['container']

    config = {
        'experiment': experiment,
        'setup': setup,
        'iteration': iteration,
        'in_file': in_file,
        'in_dataset': in_dataset,
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

    daisy.call([
        'run_lsf',
        '-c', '2',
        '-g', '1',
        '-d', 'funkey/lsd:v0.5',
        'python -u %s %s'%(
            predict_script,
            config_file
        )],
        log_out='%d.out'%config_hash,
        log_err='%d.err'%config_hash)

    print("Finished block with config %s..."%config_file)

    # if things went well, remove temporary files
    os.remove(config_file)
    os.remove('%d.out'%config_hash)
    os.remove('%d.err'%config_hash)

def check_block(out_file, out_dataset, block):

    print("Checking if block %s is complete..."%block.write_roi)

    ds = daisy.open_ds(out_file, out_dataset)

    if ds.roi.intersect(block.write_roi).empty():
        print("Block outside of output ROI")
        return True

    center_values = ds[block.write_roi.get_begin()]
    s = np.sum(center_values)
    print("Sum of center values in %s is %f"%(block.write_roi, s))

    return s != 0

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    predict_blockwise(**config)
