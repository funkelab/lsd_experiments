import hashlib
import json
import logging
import numpy as np
import os
import daisy
import sys
import time
import datetime
import pymongo

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)

def predict_blockwise(
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset,
        num_workers,
        db_host,
        db_name,
        collection='blocks_predicted',
        lsds_file=None,
        lsds_dataset=None):

    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        lsds_file (``string``):
        lsds_dataset (``string``):

            Paths to the input datasets. lsds can be None if not needed.

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

    raw_file = os.path.abspath(raw_file)
    out_file = os.path.abspath(os.path.join(out_file, setup, str(iteration), 'calyx.zarr'))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    print('Input file path: ', raw_file)
    print('Output file path: ', out_file)
    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        in_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, in_dataset)
    print("Source dataset has shape %s, ROI %s, voxel size %s"%(
        source.shape, source.roi, source.voxel_size))

    # load config
    with open(os.path.join(setup, 'config.json')) as f:
        print("Reading setup config from %s"%os.path.join(setup, 'config.json'))
        net_config = json.load(f)

    out_dims = net_config['out_dims']
    out_dtype = net_config['out_dtype']
    print('Number of dimensions is %i'%out_dims)

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    context = (net_input_size - net_output_size)/2

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    print("Following sizes in world units:")
    print("net input size  = %s"%(net_input_size,))
    print("net output size = %s"%(net_output_size,))
    print("context         = %s"%(context,))


    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    print("Following ROIs in world units:")
    print("Block read  ROI  = %s"%block_read_roi)
    print("Block write ROI  = %s"%block_write_roi)
    print("Total input  ROI  = %s"%input_roi)
    print("Total output ROI  = %s"%output_roi)

    logging.info('Preparing output dataset')
    print("Preparing output dataset...")

    ds = daisy.prepare_ds(
        out_file,
        out_dataset,
        output_roi,
        source.voxel_size,
        out_dtype,
        write_roi=block_write_roi,
        num_channels=out_dims,
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'gzip', 'level':5}
        )

    print("Starting block-wise processing...")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    if collection not in db.list_collection_names():
        blocks_predicted = db[collection]
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')
    else:
        blocks_predicted = db[collection]

    # process block-wise
    succeeded = daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            experiment,
            setup,
            iteration,
            raw_file,
            raw_dataset,
            lsds_file,
            lsds_dataset,
            out_file,
            out_dataset,
            db_host,
            db_name,
            collection),
        check_function=lambda b: check_block(
            blocks_predicted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang')

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        lsds_file,
        lsds_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        collection):

    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict_newdaisy.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']

    worker_config = {
        'queue': 'gpu_tesla',
        'num_cpus': 5,
        'num_cache_workers': 10,
        'singularity': None # TODO: use 'funkey/lsd:v0.7'
    }

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'lsds_file': lsds_file,
        'lsds_dataset': lsds_dataset,
        'out_file': out_file,
        'out_dataset': out_dataset,
        'db_host': db_host,
        'db_name': db_name,
        'collection': collection,
        'worker_config': worker_config
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    try:
        os.makedirs('.predict_configs')
    except:
        pass
    config_file = os.path.join('.predict_configs', '%d.config'%config_hash)
    log_out = os.path.join('.predict_configs', '%d.out'%config_hash)
    log_err = os.path.join('.predict_configs', '%d.err'%config_hash)
    with open(config_file, 'w') as f:
        json.dump(config, f)

    print("Running block with config %s..."%config_file)

    command = [
        'run_lsf',
        '-c', str(worker_config['num_cpus']),
        '-g', '1',
        '-q', worker_config['queue']
    ]

    if worker_config['singularity']:
        command += ['-s', worker_config['singularity']]

    command += [
        'python -u %s %s'%(
            predict_script,
            config_file
        )]

    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')


    # # if things went well, remove temporary files
    # os.remove(config_file)
    # os.remove(log_out)
    # os.remove(log_err)

def check_block(blocks_predicted, block):

    print("Checking if block %s is complete..."%block.write_roi)

    done = blocks_predicted.count({'block_id': block.block_id}) >= 1

    print("Block %s is %s" % (block, "done" if done else "NOT done"))

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    
    predict_blockwise(**config)

    end = time.time()
    
    seconds = end - start
    print('Total time to predict: %f seconds' % seconds)

