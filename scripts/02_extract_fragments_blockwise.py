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
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def extract_fragments(
        experiment,
        setup,
        iteration,
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        queue,
        **kwargs):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        affs_file,
        affs_dataset,

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    network_dir = os.path.join(experiment, setup, str(iteration))

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0, 0, 0), block_size),
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'zlib', 'level':5}
        )


    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if 'blocks_extracted' not in db.list_collection_names():
            blocks_extracted = db['blocks_extracted']
            blocks_extracted.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
    else:
        blocks_extracted = db['blocks_extracted']

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(sys.argv[1], network_dir, queue),
        check_function=lambda b: check_block(
            blocks_extracted,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def start_worker(config_file, network_dir, queue):

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.extract_fragments_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    log_out = os.path.join(output_dir, 'extract_fragments_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'extract_fragments_blockwise_%d.err' %worker_id)

    daisy.call([
        'run_lsf',
        '-c', '1',
        '-g', '0',
        '-q', queue,
        '-b',
        '-s', 'funkey/lsd:v0.8',
        'python', './02_extract_fragments_blockwise.py', sys.argv[1],
        '--run-worker'],
        log_out=log_out,
        log_err=log_err)

def check_block(blocks_extracted, block):

    print("Checking if block %s is complete..."%block.write_roi)

    done = blocks_extracted.count({'block_id': block.block_id}) >= 1

    print("Block %s is %s" % (block, "done" if done else "NOT done"))

    return done

def extract_fragments_worker(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        fragments_in_xy,
        queue,
        epsilon_agglomerate=0,
        mask_file=None,
        mask_dataset=None,
        **kwargs):
    '''
    Args:

        affs_file:
        affs_dataset:

            The input affinities dataset.

        fragments_in_xy (``bool``):

            Extract fragments section-wise.

        mask_file:
        mask_dataset (``string``):

            Where to find the affinities and mask (optional).
    '''

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')
    fragments = daisy.open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if mask_file:

        logging.info("Reading mask from %s", mask_file)
        mask = daisy.open_ds(mask_file, mask_dataset, mode='r')

    else:

        mask = None

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r+')
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_extracted = db['blocks_extracted']

    client = daisy.Client()

    while True:

        block = client.acquire_block()

        if not block:
            return

        start = time.time()

        lsd.watershed_in_block(
            affs,
            block,
            rag_provider,
            fragments,
            fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            mask=mask)

        document = {
            'num_cpus': 5,
            'queue': queue,
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': start,
            'duration': time.time() - start
        }
        blocks_extracted.insert(document)

        client.release_block(block, 0)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    if len(sys.argv) == 2:
        # run the master
        extract_fragments(**config)
    else:
        # run a worker
        extract_fragments_worker(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
