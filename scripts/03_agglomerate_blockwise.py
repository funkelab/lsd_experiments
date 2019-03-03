import json
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time
import pymongo
import psutil

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)

def agglomerate(
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
        merge_function,
        **kwargs):

    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

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

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary below.
    '''

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    network_dir = os.path.join(experiment, setup, str(iteration), merge_function)

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    blocks_agglomerated = ''.join([
        'blocks_agglomerated_',
        merge_function])

    if ''.join(['blocks_agglomerated_', merge_function]) not in db.list_collection_names():
        blocks_agglomerated = db[blocks_agglomerated]
        blocks_agglomerated.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
    else:
        blocks_agglomerated = db[blocks_agglomerated]

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
            blocks_agglomerated,
            b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def start_worker(config_file, network_dir, queue):

    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join('.agglomerate_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    log_out = os.path.join(output_dir, 'agglomerate_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'agglomerate_blockwise_%d.err' %worker_id)

    daisy.call([
        'run_lsf',
        '-c', '8',
        '-g', '0',
        '-q', queue,
        '-b',
        '-s', 'funkey/lsd:v0.8',
        'python', './03_agglomerate_blockwise.py', sys.argv[1],
        '--run_worker'],
        log_out=log_out,
        log_err=log_err)

def check_block(blocks_agglomerated, block):

    # print("Checking if block %s is complete..."%block.write_roi)

    done = blocks_agglomerated.count({'block_id': block.block_id}) >= 1

    # print("Block %s is %s" % (block, "done" if done else "NOT done"))

    return done

def agglomerate_worker(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        queue,
        merge_function,
        **kwargs):

    '''
    Args:

    '''

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }[merge_function]

    logging.info("Reading affs from %s"%affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r+')

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r+',
        directed=False,
        edges_collection='edges_' + merge_function,
        position_attribute=['center_z, center_y, center_x'])
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_agglomerated = ''.join([
        'blocks_agglomerated_',
        merge_function])

    blocks_agglomerated = db[blocks_agglomerated]

    client = daisy.Client()

    num_blocks = 0

    process = psutil.Process(os.getpid())

    while True:

        block = client.acquire_block()

        num_blocks += 1

        if not block:
            return

        start = time.time()

        lsd.agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                block,
                merge_function=waterz_merge_function,
                threshold=1.0)

        logging.info("Process %d blocks consume current memory usage %d", num_blocks, process.memory_info().rss)

        document = {
            'num_cpus': 5,
            'queue': queue,
            'block_id': block.block_id,
            'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
            'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
            'start': start,
            'duration': time.time() - start
        }
        blocks_agglomerated.insert(document)

        client.release_block(block, 0)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    if len(sys.argv) == 2:
        #run master
        agglomerate(**config)

    else:
        #run worker
        agglomerate_worker(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to agglomerate: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
