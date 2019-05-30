import json
import os
import logging
import numpy as np
import sys
import daisy
import pymongo
import lsd 
import time

logging.basicConfig(level=logging.INFO)

def run_worker(input_config):

    logging.info(sys.argv)
    config_file = input_config
    with open(config_file, 'r') as f:
        config = json.load(f)

    logging.info(config)

    for key in config:
        globals()['%s' % key] = config[key]

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
        position_attribute=['center_z', 'center_y', 'center_x'])
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_agglomerated = ''.join([
        'blocks_agglomerated_',
        merge_function])

    blocks_agglomerated = db[blocks_agglomerated]

    client = daisy.Client()

    while True:

        block = client.acquire_block()

        if block is None:
            return

        start = time.time()

        lsd.agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                block,
                merge_function=waterz_merge_function,
                threshold=1.0)

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

        client.release_block(block, ret=0)

if __name__ == '__main__':

    run_worker(sys.argv[1])

