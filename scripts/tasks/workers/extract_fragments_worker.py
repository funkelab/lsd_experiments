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

    mask_file = None
    mask_dataset = None
    fragments_in_xy = False
    epsilon_agglomerate = 0

    for key in config:
        globals()['%s' % key] = config[key]

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
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
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x']
        )
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_extracted = db['blocks_extracted']

    client = daisy.Client()

    while True:

        block = client.acquire_block()

        if block is None:
            return

        start = time.time()

        logging.info("Running fragment extraction for block %s" % block)

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

        client.release_block(block, ret=0)


if __name__ == '__main__':

    run_worker(sys.argv[1])

