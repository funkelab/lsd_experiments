import daisy
import logging
import lsd
import json
import sys
import pymongo
import time

logging.basicConfig(level=logging.INFO)


def extract_fragments_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    db_name = config['db_name']
    db_host = config['db_host']
    queue = config['queue']
    num_voxels_in_block = config['num_voxels_in_block']
    fragments_in_xy=config['fragments_in_xy']
    epsilon_agglomerate=config['epsilon_agglomerate']
    filter_fragments=config['filter_fragments']

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if 'mask_file' in config:

        logging.info("Reading mask from %s", config['mask_file'])
        mask = daisy.open_ds(
            config['mask_file'],
            config['mask_dataset'],
            mode='r')

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
            break

        start = time.time()

        logging.info("block read roi begin: %s", block.read_roi.get_begin())
        logging.info("block read roi shape: %s", block.read_roi.get_shape())
        logging.info("block write roi begin: %s", block.write_roi.get_begin())
        logging.info("block write roi shape: %s", block.write_roi.get_shape())

        lsd.watershed_in_block(
            affs,
            block,
            rag_provider,
            fragments,
            num_voxels_in_block=num_voxels_in_block,
            mask=mask,
            fragments_in_xy=fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            filter_fragments=filter_fragments)

        document = {
            'num_cpus': 5,
            'queue': queue,
            'block_id': block.block_id,
            'read_roi': (
                block.read_roi.get_begin(),
                block.read_roi.get_shape()
            ),
            'write_roi': (
                block.write_roi.get_begin(),
                block.write_roi.get_shape()
            ),
            'start': start,
            'duration': time.time() - start
        }
        blocks_extracted.insert(document)

        client.release_block(block, ret=0)


if __name__ == '__main__':

    extract_fragments_worker(sys.argv[1])
