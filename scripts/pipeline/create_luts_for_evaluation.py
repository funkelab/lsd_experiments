import daisy
import glob
import json
import logging
import numpy as np
import os
import pymongo
import shutil
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_site_segment_lut(
        segments,
        sites,
        roi):

    '''Get the segment IDs of all the sites that are contained in the given
    ROI.'''

    sites = list(sites)

    if len(sites) == 0:
        logger.info("No sites in %s, skipping", roi)
        return None

    logger.info(
        "Getting segment IDs for %d synaptic sites in %s...",
        len(sites),
        roi)

    # for a few sites, direct lookup is faster than memory copies
    if len(sites) >= 15:

        logger.info("Copying segments into memory...")
        start = time.time()

        segments = segments[roi]
        segments.materialize()

        logger.info("%.3fs", time.time() - start)

    logger.info("Getting segment IDs for synaptic sites in %s...", roi)
    start = time.time()

    segment_ids = np.array([
        segments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])

    site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    logger.info(
        "Got segment IDs for %d sites in %.3fs",
        len(segment_ids),
        time.time() - start)

    lut = np.array([site_ids, segment_ids])

    return lut


def store_lut_in_block(
        block,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        segments,
        site_segment_lut_dir):

    logger.info("Finding segment IDs in block %s", block)

    # get all skeleton nodes (which include synaptic sites)
    client = pymongo.MongoClient(annotations_db_host)
    database = client[annotations_db_name]
    skeletons_collection = \
        database[annotations_skeletons_collection_name + '.nodes']

    bz, by, bx = block.read_roi.get_begin()
    ez, ey, ex = block.read_roi.get_end()

    site_nodes = skeletons_collection.find(
        {
            'z': {'$gte': bz, '$lt': ez},
            'y': {'$gte': by, '$lt': ey},
            'x': {'$gte': bx, '$lt': ex}
        })

    # get site -> segment ID
    site_segment_lut = get_site_segment_lut(
        segments,
        site_nodes,
        block.write_roi)

    if site_segment_lut is None:
        return

    # store LUT
    block_lut_path = os.path.join(
        site_segment_lut_dir,
        str(block.block_id) + '.npz')
    np.savez_compressed(
        block_lut_path,
        site_segment_lut=site_segment_lut)


if __name__ == '__main__':

        annotations_db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke"
        annotations_db_name = "zebrafinch_gt_skeletons_new_gt_9_9_20_testing"
        annotations_skeletons_collection_name = "zebrafinch"

        segments = daisy.open_ds(
                sys.argv[1],
                'volumes/debug_seg_relabelled/s0',
                mode='r')

        site_segment_lut_dir = os.path.join(
                sys.argv[1],
                'luts/site_segment/new_gt_9_9_20/debug/')

        site_segment_lut_dir = os.path.join(
                site_segment_lut_dir,
                'debug_roi_not_masked_relabelled')

        roi = daisy.Roi(
                segments.roi.get_begin(),
                segments.roi.get_shape())

        print('Roi: %s' %roi)

        if os.path.exists(site_segment_lut_dir):

            print('Removing %s'%site_segment_lut_dir)

            shutil.rmtree(site_segment_lut_dir)

        os.makedirs(site_segment_lut_dir)

        daisy.run_blockwise(
                roi,
                daisy.Roi((0,)*3, (9000,)*3),
                daisy.Roi((0,)*3, (9000,)*3),
                lambda b: store_lut_in_block(
                    b,
                    annotations_db_host,
                    annotations_db_name,
                    annotations_skeletons_collection_name,
                    segments,
                    site_segment_lut_dir),
                num_workers=32,
                fit='shrink')




