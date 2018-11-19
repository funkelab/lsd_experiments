import daisy
import json
import logging
import lsd
import sys
import time
import glob
import numpy as np
from pymongo import MongoClient, ASCENDING

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def store_fragseg_lut(
        db_host,
        db_name,
        collection_name,
        lut):

    client = MongoClient(db_host)
    database = client[db_name]
    collection = database[collection_name]

    # no need to append, create new one whenever we are here
    collection.drop()

    collection.create_index(
        [
            ('fragment', ASCENDING)
        ],
        name='fragment',
        unique=True)

    collection.create_index(
        [
            ('segment', ASCENDING)
        ],
        name='segment')

    try:
        collection.insert_many(lut)
    except BulkWriteError as e:
        print(e.details)
        raise e

def extract_segmentation(
        db_host,
        db_name,
        edges_collection,
        thresholds,
        roi_offset=None,
        roi_shape=None,
        num_workers=1,
        **kwargs):

    edge_files = glob.glob('calyx_dump/edges_*.npy')
    print("Found %d edge files"%len(edge_files))

    start = time.time()
    edges = np.concatenate([
        np.load(edge_file)
        for edge_file in edge_files
    ])
    print("Read all edges in %.3fs"%(time.time() - start))

    score_files = glob.glob('calyx_dump/scores_*.npy')
    print("Found %d score files"%len(score_files))

    start = time.time()
    scores = np.concatenate([
        np.load(score_file)
        for score_file in score_files
    ])
    print("Read all scores in %.3fs"%(time.time() - start))

    node_files = glob.glob('calyx_dump/nodes_*.csv')
    print("Found %d node files"%len(node_files))

    start = time.time()
    nodes = np.concatenate([
        np.load(node_file)
        for node_file in node_files
    ])
    print("Read all nodes in %.3fs"%(time.time() - start))

    print("Complete RAG contains %d nodes, %d edges"%(len(nodes), len(edges)))

    for threshold in thresholds:

        print("Getting CCs for threshold %.3f..."%threshold)
        start = time.time()
        components = lsd.connected_components(nodes, edges, scores, threshold)
        print("%.3fs"%(time.time() - start))

        print("Creating fragment-segment LUT...")
        start = time.time()
        lut = list([
            {
                'fragment': int(f),
                'segment': int(s)
            }
            for f, s in zip(nodes, components)
        ])
        print("%.3fs"%(time.time() - start))

        print("Storing fragment-segment LUT...")
        start = time.time()
        store_fragseg_lut(
            db_host,
            db_name,
            'seg_%s_%d'%(edges_collection,int(threshold*100)),
            lut)
        print("%.3fs"%(time.time() - start))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
