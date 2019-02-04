import daisy
import json
import logging
import lsd
import sys
import time
import glob
import os
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from funlib.segment.graphs.impl import connected_components

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.shared_graph_provider').setLevel(logging.DEBUG)

def read_graph_from_dump(dump_dir):

    edge_files = sorted(glob.glob(dump_dir + '/edges_*.npy'))
    print("Found %d edge files"%len(edge_files))

    start = time.time()
    edges = np.concatenate([
        np.load(edge_file)
        for edge_file in edge_files
    ]).astype(np.uint64)
    print("Read all edges in %.3fs"%(time.time() - start))

    score_files = sorted(glob.glob(dump_dir + '/scores_*.npy'))
    print("Found %d score files"%len(score_files))

    start = time.time()
    scores = np.concatenate([
        np.load(score_file)
        for score_file in score_files
    ]).astype(np.float32)
    print("Read all scores in %.3fs"%(time.time() - start))

    node_files = sorted(glob.glob(dump_dir + '/nodes_*.npy'))
    print("Found %d node files"%len(node_files))

    start = time.time()
    nodes = np.concatenate([
        np.load(node_file)
        for node_file in node_files
    ]).astype(np.uint64)
    print("Read all nodes in %.3fs"%(time.time() - start))

    return (nodes, edges, scores)

def find_segments(
        db_host,
        db_name,
        fragments_file,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        **kwargs):

    if 'dump_dir' in kwargs:

        print("Reading graph from dump in ", kwargs['dump_dir'])
        nodes, edges, scores = read_graph_from_dump(kwargs['dump_dir'])

    else:

        print("Reading graph from DB ", db_name, edges_collection)
        start = time.time()

        graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            db_host,
            edges_collection=edges_collection,
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'])

        roi = daisy.Roi(
            kwargs['roi_offset'],
            kwargs['roi_shape'])

        node_attrs, edge_attrs = graph_provider.read_blockwise(
            roi,
            block_size=daisy.Coordinate((10000, 10000, 10000)),
            num_workers=kwargs['num_workers'])

        print("Read graph in %.3fs"%(time.time() - start))

        if 'id' not in node_attrs:
            print('No nodes found in roi %s' % roi)
            return

        print('id dtype: ', node_attrs['id'].dtype)
        print('edge u  dtype: ', edge_attrs['u'].dtype)
        print('edge v  dtype: ', edge_attrs['v'].dtype)

        nodes = node_attrs['id']
        edges = np.stack([edge_attrs['u'].astype(np.uint64), edge_attrs['v'].astype(np.uint64)], axis=1)
        scores = edge_attrs['merge_score'].astype(np.float32)

        print('Nodes dtype: ', nodes.dtype)
        print('edges dtype: ', edges.dtype)
        print('scores dtype: ', scores.dtype)

    print("Complete RAG contains %d nodes, %d edges"%(len(nodes), len(edges)))

    out_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    out_dir = os.makedirs(out_dir, exist_ok=True)

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    for threshold in thresholds:

        print("Getting CCs for threshold %.3f..."%threshold)
        start = time.time()
        components = connected_components(nodes, edges, scores, threshold)
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

        lut = np.array(lut)

        print("%.3fs"%(time.time() - start))

        print("Storing fragment-segment LUT...")
        start = time.time()

        lookup = 'seg_%s_%d'%(edges_collection, int(threshold*100))

        out_file = os.path.join(out_dir, lookup)

        np.savez_compress(out_file, fragment_segment_lut=lut)

        print("%.3fs"%(time.time() - start))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    find_segments(**config)
    print('Took %.3f seconds to find segments and store LUTs'%(time.time() - start))
