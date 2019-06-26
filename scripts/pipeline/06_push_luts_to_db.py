from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
import json
import logging
import numpy as np
import multiprocessing as mp
import sys
import time
import os
import glob

def push_to_mongo(
    db_host,
    db_name,
    collection_name,
    lut):

    client = MongoClient(db_host)
    database = client[db_name]
    collection = database[collection_name]

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

def get_lookup_name(edges_collection, threshold):

    lookup = 'seg_%s_%d'%(edges_collection, int(threshold*100))

    return lookup

def load_luts(
    db_host,
    db_name,
    fragments_file,
    edges_collection,
    thresholds_minmax,
    thresholds_step):

    for threshold in list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step)):

        lookup = get_lookup_name(edges_collection, threshold)

        print("Reading fragment-segment LUT for threshold %d..."%threshold)
        start = time.time()
        fragment_segment_lut_file = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment',
            lookup + '.npz')
        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']
        print("%.3fs"%(time.time() - start))

        print("Unpacking lut for threshold %d..."%threshold)
        fragment_segment_lut = list([
            {
                'fragment': int(f),
                'segment': int(s)
            }
            for f, s in zip(
                fragment_segment_lut[0],
                fragment_segment_lut[1]
                )
            ])

        print("Pushing lut for threshold %d to database..."%threshold)
        push_to_mongo(
                db_host,
                db_name,
                lookup,
                fragment_segment_lut)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    load_luts(**config)


