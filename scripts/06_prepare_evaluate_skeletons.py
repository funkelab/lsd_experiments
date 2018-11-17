from pymongo import MongoClient, ASCENDING
import daisy
import json
import logging
import sys
import time
import string
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_synaptic_sites(
        db_host,
        skeleton_db_name,
        synapse_nodes_collection,
        roi):
    '''Get a dict from synaptic site ID to a dict with 'location' and
    'neuron_id' for each synapse annotation in the given ROI.'''

    print("Reading synaptic sites...")
    start = time.time()

    client = MongoClient(db_host)
    database = client[skeleton_db_name]
    collection = database[synapse_nodes_collection]

    bz, by, bx = roi.get_begin()
    ez, ey, ex = roi.get_end()

    synaptic_sites = collection.find(
        {
            'z': { '$gte': bz, '$lt': ez },
            'y': { '$gte': by, '$lt': ey },
            'x': { '$gte': bx, '$lt': ex },
            'type': { '$in': [ 'syncenter', 'post_neuron' ] }
        })

    # synaptic site IDs are not unique, renumber them here
    synaptic_sites = [
        {
            'id': i + 1,
            'z': s['z'],
            'y': s['y'],
            'x': s['x'],
            'neuron_id': s['neuron_id'],
            'type': 'pre' if s['type'] == 'syncenter' else 'post'
        }
        for i, s in enumerate(synaptic_sites)
    ]

    print("%d synaptic sites found in %.3fs"%(
        len(synaptic_sites),
        time.time() - start))

    return synaptic_sites

def store_synaptic_sites(
        roi,
        db_host,
        annotation_db_name,
        synapse_nodes_collection,
        target_db_name,
        target_collection_name):
    '''Get all relevant synaptic sites within ROI and store them with unique
    IDs in a separate collection.'''

    client = MongoClient(db_host)
    database = client[target_db_name]

    if target_collection_name in database.collection_names():
        print("synaptic site collection found, skipping import")
        return

    # get synaptic site -> skeleton ID
    synaptic_sites = read_synaptic_sites(
        db_host,
        annotation_db_name,
        synapse_nodes_collection,
        roi)

    collection = database[target_collection_name]

    collection.create_index(
        [
            ('z', ASCENDING),
            ('y', ASCENDING),
            ('x', ASCENDING)
        ],
        name='position')

    collection.create_index(
        [
            ('id', ASCENDING)
        ],
        name='id',
        unique=True)

    collection.insert_many(synaptic_sites)

def get_fragment_ids(fragments, synaptic_sites, roi):
    '''Get the fragment IDs of all the synaptic sites that are contained in the
    given ROI.'''

    synaptic_sites = list(synaptic_sites)

    if len(synaptic_sites) == 0:
        print("No synaptic sites in %s, skipping"%roi)
        return None

    print("Getting fragment IDs for %d synaptic sites in %s..."%(
        len(synaptic_sites), roi))

    # for a few sites, direct lookup is faster than memory copies
    if len(synaptic_sites) >= 15:

        print("Copying fragments into memory...")
        start = time.time()
        fragments = fragments[roi]
        fragments.materialize()
        print("%.3fs"%(time.time() - start,))

    print("Getting fragment IDs for synaptic sites in %s..."%roi)
    start = time.time()

    fragment_ids = [
        {
            'synaptic_site_id': site['id'],
            'fragment_id': int(fragments[daisy.Coordinate((site['z'], site['y'], site['x']))])
        }
        for site in synaptic_sites
    ]

    print("Got fragment IDs for %d sites in %.3fs"%(
        len(fragment_ids),
        time.time() - start))

    return fragment_ids

def store_fragment_ids_in_block(
        db_host,
        db_name,
        synaptic_sites_collection_name,
        fragment_map_collection_name,
        fragments,
        block):

    print("Finding fragment IDs in block %s"%block)

    # get synaptic sites
    client = MongoClient(db_host)
    database = client[db_name]
    synaptic_sites_collection = database[synaptic_sites_collection_name]

    bz, by, bx = block.read_roi.get_begin()
    ez, ey, ex = block.read_roi.get_end()

    synaptic_sites = synaptic_sites_collection.find(
        {
            'z': { '$gte': bz, '$lt': ez },
            'y': { '$gte': by, '$lt': ey },
            'x': { '$gte': bx, '$lt': ex }
        })

    # get synaptic site -> fragment ID
    fragment_ids = get_fragment_ids(
        fragments,
        synaptic_sites,
        block.write_roi)

    if fragment_ids is None:
        return

    client = MongoClient(db_host)
    database = client[db_name]
    collection = database[fragment_map_collection_name]

    collection.insert_many(fragment_ids)

def store_fragment_ids(
        roi,
        db_host,
        seg_db_name,
        synaptic_site_collection_name,
        synaptic_site_fragments_collection_name,
        fragments,
        num_workers):

    client = MongoClient(db_host)
    database = client[seg_db_name]

    if synaptic_site_fragments_collection_name not in database.collection_names():

        collection = database[synaptic_site_fragments_collection_name]

        collection.create_index(
            [
                ('synaptic_site_id', ASCENDING)
            ],
            name='synaptic_site_id',
            unique=True)

        collection.create_index(
            [
                ('fragment_id', ASCENDING)
            ],
            name='fragment_id')

    daisy.run_blockwise(
        roi,
        daisy.Roi((0, 0, 0), (10000, 10000, 5000)),
        daisy.Roi((0, 0, 0), (10000, 10000, 5000)),
        lambda b: store_fragment_ids_in_block(
            db_host,
            seg_db_name,
            synaptic_site_collection_name,
            synaptic_site_fragments_collection_name,
            fragments,
            b),
        num_workers=num_workers,
        fit='shrink')

def evaluate(
        fragments_file,
        fragments_dataset,
        db_host,
        seg_db_name,
        seg_collection,
        annotation_db_name,
        skeleton_nodes_collection,
        skeleton_edges_collection,
        synapse_nodes_collection,
        synapse_edges_collection):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    roi = fragments.roi

    store_synaptic_sites(
        roi,
        db_host,
        annotation_db_name,
        synapse_nodes_collection,
        seg_db_name,
        'synaptic_sites')

    store_fragment_ids(
        roi,
        db_host,
        seg_db_name,
        'synaptic_sites',
        'synaptic_site_fragments',
        fragments,
        num_workers=40)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
