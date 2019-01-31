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
        seg_db_host,
        seg_db_name,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        site_fragments_collection_name,
        fragments,
        sites,
        block):

    print("Finding fragment IDs in block %s"%block)

    # get synaptic sites
    client = MongoClient(annotations_db_host)
    database = client[annotations_db_name]
    skeletons_collection = \
            database[annotations_skeletons_collection_name + '.nodes']

    bz, by, bx = block.read_roi.get_begin()
    ez, ey, ex = block.read_roi.get_end()

    site_nodes = skeletons_collection.find(
        {
            'z': { '$gte': bz, '$lt': ez },
            'y': { '$gte': by, '$lt': ey },
            'x': { '$gte': bx, '$lt': ex },
            'id': { '$in': sites }
        })

    # get synaptic site -> fragment ID
    fragment_ids = get_fragment_ids(
        fragments,
        site_nodes,
        block.write_roi)

    if fragment_ids is None:
        return

    client = MongoClient(seg_db_host)
    database = client[seg_db_name]
    collection = database[site_fragments_collection_name]

    collection.insert_many(fragment_ids)

def store_fragment_ids(
        roi,
        seg_db_host,
        seg_db_name,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        site_fragments_collection_name,
        fragments,
        num_workers):

    # 1. find all synaptic sites
    client = MongoClient(annotations_db_host)
    database = client[annotations_db_name]
    synapses = database[annotations_synapses_collection_name + '.edges']

    sites = synapses.find()
    sites = list(set(
        s
        for ss in sites
        for s in [ss['source'], ss['target']]
    ))

    # 2. for each site, get fragment ID

    # 3. store site->fragement ID in site_fragments_collection_name

    client = MongoClient(seg_db_host)
    database = client[seg_db_name]

    if site_fragments_collection_name not in database.collection_names():

        collection = database[site_fragments_collection_name]

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
            seg_db_host,
            seg_db_name,
            annotations_db_host,
            annotations_db_name,
            annotations_skeletons_collection_name,
            annotations_synapses_collection_name,
            site_fragments_collection_name,
            fragments,
            sites,
            b),
        num_workers=num_workers,
        fit='shrink')

def prepare_evaluate(
        fragments_file,
        fragments_dataset,
        seg_db_host,
        seg_db_name,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        **kwargs):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    roi = fragments.roi

    store_fragment_ids(
        roi,
        seg_db_host,
        seg_db_name,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        'synaptic_site_fragments',
        fragments,
        num_workers=1)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    prepare_evaluate(**config)
