from funlib.segment.graphs import find_connected_components
from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi
from pymongo import MongoClient
# from scipy.special import comb
import daisy
import json
import logging
import numpy as np
import multiprocessing as mp
# import sklearn.metrics
import sys
import time
import os
import glob

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

def get_site_fragment_lut(fragments, synaptic_sites, roi):
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

    fragment_ids = np.array([
        fragments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in synaptic_sites
    ])
    synaptic_site_ids = np.array(
        [site['id'] for site in synaptic_sites],
        dtype=np.uint64)

    print("Got fragment IDs for %d sites in %.3fs"%(
        len(fragment_ids),
        time.time() - start))

    lut = np.array([synaptic_site_ids, fragment_ids])

    return lut

def store_lut_in_block(
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        site_fragment_lut_directory,
        fragments,
        block):

    print("Finding fragment IDs in block %s"%block)

    # get all skeleton nodes (which include synaptic sites)
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
            'x': { '$gte': bx, '$lt': ex }
        })

    # get synaptic site -> fragment ID
    site_fragment_lut = get_site_fragment_lut(
        fragments,
        site_nodes,
        block.write_roi)

    if site_fragment_lut is None:
        return

    # store LUT
    block_lut_path = os.path.join(
        site_fragment_lut_directory,
        str(block.block_id) + '.npz')
    np.savez_compressed(block_lut_path, site_fragment_lut=site_fragment_lut)

def prepare_evaluate(
        fragments_file,
        fragments_dataset,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        **kwargs):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')
    roi = fragments.roi

    site_fragment_lut_directory = os.path.join(
        fragments_file,
        'luts/site_fragment')

    if os.path.exists(site_fragment_lut_directory):
        logger.warn("site-fragment LUT already exists, skipping preparation")
        return
    os.makedirs(site_fragment_lut_directory)

    # 2. store site->fragement ID LUT in file, for each block
    daisy.run_blockwise(
        roi,
        daisy.Roi((0, 0, 0), (10000, 10000, 10000)),
        daisy.Roi((0, 0, 0), (10000, 10000, 10000)),
        lambda b: store_lut_in_block(
            annotations_db_host,
            annotations_db_name,
            annotations_skeletons_collection_name,
            site_fragment_lut_directory,
            fragments,
            b),
        num_workers=40,
        fit='shrink')

def evaluate(
        experiment,
        setup,
        iteration,
        fragments_file,
        fragments_dataset,
        edges_db_name,
        edges_collection,
        scores_db_host,
        scores_db_name,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name,
        roi_offset,
        roi_shape,
        thresholds_minmax,
        thresholds_step,
        **kwargs):

    prepare_evaluate(
        fragments_file,
        fragments_dataset,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        annotations_synapses_collection_name)


    roi = daisy.Roi(roi_offset, roi_shape)

    # get all skeletons
    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
        annotations_db_name,
        annotations_db_host,
        nodes_collection=annotations_skeletons_collection_name + '.nodes',
        edges_collection=annotations_skeletons_collection_name + '.edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={'calyx_neuropil_mask': ['masked']})

    print("Fetching all skeletons...")
    start = time.time()
    skeletons = skeletons_provider[roi]
    print("Found %d skeleton nodes" % skeletons.number_of_nodes())
    print("%.3fs"%(time.time() - start))

    # remove outside edges and nodes
    remove_nodes = []
    for node, data in skeletons.nodes(data=True):
        if 'z' not in data or not data['masked']:
            remove_nodes.append(node)
    print("Removing %d nodes that were outside of ROI or masked"%len(remove_nodes))
    for node in remove_nodes:
        skeletons.remove_node(node)

    # relabel connected components
    print("Relabeling skeleton components...")
    start = time.time()
    find_connected_components(
        skeletons,
        'component_id',
        return_lut=False)
    print("%.3fs"%(time.time() - start))

    # get all synaptic sites with their fragment ID
    print("Fetching site-fragment LUT...")
    start = time.time()
    lut_files = glob.glob(
        os.path.join(
            fragments_file,
            'luts/site_fragment/*.npz'))
    site_fragment_lut = np.concatenate(
        [
            np.load(f)['site_fragment_lut']
            for f in lut_files
        ],
        axis=1)
    print("Found %d synaptic sites in site-fragment LUT" % len(site_fragment_lut[0]))
    print("%.3fs"%(time.time() - start))

    # limit to relevant sites within requested ROI
    relevant_mask = np.array([
        skeletons.has_node(site)
        for site in site_fragment_lut[0]
    ])
    site_fragment_lut = site_fragment_lut[:,relevant_mask]
    print("Limited to %d synaptic sites within ROI"%len(site_fragment_lut[0]))
    if len(site_fragment_lut[0]) < 100:
        print("Found sites: ", site_fragment_lut[0])

    # array with skeleton component ID for each site
    # (does not change between thresholds)
    component_ids = np.array([
        skeletons.nodes[site]['component_id']
        for site in site_fragment_lut[0]
    ])

    scores_config = {
            'experiment': experiment,
            'setup': setup,
            'iteration': iteration,
            'network_configuration': edges_db_name,
            'merge_function': edges_collection.strip('edges_')
            }

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    procs = []

    start = time.time()
    for threshold in thresholds:
        proc = mp.Process(
                target=evaluate_parallel,
                args=(
                    fragments_file,
                    edges_collection,
                    site_fragment_lut,
                    component_ids,
                    threshold,
                    scores_config,
                    scores_db_host,
                    scores_db_name
                    ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

def evaluate_parallel(
        fragments_file,
        edges_collection,
        site_fragment_lut,
        component_ids,
        threshold,
        scores_config,
        scores_db_host,
        scores_db_name):

        scores_client = MongoClient(scores_db_host)
        scores_db = scores_client[scores_db_name]
        scores_collection = scores_db['scores']


        # get fragment-segment LUT
        print("Reading fragment-segment LUT...")
        start = time.time()
        fragment_segment_lut_file = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment',
            'seg_%s_%d.npz'%(edges_collection,int(threshold*100)))
        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']
        print("%.3fs"%(time.time() - start))

        # get the segment ID for each site
        print("Mapping sites to segments...")
        start = time.time()
        segment_ids = replace_values(
            site_fragment_lut[1],
            fragment_segment_lut[0],
            fragment_segment_lut[1])
        print("%.3fs"%(time.time() - start))

        num_not_changed = (segment_ids == site_fragment_lut[1]).sum()
        print("%d site segments have same ID as fragment"%num_not_changed)

        # compute RAND index

        print("%d synaptic sites associated with segments"%segment_ids.size)

        report = rand_voi(
            np.array([[component_ids]]),
            np.array([[segment_ids]]),
            return_cluster_scores=True)

        remove_keys = {'voi_split_i', 'voi_merge_j'}
        updated_report = report.copy()
        for k in remove_keys:
            updated_report.pop(k, None)

        updated_report.update({'threshold': threshold})
        updated_report.update(scores_config)
        scores_collection.insert(updated_report)

        print("Calculating scores for threshold: ", threshold)
        print("VOI split: ", report['voi_split'])
        print("VOI merge: ", report['voi_merge'])

        # get most severe splits/merges
        splits = sorted([(s, i) for (i, s) in report['voi_split_i'].items()])
        merges = sorted([(s, j) for (j, s) in report['voi_merge_j'].items()])

        print("10 worst splits:")
        for (s, i) in splits[-10:]:
            print("\tcomponent %d\tVOI split %.5f" % (i, s))

        print("10 worst merges:")
        for (s, i) in merges[-10:]:
            print("\tsegment %d\tVOI merge %.5f" % (i, s))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
