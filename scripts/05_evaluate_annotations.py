from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi
from funlib.evaluate import expected_run_length, get_skeleton_lengths
from pymongo import MongoClient
from mask_skeletons import roi as calyx_mask_roi
import daisy
import json
import logging
import numpy as np
import multiprocessing as mp
import sys
import time
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # get site -> fragment ID
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
        num_workers=10,
        fit='shrink')

def read_skeletons(
        annotations_db_name,
        annotations_db_host,
        annotations_skeletons_collection_name,
        roi):

    if roi != calyx_mask_roi:
        logger.warn(
            "Requested ROI %s differs from ROI %s, for which component "
            "IDs have been generated. I hope you know what you are doing!",
            roi, calyx_mask_roi)

    # get all skeletons that are masked in
    print("Fetching all skeletons...")
    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
        annotations_db_name,
        annotations_db_host,
        nodes_collection=annotations_skeletons_collection_name + '.nodes',
        edges_collection=annotations_skeletons_collection_name + '.edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={
            'calyx_neuropil_mask': ['masked'],
            'calyx_neuropil_components': ['component_id'],
        })

    start = time.time()
    skeletons = skeletons_provider.get_graph(
        roi,
        nodes_filter={'masked': True})
    print("Found %d skeleton nodes" % skeletons.number_of_nodes())
    print("%.3fs"%(time.time() - start))

    # remove outside edges and nodes
    remove_nodes = []
    for node, data in skeletons.nodes(data=True):
        if 'z' not in data:
            remove_nodes.append(node)
        else:
            assert data['masked'] == True
            assert data['component_id'] >= 0
    print("Removing %d nodes that were outside of ROI"%len(remove_nodes))
    for node in remove_nodes:
        skeletons.remove_node(node)

    return skeletons

def evaluate(
        experiment,
        setup,
        iteration,
        config_slab,
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

    print("Preparing site-fragment LUT...")
    start = time.time()
    prepare_evaluate(
        fragments_file,
        fragments_dataset,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name)
    print("%.3fs"%(time.time() - start))

    roi = daisy.Roi(roi_offset, roi_shape)

    print("Reading skeletons...")
    start = time.time()
    skeletons = read_skeletons(
        annotations_db_name,
        annotations_db_host,
        annotations_skeletons_collection_name,
        roi)
    print("%.3fs"%(time.time() - start))

    print("Reading site-fragment LUT...")
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
    print("Found %d sites in site-fragment LUT" % len(site_fragment_lut[0]))
    print("%.3fs"%(time.time() - start))

    # limit to relevant sites within requested ROI and mask
    relevant_mask = np.array([
        skeletons.has_node(site)
        for site in site_fragment_lut[0]
    ])
    site_fragment_lut = site_fragment_lut[:,relevant_mask]
    print("Limited site-fragment LUT to %d relevant sites within ROI"%len(site_fragment_lut[0]))
    if len(site_fragment_lut[0]) < 100:
        print("Found sites: ", site_fragment_lut[0])

    # array with component ID for each site (does not change between
    # thresholds)
    component_ids = np.array([
        skeletons.nodes[site]['component_id']
        for site in site_fragment_lut[0]
    ])

    # create a mask (for the site-fragment-LUT) that limits it to synaptic sites
    print("Creating synaptic sites mask...")
    start = time.time()
    client = MongoClient(annotations_db_host)
    database = client[annotations_db_name]
    synapses_collection = database[annotations_synapses_collection_name + '.edges']
    synaptic_sites = synapses_collection.find()
    synaptic_sites = np.unique([
        s
        for ss in synaptic_sites
        for s in [ss['source'], ss['target']]
    ])
    synaptic_sites_mask = np.isin(site_fragment_lut[0], synaptic_sites)
    print("%.3fs"%(time.time() - start))

    print("Calcluating skeleton lengths...")
    start = time.time()
    skeleton_lengths = get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes=['z', 'y', 'x'],
            skeleton_id_attribute='component_id',
            store_edge_length='length')
    print("%.3fs"%(time.time() - start))

    scores_config = {
            'experiment': experiment,
            'setup': setup,
            'iteration': iteration,
            'network_configuration': edges_db_name,
            'merge_function': edges_collection.strip('edges_'),
            'config_slab': config_slab
            }

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    procs = []

    print("Evaluating thresholds...")
    start = time.time()
    for threshold in thresholds:
        proc = mp.Process(
                target=evaluate_parallel,
                args=(
                    fragments_file,
                    edges_collection,
                    site_fragment_lut,
                    synaptic_sites_mask,
                    component_ids,
                    skeletons,
                    skeleton_lengths,
                    threshold,
                    scores_config,
                    scores_db_host,
                    scores_db_name
                    ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    print("%.3fs"%(time.time() - start))

def evaluate_parallel(
        fragments_file,
        edges_collection,
        site_fragment_lut,
        synaptic_sites_mask,
        component_ids,
        skeletons,
        skeleton_lengths,
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

        number_of_segments = np.unique(segment_ids).size

        num_not_changed = (segment_ids == site_fragment_lut[1]).sum()
        print("%d site segments have same ID as fragment"%num_not_changed)

        node_segment_lut = {
            site: segment for site, segment in zip(site_fragment_lut[0], segment_ids)
        }

        print("Calculating expected run length...")
        start = time.time()

        ######## Something goes wrong when calculating erl, all entries are zero

        erl, stats = expected_run_length(
                skeletons=skeletons,
                skeleton_id_attribute='component_id',
                edge_length_attribute='length',
                node_segment_lut=node_segment_lut,
                skeleton_lengths=skeleton_lengths,
                return_merge_split_stats=True)
        print("%.3fs"%(time.time() - start))

        ######################################################

        ######## If we comment the below two lines out, we get the cython buffer dtype mismatch error when calculating rand_voi

        component_ids = component_ids.astype(np.uint64)
        segment_ids = segment_ids.astype(np.uint64)

        ######################################################

        print("Computing RAND and VOI on skeletons and synaptic sites...")
        start = time.time()
        report = rand_voi(
            np.array([[component_ids]]),
            np.array([[segment_ids]]),
            return_cluster_scores=True)
        print("VOI split: ", report['voi_split'])
        print("VOI merge: ", report['voi_merge'])

        remove_keys = {'voi_split_i', 'voi_merge_j'}
        updated_report = report.copy()
        for k in remove_keys:
            updated_report.pop(k, None)

        synapse_report = rand_voi(
            np.array([[component_ids[synaptic_sites_mask]]]),
            np.array([[segment_ids[synaptic_sites_mask]]]))
        print("%.3fs"%(time.time() - start))

        stats['merge_stats'] = {
            int(seg_id): [int(comp_id) for comp_id in comp_ids]
            for seg_id, comp_ids in stats['merge_stats'].items()
        }
        stats['split_stats'] = {
            int(comp_id): [(int(a), int(b)) for a, b in seg_ids]
            for comp_id, seg_ids in stats['split_stats'].items()
        }

        number_of_merging_segments = len(stats['merge_stats'])
        number_of_split_skeletons = len(stats['split_stats'])

        # TODO: this should be replaced with a min-cut iterative proof-reading
        # simulation
        splits_needed = 0
        for k, v in stats['merge_stats'].items():
            splits_needed += (len(v) - 1)
        print('Splits needed to fix merges: ', splits_needed, type(splits_needed))

        average_splits_needed = splits_needed/number_of_segments
        print(
            'Average splits needed per segment: ',
            average_splits_needed,
            type(average_splits_needed))

        merges_needed = 0
        for split_list in stats['split_stats'].values():
            merges_needed += len(split_list)
        average_merges_needed = merges_needed/number_of_split_skeletons

        print('Merges needed to fix splits: ', merges_needed, type(merges_needed))
        print(
            'Average merges needed per skeleton: ',
            average_merges_needed,
            type(average_merges_needed))

        ######## If we comment the below line out, we receive the error that keys need to be strings

        stats = convert_keys_to_string(stats)

        ##########################################################

        updated_report['synapse_voi_split'] = synapse_report['voi_split']
        updated_report['synapse_voi_merge'] = synapse_report['voi_merge']
        updated_report['expected_run_length'] = erl
        updated_report['number_of_segments'] = number_of_segments
        updated_report['number_of_merging_segments'] = number_of_merging_segments
        updated_report['number_of_split_skeletons'] = number_of_split_skeletons
        updated_report['total_splits_needed_to_fix_merges'] = splits_needed
        updated_report['average_splits_needed_to_fix_merges'] = average_splits_needed
        updated_report['total_merges_needed_to_fix_splits'] = merges_needed
        updated_report['average_merges_needed_to_fix_splits'] = average_merges_needed

        ######## If we add the below two lines to the updated report we receive BSON invalid doc error

        # updated_report['merge_stats'] = stats['merge_stats']
        # updated_report['split_stats'] = stats['split_stats']

        ##########################################################

        updated_report.update({'threshold': threshold})
        updated_report.update(scores_config)

        ######## We should ensure unique doc entries with network_configuration, merge_function, threshold

        scores_collection.insert(updated_report)

        ##########################################################

        # get most severe splits/merges
        splits = sorted([(s, i) for (i, s) in report['voi_split_i'].items()])
        merges = sorted([(s, j) for (j, s) in report['voi_merge_j'].items()])

        print("10 worst splits:")
        for (s, i) in splits[-10:]:
            print("\tcomponent %d\tVOI split %.5f" % (i, s))

        print("10 worst merges:")
        for (s, i) in merges[-10:]:
            print("\tsegment %d\tVOI merge %.5f" % (i, s))

def convert_keys_to_string(d):

    new_dict = {}

    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_keys_to_string(v)
        new_dict[str(k)] = v

    return new_dict

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
