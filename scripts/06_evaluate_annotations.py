from funlib.segment.graphs import find_connected_components
from pymongo import MongoClient
from scipy.special import comb
import daisy
import json
import logging
import numpy as np
import sklearn.metrics
import sys
import time
import waterz

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

    lut = np.array([fragment_ids, synaptic_site_ids])

    return lut

def store_lut_in_block(
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        site_fragment_lut_directory,
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

    site_fragment_lut_directory = os.path.join(
        fragments_file,
        'luts/site_fragment')

    if os.path.exists(site_fragment_lut_directory):
        logger.warn("site-fragment LUT already exists, skipping preparation")
        return
    os.makedirs(site_fragment_collection_name)

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
            sites,
            b),
        num_workers=num_workers,
        fit='shrink')

def comb2(n):
    return comb(n, 2, exact=1)

def rand_index(labels_true, labels_pred):

    contingency = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, sparse=True)

    sum_squares_ij = sum(n_ij*n_ij for n_ij in contingency.data)
    sum_squares_i = sum(n_i*n_i for n_i in np.ravel(contingency.sum(axis=1)))
    sum_squares_j = sum(n_j*n_j for n_j in np.ravel(contingency.sum(axis=0)))

    rand_split = sum_squares_ij/sum_squares_i
    rand_merge = sum_squares_ij/sum_squares_j

    return rand_split, rand_merge

def evaluate(
        fragments_file,
        fragments_dataset,
        fragment_segment_lut,
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

    annotations_client = MongoClient(annotations_db_host)
    annotations_database = annotations_client[annotations_db_name]

    # get all skeletons
    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
        annotations_db_name,
        annotations_db_host,
        nodes_collection=annotations_skeletons_collection_name + '.nodes',
        edges_collection=annotations_skeletons_collection_name + '.edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'])

    print("Fetching all skeletons...")
    start = time.time()
    skeletons = skeletons_provider[roi]
    print("Found %d skeleton nodes" % skeletons.number_of_nodes())
    print("%.3fs"%(time.time() - start))

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

    # array with skeleton component ID for each site
    # (does not change between thresholds)
    component_ids = np.array([
        skeletons.nodes[site]['component_id']
        for site in site_fragment_lut[0]
    ])

    for threshold in np.arange(
            thresholds_minmax[0],
            thresholds_minmax[1],
            thresholds_step):

        fragment_segment_lut_file = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment',
            'seg_%s_%d.npz'%(edges_collection,int(threshold*100)))

        # get fragment-segment LUT
        print("Reading fragment-segment LUT...")
        start = time.time()
        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']
        print("%.3fs"%(time.time() - start))

        # get the segment ID for each site
        print("Mapping sites to segments...")
        start = time.time()
        segment_ids = funlib.segment.arrays.replace_values(
            site_fragment_lut[1],
            fragment_segment_lut[0],
            fragment_segment_lut[1])
        print("%.3fs"%(time.time() - start))

        # compute RAND index

        print("%d synaptic sites associated with segments"%segment_ids.size)

        arand = sklearn.metrics.adjusted_rand_score(component_ids, segment_ids)
        print("ARI: %.5f"%arand)

        rand_split, rand_merge = rand_index(component_ids, segment_ids)
        print("RI split %.5f"%rand_split)
        print("RI merge %.5f"%rand_merge)
        print("RI total %.5f"%(rand_split + rand_merge))

        # waterz evaluation
        report = waterz.evaluate(
            np.array([[component_ids]]),
            np.array([[segment_ids]]))
        print(report)

        # # get most merging segments

        # segment_to_neurons = {}
        # for site in synaptic_sites:
            # segment_id = site['segment_id']
            # if segment_id is None:
                # continue
            # if segment_id not in segment_to_neurons:
                # segment_to_neurons[segment_id] = set()
            # segment_to_neurons[site['segment_id']].add(site['neuron_id'])

        # merges = list(segment_to_neurons.items())
        # merges.sort(key=lambda x: len(x[1]))

        # print("Largest mergers:")
        # for m in merges[-10:]:
            # print("%d: merges %d neurons"%(m[0], len(m[1])))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
