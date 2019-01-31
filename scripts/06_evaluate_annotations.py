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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        seg_db_host,
        seg_db_name,
        edges_collection,
        annotations_db_host,
        annotations_db_name,
        annotations_skeletons_collection_name,
        roi_offset,
        roi_shape,
        thresholds_minmax,
        thresholds_step,
        **kwargs):

    roi = daisy.Roi(roi_offset, roi_shape)

    seg_client = MongoClient(seg_db_host)
    seg_database = seg_client[seg_db_name]
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
    site_fragment_collection = seg_database['synaptic_site_fragments']
    site_fragment_lut = list(site_fragment_collection.find())
    site_fragment_lut = {
        s['synaptic_site_id']: s['fragment_id']
        for s in site_fragment_lut
        if skeletons.has_node(s['synaptic_site_id'])
    }
    print("Found %d synaptic sites" % len(site_fragment_lut))
    print("%.3fs"%(time.time() - start))

    print("Getting all relevant fragment IDs...")
    start = time.time()
    fragment_ids = np.unique(np.array(list(site_fragment_lut.values())))
    print("Found %d relevant fragment IDs" % len(fragment_ids))
    print("%.3fs"%(time.time() - start))

    # array with skeleton component ID for each site
    # (does not change between thresholds)
    component_ids = np.array([
        skeletons.nodes[site]['component_id']
        for site in site_fragment_lut.keys()
    ])

    for threshold in np.arange(
            thresholds_minmax[0],
            thresholds_minmax[1],
            thresholds_step):

        lut_collection_name = 'seg_%s_%d'%(edges_collection,int(threshold*100))


        # get fragment-segment LUT
        print("Fetching fragment-segment LUT...")
        start = time.time()
        lut_collection = seg_database[lut_collection_name]
        fragment_segment_lut = list(lut_collection.find({
            'fragment': {'$in': list(int(x) for x in fragment_ids) }
        }))
        fragment_segment_lut = {
            f['fragment']: f['segment']
            for f in fragment_segment_lut
        }
        print("%.3fs"%(time.time() - start))

        # add segment ID to each site
        print("Mapping sites to segments...")
        start = time.time()
        site_segment_lut = {
            s: fragment_segment_lut[f]
            for s, f in site_fragment_lut.items()
        }
        print("%.3fs"%(time.time() - start))

        # array with segment ID for each site
        segment_ids = np.array([
            site_segment_lut[site]
            for site in site_fragment_lut.keys()
        ])

        # compute RAND index

        print("%d synaptic sites associated with segments"%segment_ids.size)

        arand = sklearn.metrics.adjusted_rand_score(component_ids, segment_ids)
        print("ARI: %.5f"%arand)

        rand_split, rand_merge = rand_index(component_ids, segment_ids)
        print("RI split %.5f"%rand_split)
        print("RI merge %.5f"%rand_merge)
        print("RI total %.5f"%(rand_split + rand_merge))

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
