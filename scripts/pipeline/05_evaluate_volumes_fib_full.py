from pymongo import MongoClient
import daisy
import json
import logging
import lsd
# import malis
import numpy as np
import os
# import scipy
import sys
import waterz
from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi

logging.basicConfig(level=logging.INFO)

def evaluate(
        experiment,
        setup,
        iteration,
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        num_workers,
        configuration,
        volume,
        roi_offset=None,
        roi_shape=None):

    # open fragments
    logging.info("Reading fragments from %s" %fragments_file)
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    # logging.info("Opening RAG DB...")
   #  rag_provider = daisy.persistence.MongoDbGraphProvider(
        # rag_db_name,
        # host=db_host,
        # mode='r',
        # edges_collection=edges_collection,
        # position_attribute=['center_z', 'center_y', 'center_x'])
    # logging.info("RAG DB opened")

    # total_roi = fragments.roi

    # total_roi = daisy.Roi((40576, 0, 0), (23016, 53248, 53248))

    # slice
    # logging.info("Reading fragments and RAG in %s", total_roi)
    # fragments = fragments[total_roi]
    # rag = rag_provider[total_roi]

    # logging.info("Number of nodes in RAG: %d", len(rag.nodes()))
    # logging.info("Number of edges in RAG: %d", len(rag.edges()))

    #read gt data
    gt = daisy.open_ds(gt_file, gt_dataset)

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = fragments.roi.intersect(gt.roi)

    read_roi = daisy.Roi((0, 0, 0), (4096, 4096, 4096))
    write_roi = daisy.Roi((0, 0, 0), (4096, 4096, 4096))

    # evaluate only where we have both fragments and GT
    logging.info("Cropping fragments and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    logging.info("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    logging.info("Evaluating thresholds...")
    for threshold in thresholds:

        segment_ids = get_segmentation(
                fragments,
                fragments_file,
                edges_collection,
                threshold)

        evaluate_threshold(
                experiment,
                setup,
                iteration,
                db_host,
                scores_db_name,
                edges_collection,
                segment_ids,
                gt,
                threshold,
                configuration,
                volume)

def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold):

    logging.info("Loading fragment - segment lookup table for threshold %s..." %threshold)
    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            'luts_full/fragment_segment')

    fragment_segment_lut_file = os.path.join(
            fragment_segment_lut_dir,
            'seg_%s_%d.npz' % (edges_collection, int(threshold*100)))

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids

def evaluate_threshold(
        experiment,
        setup,
        iteration,
        db_host,
        scores_db_name,
        edges_collection,
        segment_ids,
        gt,
        threshold,
        configuration,
        volume):

        #open score DB
        client = MongoClient(db_host)
        database = client[scores_db_name]
        score_collection = database['scores']

        #get VOI and RAND
        logging.info("Calculating VOI scores for threshold %f...", threshold)

        logging.info(type(segment_ids))

        rand_voi_report = rand_voi(
                gt,
                segment_ids,
                return_cluster_scores=False)

        metrics = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        logging.info("Storing VOI values for threshold %f in DB" %threshold)

        metrics['threshold'] = threshold
        metrics['experiment'] = experiment
        metrics['setup'] = setup
        metrics['iteration'] = iteration
        metrics['network'] = configuration
        metrics['volume'] = volume
        metrics['merge_function'] = edges_collection.strip('edges_')

        logging.info(metrics)

        score_collection.replace_one(
                filter={
                    'network': metrics['network'],
                    'volume': metrics['volume'],
                    'merge_function': metrics['merge_function'],
                    'threshold': metrics['threshold']
                },
                replacement=metrics,
                upsert=True)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
