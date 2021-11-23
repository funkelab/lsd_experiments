#supress tensorflow future warnings lol
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
logger = logging.getLogger(__name__)

def evaluate(
        experiment,
        setup,
        gt_file,
        gt_dataset,
        seg_file,
        seg_dataset,
        db_host,
        scores_db_name,
        configuration,
        volume,
        threshold,
        iteration=None,
        roi_offset=None,
        roi_shape=None):

    logging.info("Reading seg  from %s" %seg_file)

    try:
        seg = daisy.open_ds(seg_file, seg_dataset)
    except:
        seg = daisy.open_ds(seg_file, seg_dataset + '/s0')

    logging.info("Reading gt from %s" %gt_file)

    try:
        gt = daisy.open_ds(gt_file, gt_dataset)
    except:
        gt = daisy.open_ds(gt_file, gt_dataset + '/s0')

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = seg.roi.intersect(gt.roi)

    # evaluate only where we have both seg and GT
    logging.info("Cropping seg and GT to common ROI %s", common_roi)
    seg = seg[common_roi]
    gt = gt[common_roi]

    logging.info("Converting seg to nd array...")
    seg = seg.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    #open score DB
    client = MongoClient(db_host)
    database = client[scores_db_name]
    score_collection = database['scores']

    #get VOI and RAND
    logger.info("Calculating VOI scores for ffn...")
    rand_voi_report = rand_voi(
            gt,
            seg,
            return_cluster_scores=True)

    metrics = rand_voi_report.copy()

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    logging.info("Storing VOI values for threshold %f in DB" %threshold)

    metrics['threshold'] = threshold
    metrics['experiment'] = experiment
    metrics['setup'] = setup
    metrics['method'] = configuration
    metrics['run_type'] = volume

    print('VOI split: ', metrics['voi_split'])
    print('VOI merge: ', metrics['voi_merge'])

    logging.info(metrics)

    score_collection.replace_one(
            filter={
                'method': metrics['method'],
                'run_type': metrics['run_type'],
                'threshold': metrics['threshold']
            },
            replacement=metrics,
            upsert=True)


    find_worst_split_merges(rand_voi_report)

def find_worst_split_merges(rand_voi_report):

    # get most severe splits/merges
    splits = sorted([
        (s, i)
        for (i, s) in rand_voi_report['voi_split_i'].items()
    ])
    merges = sorted([
        (s, j)
        for (j, s) in rand_voi_report['voi_merge_j'].items()
    ])

    logger.info("10 worst splits:")
    for (s, i) in splits[-10:]:
        logging.info("\tcomponent %d\tVOI split %.5f" % (i, s))

    logger.info("10 worst merges:")
    for (s, i) in merges[-10:]:
        logging.info("\tsegment %d\tVOI merge %.5f" % (i, s))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
