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
        volume_size,
        threshold):

    #read gt data
    gt = daisy.open_ds(gt_file, gt_dataset)
    seg = daisy.open_ds(seg_file, seg_dataset)

    logger.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    logger.info("Converting seg to nd array...")
    seg = seg.to_ndarray()

    #open score DB
    client = MongoClient(db_host)
    database = client[scores_db_name]
    score_collection = database['scores']

    #get VOI and RAND
    logger.info("Calculating VOI scores for ffn...")
    rand_voi_report = rand_voi(
            gt,
            seg,
            return_cluster_scores=False)

    metrics = rand_voi_report.copy()

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    logger.info("Storing VOI values for ffn in DB")

    metrics['threshold'] = threshold
    metrics['experiment'] = experiment
    metrics['setup'] = setup
    metrics['network'] = configuration
    metrics['volume_size'] = volume_size

    logger.info(metrics)

    score_collection.replace_one(
            filter={
                'network': metrics['network'],
                'volume_size': metrics['volume_size'],
                'threshold': metrics['threshold']
            },
            replacement=metrics,
            upsert=True)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
