from pymongo import MongoClient
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from parallel_mergewise_score import parallel_mergewise_score
import daisy
import json
import logging
import lsd
import malis
import numpy as np
import scipy
import sys
import waterz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(gt_file,
             gt_dataset,
             fragments_file,
             fragments_dataset,
             border_threshold,
             db_host,
             rag_db_name,
             edges_collection,
             scores_db_name,
             block_size,
             chunk_size,
             num_workers,
             thresholds_minmax,
             thresholds_step,
             configuration):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)
    total_roi = fragments.roi

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        rag_db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection)

    # open score DB
    client = MongoClient(db_host)
    database = client[scores_db_name]
    score_collection = database['scores']

    # slice
    logger.info("Reading RAG in {0}".format(total_roi))
    rag = rag_provider[total_roi]

    logger.info("Number of nodes in RAG: %d", len(rag.nodes()))
    logger.info("Number of edges in RAG: %d", len(rag.edges()))

    # read gt data and determine where we have both fragments and GT
    gt = daisy.open_ds(gt_file, gt_dataset) # NOTE: assuming renumbered GT
    common_roi = fragments.roi.intersect(gt.roi)
    logger.info("Found common ROI {0}".format(common_roi))

    # calculate thresholds and scores
    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    scores = parallel_mergewise_score(rag,
                                      fragments_file,
                                      fragments_dataset,
                                      gt_file,
                                      gt_dataset,
                                      common_roi,
                                      block_size,
                                      chunk_size,
                                      thresholds,
                                      num_workers,
                                      retry=2)

    for i in range(len(scores)):
        # get score values
        (voi_split, voi_merge) = scores[i]
        threshold = thresholds[i]
        #store values in db
        logger.info("Storing VOI values for threshold {0:.2f} in DB".format(threshold))
        logger.info("VOI split: {0:.5f} VOI merge: {1:.5f} ".format(voi_split, voi_merge))
        metrics = {'voi_split': voi_split,
                   'voi_merge': voi_merge,
                   'threshold': threshold}
        metrics.update(configuration)
        score_collection.insert(metrics)
        logger.info(metrics)
        
if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
