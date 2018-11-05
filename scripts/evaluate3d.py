from pymongo import MongoClient
from parallel_read_rag import parallel_read_rag
from parallel_relabel import parallel_relabel
from parallel_renumber import parallel_renumber
from parallel_score import parallel_score
from skimage.measure import label
from skimage.morphology import remove_small_objects
import daisy
import json
import logging
import lsd
import time
import malis
import shutil
import numpy as np
import os
import scipy
import sys
import waterz

logging.basicConfig(level=logging.INFO)

def evaluate(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        block_size,
        chunk_size,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        num_workers,
        retry):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)
    
    # open score DB

    client = MongoClient(db_host)
    scores_database = client[scores_db_name]
    score_collection = scores_database['scores']

    #read gt data
    gt = daisy.open_ds(gt_file, gt_dataset)

    # evaluate only where we have both fragments and GT
    voxel_size = gt.voxel_size
    common_roi = fragments.roi.intersect(gt.roi)
    logging.info("Cropped fragments and GT ROIs to common ROI %s"%common_roi)
    
    # open RAG in parallel
    logging.info("Reading RAG in %s"%fragments.roi)
    rag = parallel_read_rag(
            common_roi,
            db_host,
            rag_db_name,
            edges_collection,
            block_size,
            num_workers,
            retry)

    logging.info("Number of nodes in RAG: %d"%(len(rag.nodes())))
    logging.info("Number of edges in RAG: %d"%(len(rag.edges())))

    renumbered_gt_dataset = 'volumes/labels/renumbered_neuron_ids'

    if not os.path.isdir(os.path.join(fragments_file, renumbered_gt_dataset)):
        gt = gt[common_roi]
        gt.materialize()
        #relabel connected components in common ROI
        logging.info("Relabelling connected components in GT...")
        components = gt.data
        dtype = components.dtype
        relabeled_components = label(components, connectivity=1)
        relabeled_components = remove_small_objects(relabeled_components, min_size=2, in_place=True)
        logging.info("Equivalent to original GT: {}".format(np.all(gt.data == relabeled_components)))
        logging.info("Done relabeling with skimage")
        # curate GT
        gt.data = relabeled_components.astype(dtype)
        renumbered_gt = daisy.prepare_ds(fragments_file,
                                         renumbered_gt_dataset,
                                         common_roi,
                                         gt.voxel_size,
                                         gt.data.dtype)
        renumbered_gt[common_roi] = gt.data
        logging.info('Stored relabeled GT connected components')
    
    gt = daisy.open_ds(fragments_file, renumbered_gt_dataset)
    
    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    for threshold in thresholds:

        # create a segmentation
        logging.info("Creating segmentation for threshold %f..."%threshold)
        seg_dataset = 'volumes/segmentation_{0:.2f}'.format(threshold)
        seg_components = rag.get_connected_components(threshold)
        seg_counts_shape = (int(10e7), 1)
        gt_seg_counts_shape = (1, int(10e7))
        contingencies_shape = (int(10e7), int(10e7))
        
        parallel_relabel(
                seg_components,
                fragments_file,
                fragments_dataset,
                common_roi,
                block_size,
                fragments_file,
                seg_dataset,
                num_workers=num_workers,
                retry=retry)

        # get VOI and RAND
        logging.info("Calculating VOI scores for threshold {0}...".format(threshold))
        (voi_split, voi_merge) = parallel_score(
                fragments_file,
                seg_dataset,
                renumbered_gt_dataset,
                common_roi,
                block_size,
                chunk_size,
                seg_counts_shape,
                gt_seg_counts_shape,
                contingencies_shape,
                num_workers=8,
                retry=retry)
        
        # store values in db
        logging.info("Storing VOI values for threshold %f in DB" %threshold)
        metrics = {'voi_split': voi_split, 'voi_merge': voi_merge, 'threshold': threshold}
        score_collection.insert(metrics)
        logging.info("Threshold: {0} VOI split: {1} VOI merge: {2}".format(
            threshold, voi_split, voi_merge))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
