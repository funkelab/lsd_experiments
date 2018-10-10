from pymongo import MongoClient
from parallel_read_rag import parallel_read_rag
from parallel_relabel import parallel_relabel
from parallel_renumber import parallel_renumber
from parallel_score import parallel_score
import daisy
import json
import logging
import lsd
import malis
import shutil
import numpy as np
import os
import scipy
import sys
import waterz

logging.basicConfig(level=logging.INFO)

def prune_and_renum(seg,
                    sizeThreshold=1):
    # renumber the components in descending order by size
    segId,segSizes = np.unique(seg, return_counts=True)
    descOrder = np.argsort(segSizes)[::-1]
    renum = np.zeros(int(segId.max()+1),dtype=np.uint64)
    segId = segId[descOrder]
    segSizes = segSizes[descOrder]
    renum[segId] = np.arange(1,len(segId)+1, dtype=np.uint64)

    if sizeThreshold>0:
        renum[segId[segSizes<=sizeThreshold]] = 0
        segSizes = segSizes[segSizes>sizeThreshold]

    seg = renum[seg]
    return (seg, segSizes)

def evaluate(
        experiment,
        setup,
        iteration,
        sample,
        block_size,
        chunk_size,
        border_threshold,
        db_host,
        db_name,
        thresholds,
        num_workers,
        retry):

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    predict_file = os.path.join(predict_dir, sample)

    # open fragments
    fragments = daisy.open_ds(predict_file, 'volumes/fragments')
    
    #open score DB

    client = MongoClient(db_host)
    database = client[db_name]
    score_collection = database['scores']

    # slice
    print("Reading RAG in %s"%fragments.roi)
    # open RAG in parallel
    rag = parallel_read_rag(
            experiment,
            setup,
            iteration,
            sample,
            db_host,
            db_name,
            block_size,
            num_workers,
            retry)

    print("Number of nodes in RAG: %d"%(len(rag.nodes())))
    print("Number of edges in RAG: %d"%(len(rag.edges())))

    #read gt data
    gt_file = os.path.join(
        experiment_dir,
        '01_data',
        sample)

    gt = daisy.open_ds(gt_file, 'volumes/labels/neuron_ids')

    # evaluate only where we have both fragments and GT
    common_roi = fragments.roi.intersect(gt.roi)
    voxel_size = gt.voxel_size
    print("Cropped fragments and GT to common ROI %s"%common_roi)

    
    logging.info("Constructing temporary segmentation dataset file")
    tmp_fname = 'tmp_seg.n5'
    tmp_seg_name = 'volumes/segmentation'
    tmp_gt_seg_name = 'volumes/labels/neuron_ids'
    if os.path.isdir(tmp_fname):
        shutil.rmtree(tmp_fname)
    shutil.copytree(gt.data.path, os.path.join(tmp_fname, tmp_gt_seg_name))
    
    relabel connected components
     num_neurons = parallel_renumber(
         gt_segmentation_ds,
         common_roi,
         voxel_size,
         block_size,
         num_workers,
         retry)

    print('Relabeled GT connected components')
  
    for threshold in thresholds:

        # create a segmentation
        print("Creating segmentation for threshold %f..."%threshold)
        seg_components = rag.get_connected_components(threshold)
        seg_counts_shape = (len(seg_components)+1, 1)
        gt_seg_counts_shape = (1, int(10e7))
        contingencies_shape = (len(seg_components)+1, int(10e7))
        parallel_relabel(
                seg_components,
                fragments,
                common_roi,
                block_size,
                tmp_fname,
                tmp_seg_name,
                num_workers=num_workers,
                retry=retry)

        #get VOI and RAND
        # segmentation = daisy.open_ds(tmp_fname, tmp_dsname)[fragments.roi].data
        # print("Calculating VOI scores for threshold %f..."%threshold)
        # metrics = waterz.evaluate(segmentation, gt.data)
        print("Calculating VOI scores for threshold {0}...".format(threshold))
        (voi_split, voi_merge) = parallel_score(
                tmp_fname,
                tmp_seg_name,
                tmp_gt_seg_name,
                common_roi,
                block_size,
                chunk_size,
                seg_counts_shape,
                gt_seg_counts_shape,
                contingencies_shape,
                num_workers=8,
                retry=retry)

        # #store values in db
        # print("Storing VOI and RAND values for threshold %f in DB" %threshold)
        # metrics.update({'threshold': threshold})
        # score_collection.insert(metrics)
        print((threshold, voi_split, voi_merge))

        # print(metrics)


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
