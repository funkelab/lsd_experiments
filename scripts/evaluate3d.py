from pymongo import MongoClient
import daisy
import json
import logging
import lsd
import malis
import numpy as np
import os
import scipy
import sys
import waterz

logging.basicConfig(level=logging.INFO)

def evaluate(
        experiment,
        setup,
        iteration,
        sample,
        border_threshold,
        db_host,
        db_name,
        thresholds):

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    predict_file = os.path.join(predict_dir, sample)

    # open fragments
    fragments = daisy.open_ds(predict_file, 'volumes/fragments')

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r')

    #open score DB

    client = MongoClient(db_host)
    database = client[db_name]
    score_collection = database['scores']

    total_roi = fragments.roi

    # slice
    print("Reading fragments and RAG in %s"%total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi]

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
    fragments = fragments[common_roi]
    gt = gt[common_roi]
    print("Cropped fragments and GT to common ROI %s"%common_roi)

    #relabel connected components
    components = gt.data
    dtype = components.dtype
    simple_neighborhood = malis.mknhood3d()
    affs_from_components = malis.seg_to_affgraph(
            components,
            simple_neighborhood
            )
    components, _ = malis.connected_components_affgraph(
            affs_from_components,
            simple_neighborhood
            )
    # curate GT
    components[gt.data>np.uint64(-10)] = 0
    gt.data = components.astype(dtype)

    print('Relabeled connected components')
   
   #check connected components
    '''print("Writing connected components gt...")
    ground_truth = daisy.prepare_ds(
        predict_file,
        'volumes/gt_connected_components',
        gt.roi,
        gt.voxel_size,
        gt.data.dtype)
    ground_truth.data[:] = gt.data'''


    for z in range(gt.data.shape[0]):
        border_mask = create_border_mask_2d(
            gt.data[z],
            float(border_threshold)/gt.voxel_size[1])
        gt.data[z][border_mask] = 0

    print('Created 2d border mask')
    
    #check border mask
    '''print("Writing border mask gt...")
    ground_truth = daisy.prepare_ds(
        predict_file,
        'volumes/gt_border_mask',
        gt.roi,
        gt.voxel_size,
        gt.data.dtype)
    ground_truth.data[:] = gt.data'''

    for threshold in thresholds:

        segmentation = fragments.data.copy()

        # create a segmentation
        print("Creating segmentation for threshold %f..."%threshold)
        rag.get_segmentation(threshold, segmentation)

        # store segmentation
        '''print("Writing segmentation...")
        seg = daisy.prepare_ds(
            predict_file,
            'volumes/segmentation',
            fragments.roi,
            fragments.voxel_size,
            fragments.data.dtype)
        seg.data[:] = segmentation.data'''

        #get VOI and RAND
        print("Calculating VOI scores for threshold %f..."%threshold)
        metrics = waterz.evaluate(segmentation, gt.data)

        #store values in db
        print("Storing VOI and RAND values for threshold %f in DB" %threshold)
        metrics.update({'threshold': threshold})
        score_collection.insert(metrics)

        print(metrics)

def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and( image == padded[:-2, 1:-1], image == padded[2:, 1:-1] ),
        np.logical_and( image == padded[1:-1, :-2], image == padded[1:-1, 2:] )
        )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
        )

    return distances <= max_dist

def create_border_mask_3d(volume, max_dist):
    """
    Create binary border mask for volume.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_voxels = np.logical_and(
        np.logical_and( voxels == padded[:-2, 1:-1], voxels == padded[2:, 1:-1] ),
        np.logical_and( voxels == padded[1:-1, :-2], voxels == padded[1:-1, 2:] )
        )

    distances = scipy.ndimage.distance_transform_edt(
        border_voxels,
        return_distances=True,
        return_indices=False
        )

    return distances <= max_dist

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
