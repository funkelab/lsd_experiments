from pymongo import MongoClient
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

def evaluate(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        border_threshold,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        configuration):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        rag_db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection)

    #open score DB

    client = MongoClient(db_host)
    database = client[scores_db_name]
    score_collection = database['scores']

    total_roi = fragments.roi

    # slice
    logger.info("Reading fragments and RAG in %s", total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi]

    logger.info("Number of nodes in RAG: %d", len(rag.nodes()))
    logger.info("Number of edges in RAG: %d", len(rag.edges()))

    #read gt data
    gt = daisy.open_ds(gt_file, gt_dataset)
    common_roi = fragments.roi.intersect(gt.roi)

    # evaluate only where we have both fragments and GT
    logger.info("Cropping fragments and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    #relabel connected components
    logger.info("Relabelling connected components in GT...")
    gt.materialize()
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

    logger.info("Creating 2D border mask...")
    for z in range(gt.data.shape[0]):
        border_mask = create_border_mask_2d(
            gt.data[z],
            float(border_threshold)/gt.voxel_size[1])
        gt.data[z][border_mask] = 0

    # DEBUG
    #
    # logger.info("Writing GT to debug file...")
    # ground_truth = daisy.prepare_ds(
        # 'debug_evaluation.zarr',
        # 'volumes/gt',
        # gt.roi,
        # gt.voxel_size,
        # gt.data.dtype)
    # ground_truth.data[:] = gt.data
    # logger.info("Writing fragments to debug file...")
    # fragments_ds = daisy.prepare_ds(
        # 'debug_evaluation.zarr',
        # 'volumes/fragments',
        # fragments.roi,
        # fragments.voxel_size,
        # fragments.data.dtype)
    # fragments_ds.data[:] = fragments.data

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    for threshold in thresholds:

        segmentation = fragments.to_ndarray()

        # create a segmentation
        logger.info("Creating segmentation for threshold %f...", threshold)
        rag.get_segmentation(threshold, segmentation)

        # DEBUG
        #
        # logger.info("Writing segmentation to debug file...")
        # seg = daisy.prepare_ds(
            # 'debug_evaluation.zarr',
            # 'volumes/segmentation_%d'%(threshold*100),
            # fragments.roi,
            # fragments.voxel_size,
            # fragments.data.dtype)
        # seg.data[:] = segmentation

        #get VOI and RAND
        logger.info("Calculating VOI scores for threshold %f...", threshold)
        metrics = waterz.evaluate(segmentation, gt.data)

        voi_sum = np.array(metrics['voi_split'] + metrics['voi_merge'], dtype=np.float32)
        a_rand = np.array(1.0 - (2.0*metrics['rand_split']*metrics['rand_merge'])/(metrics['rand_split']+metrics['rand_merge']), dtype=np.float32)
        cremi_score = np.sqrt(np.array(voi_sum*a_rand).astype(float))

        logger.info('VOI sum: %f', voi_sum)
        logger.info('Adapted rand: %f', a_rand)
        logger.info('CREMI score: %f', cremi_score)

        #store values in db
        logger.info("Storing VOI and RAND values for threshold %f in DB" %threshold)
        metrics.update({
            'threshold': threshold,
            'cremi_score': cremi_score
        })
        metrics.update(configuration)
        score_collection.insert(metrics)

        logger.info(metrics)

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

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
