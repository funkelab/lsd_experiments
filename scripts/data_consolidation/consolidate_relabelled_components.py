import daisy
import logging
import numpy as np
import pymongo
import sys
from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def consolidate_components(
        block,
        old_seg,
        relabelled_seg,
        graph,
        out_ds):

    cropped = seg.to_ndarray(block.write_roi)

    # cropped_ds[block.write_roi] = cropped

def consolidate(
        in_file,
        seg_ds,
        relabelled_ds,
        out_file,
        out_ds,
        graph,
        num_workers):

    logging.info('Loading segs...')

    old_seg = daisy.open_ds(in_file, seg_ds)
    relabelled_seg = daisy.open_ds(in_file, relabelled_ds)

    total_roi = relabelled_seg.roi

    read_roi = daisy.Roi((0, 0, 0), (3600, 3600, 3600))
    write_roi = read_roi

    out_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    old_seg.voxel_size,
                    dtype=old_seg.dtype,
                    write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        crop_roi,
        read_roi,
        write_roi,
        process_function=lambda b: consolidate_components(
            b,
            old_seg,
            relabelled_seg,
            graph,
            out_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)

def read_skeletons(
        db_host,
        db_name,
        roi):

    print("Opening graph db...")
    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection='zebrafinch.nodes',
        edges_collection='zebrafinch.edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={
            'zebrafinch_components_debug_roi_not_masked_relabelled': ['component_id'],
        })

    skeletons = skeletons_provider.get_graph(roi)
    logger.info("Found %d skeleton nodes" % skeletons.number_of_nodes())

    # remove outside edges and nodes
    remove_nodes = []
    for node, data in skeletons.nodes(data=True):
        if 'z' not in data:
            remove_nodes.append(node)
        else:
            assert data['component_id'] >= 0

    logger.info(
        "Removing %d nodes that were outside of ROI",
        len(remove_nodes))

    for node in remove_nodes:
        skeletons.remove_node(node)

    return skeletons

def get_site_segment_lut(segments, sites, roi):
    '''Get the segment IDs of all the sites that are contained in the given
    ROI.'''

    sites = list(sites)

    print(sites)

    if len(sites) == 0:
        logger.info("No sites in %s, skipping", roi)
        return None

    # logger.info("Getting segment IDs for %d synaptic sites in %s...",len(sites),roi)

    # for a few sites, direct lookup is faster than memory copies
    if len(sites) >= 15:

        logger.info("Copying segments into memory...")
        segments = segments[roi]
        segments.materialize()

    logger.info("Getting segment IDs for synaptic sites in %s...", roi)

    segment_ids = np.array([
        segments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    # logger.info("Got segment IDs for %d sites in %.3fs",len(segment_ids))

    lut = np.array([site_ids, segment_ids])

    return lut

if __name__ == '__main__':

    db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
    db_name = 'zebrafinch_gt_skeletons_new_gt_9_9_20_testing'

    non_relabelled_seg = daisy.open_ds(sys.argv[1], 'volumes/debug_seg/s0')
    relabelled_seg = daisy.open_ds(sys.argv[1], 'volumes/debug_seg_relabelled/s0')

    roi = relabelled_seg.roi

    # skels = read_skeletons(db_host, db_name, roi)

    # site_nodes = [data for _, data in skels.nodes(data=True)]

    client = pymongo.MongoClient(db_host)
    database = client[db_name]
    skeletons_collection = database['zebrafinch.nodes']

    bz, by, bx = roi.get_begin()
    ez, ey, ex = roi.get_end()

    site_nodes = skeletons_collection.find(
        {
            'z': {'$gte': bz, '$lt': ez},
            'y': {'$gte': by, '$lt': ey},
            'x': {'$gte': bx, '$lt': ex}
        })


  #   nr_site_segment_lut = get_site_segment_lut(
            # non_relabelled_seg,
            # site_nodes,
            # roi)

    r_site_segment_lut = get_site_segment_lut(
            relabelled_seg,
            site_nodes,
            roi)

    print(r_site_segment_lut)

   #  for a,b,c,d in zip(
            # nr_site_segment_lut[0],
            # r_site_segment_lut[0],
            # nr_site_segment_lut[1],
            # r_site_segment_lut[1]):
        # print(a,b,c,d)



