from neuclease.dvid import round_box
from neuclease.dvid import fetch_instance_info, fetch_roi, fetch_synapses_in_batches, determine_point_rois, fetch_labels_batched

from neuclease import configure_default_logging
configure_default_logging()

import logging
logger = logging.getLogger(__name__)

import sys

PARALLELISM = 16

def fetch_roi_synapses(server, uuid, synapses_instance, roi):
    """
    Fetch the coordinates and body labels for 
    all synapses that fall within the given ROI.

    Args:

        server:
            DVID server, e.g. 'emdata4:8900'

        uuid:
            DVID uuid, e.g. 'abc9'

        synapses_instance:
            DVID synapses instance name, e.g. 'synapses'

        roi:
            DVID ROI instance name, e.g. 'EB'

    Returns:
        pandas DataFrame with columns:
        ['z', 'y', 'x', 'kind', 'conf', 'body']

    """
    # Determine name of the segmentation instance that's
    # associated with the given synapses instance.
    syn_info = fetch_instance_info(server, uuid, synapses_instance)
    seg_instance = syn_info["Base"]["Syncs"][0]

    logger.info(f"Fetching mask for ROI: '{roi}'")
    # Fetch the ROI as a low-res array (scale 5, i.e. 32-px resolution)
    roi_mask_s5, roi_box_s5 = fetch_roi(server, uuid, roi, 'mask')

    # Convert to full-res box
    roi_box = (2**5) * roi_box_s5

    # The synapses ROI must be 64-px-aligned
    roi_box = round_box(roi_box, 64, 'out')

    logger.info("Fetching synapse points")
    # points_df is a DataFrame with columns for [z,y,x]
    points_df, _partners_df = fetch_synapses_in_batches(server, uuid, synapses_instance, roi_box, processes=PARALLELISM)

    # Append a 'roi_name' column to points_df
    logger.info("Labeling ROI for each point")
    determine_point_rois(server, uuid, [roi], points_df, roi_mask_s5, roi_box_s5)

    logger.info("Discarding points that don't overlap with the roi")
    points_df = points_df.query('roi == @roi').copy()

    logger.info("Fetching body label under each point")
    bodies = fetch_labels_batched(server, uuid, seg_instance, points_df[['z', 'y', 'x']].values, processes=PARALLELISM)
    points_df['body'] = bodies

    return points_df[['z', 'y', 'x', 'kind', 'conf', 'body']]

if __name__ == "__main__":
    eb_synapses_df = fetch_roi_synapses('emdata4.int.janelia.org:8900', '3c281', 'synapses', sys.argv[1])

    logger.info("Writing %s-synapses.csv",sys.argv[1])
    eb_synapses_df.to_csv('%s-synapses.csv' % sys.argv[1], index=False, header=True)

    logger.info("DONE")
