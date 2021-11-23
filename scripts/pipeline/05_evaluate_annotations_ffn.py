from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi
from funlib.evaluate import \
    expected_run_length, \
    get_skeleton_lengths, \
    split_graph
from pymongo import MongoClient
from mask_calyx_skeletons import roi as calyx_mask_roi
import daisy
import glob
import json
import logging
import multiprocessing as mp
import networkx
import numpy as np
import os
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluateAnnotations():

    def __init__(
            self,
            experiment,
            setup,
            config_slab,
            segmentation_file,
            segmentation_dataset,
            scores_db_host,
            scores_db_name,
            annotations_db_host,
            annotations_db_name,
            annotations_skeletons_collection_name,
            roi_offset,
            roi_shape,
            thresholds_minmax,
            thresholds_step,
            annotations_synapses_collection_name=None,
            run_type=None,
            compute_mincut_metric=False,
            **kwargs):

        self.experiment = experiment
        self.setup = setup
        self.config_slab = config_slab
        self.segmentation_file = segmentation_file
        self.segmentation_dataset = segmentation_dataset
        self.scores_db_host = scores_db_host
        self.scores_db_name = scores_db_name
        self.annotations_db_host = annotations_db_host
        self.annotations_db_name = annotations_db_name
        self.annotations_skeletons_collection_name = \
            annotations_skeletons_collection_name
        self.annotations_synapses_collection_name = \
            annotations_synapses_collection_name
        self.roi = daisy.Roi(roi_offset, roi_shape)
        self.thresholds_minmax = thresholds_minmax
        self.thresholds_step = thresholds_step
        self.run_type = run_type
        self.compute_mincut_metric = compute_mincut_metric

        self.site_segment_lut_directory = os.path.join(
            self.segmentation_file,
            'luts/site_segment/new_gt_9_9_20/sub_rois')

        if self.run_type:
            logger.info("Run type set, evaluating on %s dataset", self.run_type)
            self.site_segment_lut_directory = os.path.join(
                    self.site_segment_lut_directory,
                    self.run_type)
            logger.info("Path to site segment luts: %s", self.site_segment_lut_directory)

        try:
            self.segments = daisy.open_ds(
                self.segmentation_file,
                self.segmentation_dataset,
                mode='r')
        except:
            self.segments = daisy.open_ds(
                self.segmentation_file,
                self.segmentation_dataset + '/s0',
                mode='r')

    def store_lut_in_block(self, block):

        logger.info("Finding segment IDs in block %s", block)

        # get all skeleton nodes (which include synaptic sites)
        client = MongoClient(self.annotations_db_host)
        database = client[self.annotations_db_name]
        skeletons_collection = \
            database[self.annotations_skeletons_collection_name + '.nodes']

        bz, by, bx = block.read_roi.get_begin()
        ez, ey, ex = block.read_roi.get_end()

        site_nodes = skeletons_collection.find(
            {
                'z': {'$gte': bz, '$lt': ez},
                'y': {'$gte': by, '$lt': ey},
                'x': {'$gte': bx, '$lt': ex}
            })

        # logger.info("Site nodes %s", list(site_nodes))

        # get site -> segment ID
        site_segment_lut = get_site_segment_lut(
            self.segments,
            site_nodes,
            block.write_roi)

        if site_segment_lut is None:
            return

        # store LUT
        block_lut_path = os.path.join(
            self.site_segment_lut_directory,
            str(block.block_id) + '.npz')
        np.savez_compressed(
            block_lut_path,
            site_segment_lut=site_segment_lut)

    def prepare_for_roi(self):

        logger.info("Preparing evaluation for ROI %s...", self.roi)

        self.skeletons = self.read_skeletons()

        # array with site IDs
        self.site_ids = np.array([
            n
            for n in self.skeletons.nodes()
        ], dtype=np.uint64)

        # array with component ID for each site
        self.site_component_ids = np.array([
            data['component_id']
            for _, data in self.skeletons.nodes(data=True) if 'component_id' in data
        ])
        assert self.site_component_ids.min() >= 0
        self.site_component_ids = self.site_component_ids.astype(np.uint64)
        self.number_of_components = np.unique(self.site_component_ids).size

        print(len(self.site_ids), len(self.site_component_ids))

        if self.annotations_synapses_collection_name:
            # create a mask that limits sites to synaptic sites
            logger.info("Creating synaptic sites mask...")
            start = time.time()
            client = MongoClient(self.annotations_db_host)
            database = client[self.annotations_db_name]
            synapses_collection = \
                database[self.annotations_synapses_collection_name + '.edges']
            synaptic_sites = synapses_collection.find()
            synaptic_sites = np.unique([
                s
                for ss in synaptic_sites
                for s in [ss['source'], ss['target']]
            ])
            self.synaptic_sites_mask = np.isin(self.site_ids, synaptic_sites)
            logger.info("%.3fs", time.time() - start)

        logger.info("Calculating skeleton lengths...")
        start = time.time()
        self.skeleton_lengths = get_skeleton_lengths(
                self.skeletons,
                skeleton_position_attributes=['z', 'y', 'x'],
                skeleton_id_attribute='component_id',
                store_edge_length='length')

        self.total_length = np.sum([l for _, l in self.skeleton_lengths.items()])

        logger.info("%.3fs", time.time() - start)

    def prepare_for_segments(self):
        '''Get the segment ID for each site in site_ids.'''

        logger.info(
            "Preparing evaluation for segments in %s...",
            self.segmentation_file)

        if not os.path.exists(self.site_segment_lut_directory):

            logger.info("site-segment LUT does not exist, creating it...")

            os.makedirs(self.site_segment_lut_directory)
            daisy.run_blockwise(
                self.roi,
                daisy.Roi((0,)*3, (9000,)*3),
                daisy.Roi((0,)*3, (9000,)*3),
                lambda b: self.store_lut_in_block(b),
                num_workers=32,
                fit='shrink')

        else:

            logger.info(
                "site-segment LUT already exists, skipping preparation")

        logger.info("Reading site-segment LUTs from %s...", self.site_segment_lut_directory)
        start = time.time()
        lut_files = glob.glob(
            os.path.join(
                self.site_segment_lut_directory,
                '*.npz'))
        site_segment_lut = np.concatenate(
            [
                np.load(f)['site_segment_lut']
                for f in lut_files
            ],
            axis=1)
        assert site_segment_lut.dtype == np.uint64
        logger.info(
            "Found %d sites in site-segment LUT",
            len(site_segment_lut[0]))
        logger.info("%.3fs", time.time() - start)

        # convert to dictionary
        site_segment_lut = {
            site: segment
            for site, segment in zip(
                site_segment_lut[0],
                site_segment_lut[1])
        }

        print('Length of site segment lut: %i'%len(site_segment_lut))
        print('Length of site ids: %i'%len(self.site_ids))

        # create segment ID array congruent to site_ids

      #   self.site_segment_ids = []

        # for s in self.site_ids:
            # if s in site_segment_lut:
                # self.site_segment_ids.append(site_segment_lut[s])
            # else:
                # print('id %i not in site_segment_lut'%s) 
                # self.site_segment_ids.append(0)

        # self.site_segment_ids = np.array([
            # site_segment_lut[s] for s in self.site_ids if s in site_segment_lut
        # ])

        # self.site_segment_ids = np.array(self.site_segment_ids)

        self.site_segment_ids = np.array([
            site_segment_lut[s] for s in self.site_ids
        ])

        print('Length of site segment ids: %i'%len(self.site_segment_ids))

        return self.site_segment_ids

    def read_skeletons(self):

       #  if self.roi != calyx_mask_roi:
            # logger.warn(
                # "Requested ROI %s differs from ROI %s, for which component "
                # "IDs have been generated. I hope you know what you are doing!",
                # self.roi, calyx_mask_roi)

        # node_mask = 'zebrafinch_mask'
        node_components = 'zebrafinch_components'

        if self.run_type:
            # logger.info("Using components for %s data", self.run_type)
            # node_mask = node_mask + '_' + self.run_type.strip('_masked_ffn')
            # node_components = node_components + '_' + self.run_type.strip('_not_masked')
            # node_mask = node_mask + '_' + self.run_type.strip('_ffn')
            node_components = node_components + '_' + self.run_type

        # logger.info("Reading mask from: %s", node_mask)
        # logger.info("Reading components from: %s", node_components)

        # get all skeletons that are masked in
        logger.info("Fetching all skeletons...")
        skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            self.annotations_db_name,
            self.annotations_db_host,
            nodes_collection=self.annotations_skeletons_collection_name +
            '.nodes',
            edges_collection=self.annotations_skeletons_collection_name +
            '.edges',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                node_components: ['component_id']
            })

        infinite_roi = daisy.Roi((-10e6,)*3, (20e6,)*3)

        start = time.time()
        skeletons = skeletons_provider.get_graph(
                self.roi)
        logger.info("Found %d skeleton nodes" % skeletons.number_of_nodes())
        logger.info("%.3fs", time.time() - start)

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

        for node,data in skeletons.nodes(data=True):
            if 'z' not in data:
                print(node, data)

        return skeletons

    def evaluate(self):

        self.prepare_for_roi()

        self.prepare_for_segments()

        thresholds = list(np.arange(
            self.thresholds_minmax[0],
            self.thresholds_minmax[1],
            self.thresholds_step))

        procs = []

        logger.info("Evaluating thresholds...")
        for threshold in thresholds:
            proc = mp.Process(
                target=lambda: self.evaluate_threshold(threshold)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    def compute_expected_run_length(self, site_segment_ids):

        logger.info("Calculating expected run length...")
        start = time.time()

        node_segment_lut = {
            site: segment for site, segment in zip(
                self.site_ids,
                site_segment_ids)
        }

        erl, stats = expected_run_length(
                skeletons=self.skeletons,
                skeleton_id_attribute='component_id',
                edge_length_attribute='length',
                node_segment_lut=node_segment_lut,
                skeleton_lengths=self.skeleton_lengths,
                return_merge_split_stats=True)

        perfect_lut = {
                node: data['component_id'] for node, data in \
                        self.skeletons.nodes(data=True)
        }

        max_erl, _ = expected_run_length(
                skeletons=self.skeletons,
                skeleton_id_attribute='component_id',
                edge_length_attribute='length',
                node_segment_lut=perfect_lut,
                skeleton_lengths=self.skeleton_lengths,
                return_merge_split_stats=True)

        split_stats = [
            {
                'comp_id': int(comp_id),
                'seg_ids': [(int(a), int(b)) for a, b in seg_ids]
            }
            for comp_id, seg_ids in stats['split_stats'].items()
        ]
        merge_stats = [
            {
                'seg_id': int(seg_id),
                'comp_ids': [int(comp_id) for comp_id in comp_ids]
            }
            for seg_id, comp_ids in stats['merge_stats'].items()
        ]

        logger.info("%.3fs", time.time() - start)

        return erl, max_erl, split_stats, merge_stats

    def compute_rand_voi(
            self,
            site_component_ids,
            site_segment_ids,
            return_cluster_scores=False):

        logger.info("Computing RAND and VOI...")
        start = time.time()

        rand_voi_report = rand_voi(
            np.array([[site_component_ids]]),
            np.array([[site_segment_ids]]),
            return_cluster_scores=return_cluster_scores)

        logger.info("VOI split: %f", rand_voi_report['voi_split'])
        logger.info("VOI merge: %f", rand_voi_report['voi_merge'])
        logger.info("%.3fs", time.time() - start)

        return rand_voi_report

    def evaluate_threshold(self, threshold):

        scores_client = MongoClient(self.scores_db_host)
        scores_db = scores_client[self.scores_db_name]
        scores_collection = scores_db['scores']

        number_of_segments = np.unique(self.site_segment_ids).size

        erl, max_erl, split_stats, merge_stats = self.compute_expected_run_length(self.site_segment_ids)

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        rand_voi_report = self.compute_rand_voi(
            self.site_component_ids,
            self.site_segment_ids,
            return_cluster_scores=True)

        if self.annotations_synapses_collection_name:
            synapse_rand_voi_report = self.compute_rand_voi(
                self.site_component_ids[self.synaptic_sites_mask],
                site_segment_ids[self.synaptic_sites_mask])

        print('ERL: ', erl)
        print('Max ERL: ', max_erl)
        print('Total path length: ', self.total_length)

        normalized_erl = erl/max_erl
        print('Normalized ERL: ', normalized_erl)

        report = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del report[k]

        if self.annotations_synapses_collection_name:
            report['synapse_voi_split'] = synapse_rand_voi_report['voi_split']
            report['synapse_voi_merge'] = synapse_rand_voi_report['voi_merge']

        report['expected_run_length'] = erl
        report['max_erl'] = max_erl
        report['normalized_erl'] = normalized_erl
        report['total path length'] = self.total_length
        report['number_of_segments'] = number_of_segments
        report['number_of_merging_segments'] = number_of_merging_segments
        report['number_of_split_skeletons'] = number_of_split_skeletons
        report['merge_stats'] = merge_stats
        report['split_stats'] = split_stats
        report['threshold'] = threshold
        report['experiment'] = self.experiment
        report['setup'] = self.setup
        report['config_slab'] = self.config_slab

        if self.run_type:
            report['run_type'] = self.run_type

        scores_collection.replace_one(
            filter={
                'config_slab': report['config_slab'],
                'threshold': report['threshold'],
                'run_type': report['run_type']
            },
            replacement=report,
            upsert=True)

        find_worst_split_merges(rand_voi_report)

def get_site_segment_lut(segments, sites, roi):
    '''Get the segment IDs of all the sites that are contained in the given
    ROI.'''

    sites = list(sites)

    if len(sites) == 0:
        logger.info("No sites in %s, skipping", roi)
        return None

    logger.info(
        "Getting segment IDs for %d synaptic sites in %s...",
        len(sites),
        roi)

    # for a few sites, direct lookup is faster than memory copies
    if len(sites) >= 15:

        logger.info("Copying segments into memory...")
        start = time.time()
        segments = segments[roi]
        segments.materialize()
        logger.info("%.3fs", time.time() - start)

    logger.info("Getting segment IDs for synaptic sites in %s...", roi)
    start = time.time()

    segment_ids = np.array([
        segments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    synaptic_site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    logger.info(
        "Got segment IDs for %d sites in %.3fs",
        len(segment_ids),
        time.time() - start)

    lut = np.array([synaptic_site_ids, segment_ids])

    return lut

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
        logger.info("\tcomponent %d\tVOI split %.5f" % (i, s))

    logger.info("10 worst merges:")
    for (s, i) in merges[-10:]:
        logger.info("\tsegment %d\tVOI merge %.5f" % (i, s))

if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate = EvaluateAnnotations(**config)
    evaluate.evaluate()



