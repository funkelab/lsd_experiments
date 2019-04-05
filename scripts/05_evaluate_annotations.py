from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi
from funlib.evaluate import \
    expected_run_length, \
    get_skeleton_lengths, \
    split_graph
from pymongo import MongoClient
from mask_skeletons import roi as calyx_mask_roi
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
            iteration,
            config_slab,
            fragments_file,
            fragments_dataset,
            edges_db_name,
            edges_collection,
            scores_db_host,
            scores_db_name,
            annotations_db_host,
            annotations_db_name,
            annotations_skeletons_collection_name,
            annotations_synapses_collection_name,
            roi_offset,
            roi_shape,
            thresholds_minmax,
            thresholds_step,
            compute_mincut_metric=False,
            **kwargs):

        self.experiment = experiment
        self.setup = setup
        self.iteration = iteration
        self.config_slab = config_slab
        self.fragments_file = fragments_file
        self.fragments_dataset = fragments_dataset
        self.edges_db_name = edges_db_name
        self.edges_collection = edges_collection
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
        self.compute_mincut_metric = compute_mincut_metric

        self.site_fragment_lut_directory = os.path.join(
            self.fragments_file,
            'luts/site_fragment')
        self.fragments = daisy.open_ds(
            self.fragments_file,
            self.fragments_dataset,
            mode='r')

    def store_lut_in_block(self, block):

        logger.info("Finding fragment IDs in block %s", block)

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

        # get site -> fragment ID
        site_fragment_lut = get_site_fragment_lut(
            self.fragments,
            site_nodes,
            block.write_roi)

        if site_fragment_lut is None:
            return

        # store LUT
        block_lut_path = os.path.join(
            self.site_fragment_lut_directory,
            str(block.block_id) + '.npz')
        np.savez_compressed(
            block_lut_path,
            site_fragment_lut=site_fragment_lut)

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
            for _, data in self.skeletons.nodes(data=True)
        ])
        assert self.site_component_ids.min() >= 0
        self.site_component_ids = self.site_component_ids.astype(np.uint64)
        self.number_of_components = np.unique(self.site_component_ids).size

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

        logger.info("Calcluating skeleton lengths...")
        start = time.time()
        self.skeleton_lengths = get_skeleton_lengths(
                self.skeletons,
                skeleton_position_attributes=['z', 'y', 'x'],
                skeleton_id_attribute='component_id',
                store_edge_length='length')
        logger.info("%.3fs", time.time() - start)

    def prepare_for_fragments(self):
        '''Get the fragment ID for each site in site_ids.'''

        logger.info(
            "Preparing evaluation for fragments in %s...",
            self.fragments_file)

        if not os.path.exists(self.site_fragment_lut_directory):

            logger.info("site-fragment LUT does not exist, creating it...")

            os.makedirs(self.site_fragment_lut_directory)
            daisy.run_blockwise(
                self.roi,
                daisy.Roi((0, 0, 0), (10000, 10000, 10000)),
                daisy.Roi((0, 0, 0), (10000, 10000, 10000)),
                lambda b: self.store_lut_in_block(b),
                num_workers=10,
                fit='shrink')

        else:

            logger.info(
                "site-fragment LUT already exists, skipping preparation")

        logger.info("Reading site-fragment LUT...")
        start = time.time()
        lut_files = glob.glob(
            os.path.join(
                self.fragments_file,
                'luts/site_fragment/*.npz'))
        site_fragment_lut = np.concatenate(
            [
                np.load(f)['site_fragment_lut']
                for f in lut_files
            ],
            axis=1)
        assert site_fragment_lut.dtype == np.uint64
        logger.info(
            "Found %d sites in site-fragment LUT",
            len(site_fragment_lut[0]))
        logger.info("%.3fs", time.time() - start)

        # convert to dictionary
        site_fragment_lut = {
            site: fragment
            for site, fragment in zip(
                site_fragment_lut[0],
                site_fragment_lut[1])
        }

        # create fragment ID array congruent to site_ids
        self.site_fragment_ids = np.array([
            site_fragment_lut[s] for s in self.site_ids
        ])

    def read_skeletons(self):

        if self.roi != calyx_mask_roi:
            logger.warn(
                "Requested ROI %s differs from ROI %s, for which component "
                "IDs have been generated. I hope you know what you are doing!",
                self.roi, calyx_mask_roi)

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
                'calyx_neuropil_mask': ['masked'],
                'calyx_neuropil_components': ['component_id'],
            })

        start = time.time()
        skeletons = skeletons_provider.get_graph(
            self.roi,
            nodes_filter={'masked': True})
        logger.info("Found %d skeleton nodes" % skeletons.number_of_nodes())
        logger.info("%.3fs", time.time() - start)

        # remove outside edges and nodes
        remove_nodes = []
        for node, data in skeletons.nodes(data=True):
            if 'z' not in data:
                remove_nodes.append(node)
            else:
                assert data['masked']
                assert data['component_id'] >= 0

        logger.info(
            "Removing %d nodes that were outside of ROI",
            len(remove_nodes))
        for node in remove_nodes:
            skeletons.remove_node(node)

        return skeletons

    def evaluate(self):

        self.prepare_for_roi()

        self.prepare_for_fragments()

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

    def get_site_segment_ids(self, threshold):

        # get fragment-segment LUT
        logger.info("Reading fragment-segment LUT...")
        start = time.time()
        fragment_segment_lut_file = os.path.join(
            self.fragments_file,
            'luts',
            'fragment_segment',
            'seg_%s_%d.npz' % (self.edges_collection, int(threshold*100)))
        fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']
        assert fragment_segment_lut.dtype == np.uint64
        logger.info("%.3fs", time.time() - start)

        # get the segment ID for each site
        logger.info("Mapping sites to segments...")
        start = time.time()
        site_mask = np.isin(fragment_segment_lut[0], self.site_fragment_ids)
        site_segment_ids = replace_values(
            self.site_fragment_ids,
            fragment_segment_lut[0][site_mask],
            fragment_segment_lut[1][site_mask])
        logger.info("%.3fs", time.time() - start)

        return site_segment_ids, fragment_segment_lut

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

        return erl, split_stats, merge_stats

    def compute_splits_merges_needed(
            self,
            fragment_segment_lut,
            site_segment_ids,
            split_stats,
            merge_stats,
            threshold):

        total_splits_needed = 0
        total_additional_merges_needed = 0
        total_unsplittable_fragments = []

        logger.info("Computing min-cut metric for each merging segment...")

        for i, merge in enumerate(merge_stats):

            logger.info("Processing merge %d/%d...", i + 1, len(merge_stats))
            (
                splits_needed,
                additional_merges_needed,
                unsplittable_fragments) = self.mincut_metric(
                    fragment_segment_lut,
                    site_segment_ids,
                    merge['seg_id'],
                    merge['comp_ids'],
                    threshold)
            total_splits_needed += splits_needed
            total_additional_merges_needed += additional_merges_needed
            total_unsplittable_fragments += unsplittable_fragments

        total_merges_needed = 0
        for split in split_stats:
            total_merges_needed += len(split['seg_ids']) - 1
        total_merges_needed += total_additional_merges_needed

        return (
            total_splits_needed,
            total_merges_needed,
            total_unsplittable_fragments)

    def mincut_metric(
            self,
            fragment_segment_lut,
            site_segment_ids,
            segment_id,
            component_ids,
            threshold):

        # get RAG for segment ID
        rag = self.get_segment_rag(segment_id, fragment_segment_lut, threshold)

        logger.info("Preparing RAG for split_graph call")
        start = time.time()

        # replace merge_score with weight
        for _, _, data in rag.edges(data=True):
            data['weight'] = 1.0 - data['merge_score']

        # find fragments for each component in segment_id
        component_fragments = {}

        # True for every site that maps to segment_id
        segment_mask = site_segment_ids == segment_id

        for component_id in component_ids:

            # limit following to sites that are part of component_id and
            # segment_id
            component_mask = self.site_component_ids == component_id
            mask = np.logical_and(component_mask, segment_mask)
            site_ids = self.site_ids[mask]
            site_fragment_ids = self.site_fragment_ids[mask]

            component_fragments[component_id] = site_fragment_ids

            for site_id, fragment_id in zip(site_ids, site_fragment_ids):

                # For each fragment containing a site, we need a position for
                # the split_graph call. We just take the position of the
                # skeleton node that maps to it, if there are several, we take
                # the last one.
                site_data = self.skeletons.nodes[site_id]
                fragment = rag.nodes[fragment_id]
                fragment['z'] = site_data['z']
                fragment['y'] = site_data['y']
                fragment['x'] = site_data['x']

                # Keep track of how many components share a fragment. If it is
                # more than one, this fragment is unsplittable.
                if 'component_ids' not in fragment:
                    fragment['component_ids'] = set()
                fragment['component_ids'].add(component_id)

        # find all unsplittable fragments...
        unsplittable_fragments = []
        for fragment_id, data in rag.nodes(data=True):
            if 'component_ids' in data and len(data['component_ids']) > 1:
                unsplittable_fragments.append(fragment_id)
        # ...and remove them from the component lists
        for component_id in component_ids:

            fragment_ids = component_fragments[component_id]
            valid_mask = np.logical_not(
                np.isin(
                    fragment_ids,
                    unsplittable_fragments))
            valid_fragment_ids = fragment_ids[valid_mask]
            if len(valid_fragment_ids) > 0:
                component_fragments[component_id] = valid_fragment_ids
            else:
                del component_fragments[component_id]

        logger.info(
            "%d fragments are merging and can not be split",
            len(unsplittable_fragments))

        if len(component_fragments) <= 1:
            logger.info(
                "after removing unsplittable fragments, there is nothing to "
                "do anymore")
            return 0, 0, unsplittable_fragments

        # these are the fragments that need to be split
        split_fragments = list(component_fragments.values())

        logger.info("Preparation took %.3fs", time.time() - start)

        logger.info(
            "Splitting segment into %d components with sizes %s",
            len(split_fragments), [len(c) for c in split_fragments])

        logger.info("Calling split_graph...")
        start = time.time()

        # call split_graph
        num_splits_needed = split_graph(
            rag,
            split_fragments,
            position_attributes=['z', 'y', 'x'],
            weight_attribute='weight',
            split_attribute='split')

        logger.info("split_graph took %.3fs", time.time() - start)

        logger.info(
            "%d splits needed for segment %d",
            num_splits_needed,
            segment_id)

        # get number of additional merges needed after splitting the current
        # segment
        #
        # this is the number of split labels per component minus 1
        additional_merges_needed = 0
        for component, fragments in component_fragments.items():
            split_ids = np.unique([rag.node[f]['split'] for f in fragments])
            additional_merges_needed += len(split_ids) - 1

        logger.info(
            "%d additional merges needed to join components again",
            additional_merges_needed)

        return (
            num_splits_needed,
            additional_merges_needed,
            unsplittable_fragments)

    def get_segment_rag(self, segment_id, fragment_segment_lut, threshold):

        logger.info("Reading RAG for segment %d", segment_id)
        start = time.time()

        rag_provider = daisy.persistence.MongoDbGraphProvider(
            self.edges_db_name,
            self.annotations_db_host,
            mode='r',
            edges_collection=self.edges_collection,
            position_attribute=['z', 'y', 'x'])

        # get all fragments for the given segment
        segment_mask = fragment_segment_lut[1] == segment_id
        fragment_ids = fragment_segment_lut[0][segment_mask]

        # get the RAG containing all fragments
        nodes = [
            {'id': fragment_id, 'segment_id': segment_id}
            for fragment_id in fragment_ids
        ]
        edges = rag_provider.read_edges(self.roi, nodes=nodes)

        logger.info("RAG contains %d nodes/%d edges", len(nodes), len(edges))

        rag = networkx.Graph()
        node_list = [
            (n['id'], {'segment_id': n['segment_id']})
            for n in nodes
        ]
        edge_list = [
            (e['u'], e['v'], {'merge_score': e['merge_score']})
            for e in edges
            if e['merge_score'] <= threshold
        ]
        rag.add_nodes_from(node_list)
        rag.add_edges_from(edge_list)
        rag.remove_nodes_from([
            n
            for n, data in rag.nodes(data=True)
            if 'segment_id' not in data])

        logger.info(
            "after filtering dangling node and not merged edges "
            "RAG contains %d nodes/%d edges",
            rag.number_of_nodes(),
            rag.number_of_edges())

        logger.info("Reading RAG took %.3fs", time.time() - start)

        return rag

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

        site_segment_ids, fragment_segment_lut = \
            self.get_site_segment_ids(threshold)

        number_of_segments = np.unique(site_segment_ids).size

        erl, split_stats, merge_stats = self.compute_expected_run_length(
            site_segment_ids)

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        if self.compute_mincut_metric:

            splits_needed, merges_needed, unsplittable_fragments = \
                self.compute_splits_merges_needed(
                    fragment_segment_lut,
                    site_segment_ids,
                    split_stats,
                    merge_stats,
                    threshold)

            average_splits_needed = splits_needed/number_of_segments
            average_merges_needed = merges_needed/self.number_of_components

        rand_voi_report = self.compute_rand_voi(
            self.site_component_ids,
            site_segment_ids,
            return_cluster_scores=True)

        synapse_rand_voi_report = self.compute_rand_voi(
            self.site_component_ids[self.synaptic_sites_mask],
            site_segment_ids[self.synaptic_sites_mask])

        report = rand_voi_report.copy()
        for k in {'voi_split_i', 'voi_merge_j'}:
            del report[k]
        report['synapse_voi_split'] = synapse_rand_voi_report['voi_split']
        report['synapse_voi_merge'] = synapse_rand_voi_report['voi_merge']
        report['expected_run_length'] = erl
        report['number_of_segments'] = number_of_segments
        report['number_of_merging_segments'] = number_of_merging_segments
        report['number_of_split_skeletons'] = number_of_split_skeletons
        if self.compute_mincut_metric:
            report['total_splits_needed_to_fix_merges'] = splits_needed
            report['average_splits_needed_to_fix_merges'] = average_splits_needed
            report['total_merges_needed_to_fix_splits'] = merges_needed
            report['average_merges_needed_to_fix_splits'] = average_merges_needed
            report['number_of_unsplittable_fragments'] = \
                len(unsplittable_fragments)
            report['unsplittable_fragments'] = [
                int(f) for f in unsplittable_fragments
            ]
        report['merge_stats'] = merge_stats
        report['split_stats'] = split_stats
        report['threshold'] = threshold
        report['experiment'] = self.experiment
        report['setup'] = self.setup
        report['iteration'] = self.iteration
        report['network_configuration'] = self.edges_db_name
        report['merge_function'] = self.edges_collection.strip('edges_')
        report['config_slab'] = self.config_slab

        scores_collection.replace_one(
            filter={
                'network_configuration': report['network_configuration'],
                'merge_function': report['merge_function'],
                'threshold': report['threshold']
            },
            replacement=report,
            upsert=True)

        find_worst_split_merges(rand_voi_report)


def get_site_fragment_lut(fragments, sites, roi):
    '''Get the fragment IDs of all the sites that are contained in the given
    ROI.'''

    sites = list(sites)

    if len(sites) == 0:
        logger.info("No sites in %s, skipping", roi)
        return None

    logger.info(
        "Getting fragment IDs for %d synaptic sites in %s...",
        len(sites),
        roi)

    # for a few sites, direct lookup is faster than memory copies
    if len(sites) >= 15:

        logger.info("Copying fragments into memory...")
        start = time.time()
        fragments = fragments[roi]
        fragments.materialize()
        logger.info("%.3fs", time.time() - start)

    logger.info("Getting fragment IDs for synaptic sites in %s...", roi)
    start = time.time()

    fragment_ids = np.array([
        fragments[daisy.Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    synaptic_site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)

    logger.info(
        "Got fragment IDs for %d sites in %.3fs",
        len(fragment_ids),
        time.time() - start)

    lut = np.array([synaptic_site_ids, fragment_ids])

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


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate = EvaluateAnnotations(**config)
    evaluate.evaluate()
