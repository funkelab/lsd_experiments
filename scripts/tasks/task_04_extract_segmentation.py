import daisy
import hashlib
import json
import logging
import numpy as np
import os
import pymongo
import sys
import task_helper
import time
from task_empty_task import EmptyTask
from funlib.segment.graphs.impl import connected_components
from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)

class SegmentationTask(daisy.Task):

    experiment = daisy.Parameter()
    setup = daisy.Parameter()
    iteration = daisy.Parameter()
    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    out_dataset = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    edges_collection = daisy.Parameter()
    threshold = daisy.Parameter()
    roi_offset = daisy.Parameter()
    roi_shape = daisy.Parameter()
    num_workers = daisy.Parameter()
    queue = daisy.Parameter()
    no_check = daisy.Parameter(default=0)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading graph from DB %s, collection %s", self.db_name, self.edges_collection)
        start = time.time()

        self.fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset, mode='r')

        self.network_dir = os.path.join(self.experiment, self.setup, str(self.iteration), self.edges_collection)

        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        if 'blocks_segmented' not in db.list_collection_names():
                self.blocks_segmented = db['blocks_segmented']
                self.blocks_segmented.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
        else:
            self.blocks_segmented = db['blocks_segmented']

        graph_provider = daisy.persistence.MongoDbGraphProvider(
            self.db_name,
            self.db_host,
            edges_collection=self.edges_collection,
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'])

        self.total_roi = self.fragments.roi

        self.read_roi = daisy.Roi((0, 0, 0), (2000, 2000, 2000))
        self.write_roi = daisy.Roi((0, 0, 0), (2000, 2000, 2000))

        self.segments = daisy.prepare_ds(
                self.out_file,
                self.out_dataset + "_%.3f" %self.threshold,
                self.fragments.roi,
                self.fragments.voxel_size,
                dtype=np.uint64,
                write_roi=self.write_roi)

        agglom_blocks = 'blocks_agglomerated_hist_quant_50'

        # if agglom_blocks in db.list_collection_names():
            # if db[agglom_blocks].count() >= 1:
        node_attrs, edge_attrs = graph_provider.read_blockwise(
            self.total_roi,
            block_size=self.read_roi.get_end(),
            num_workers=self.num_workers)

        if 'id' not in node_attrs:
            logging.info('No nodes found in roi %s' % self.read_roi)
            return

        logging.info('id dtype: %s', node_attrs['id'].dtype)
        logging.info('edge u  dtype: %s', edge_attrs['u'].dtype)
        logging.info('edge v  dtype: %s', edge_attrs['v'].dtype)

        self.nodes = node_attrs['id']
        self.edges = np.stack([edge_attrs['u'].astype(np.uint64), edge_attrs['v'].astype(np.uint64)], axis=1)
        self.scores = edge_attrs['merge_score'].astype(np.float32)

        logging.info("Complete RAG contains %d nodes, %d edges", len(self.nodes), len(self.edges))

        self.schedule(
            self.total_roi,
            self.read_roi,
            self.write_roi,
            process_function=self.segment_in_block,
            check_function=self.check_block,
            num_workers=self.num_workers,
            read_write_conflict=False,
            fit='shrink')

    def segment_in_block(self, block):

        start = time.time()

        logging.info("Getting CCs for threshold %.3f..." % self.threshold)
        components = connected_components(self.nodes, self.edges, self.scores, self.threshold)

        logging.info("Getting fragments for threshold %.3f..." % self.threshold)
        fragments = self.fragments.to_ndarray(block.write_roi)

        logging.info("Relabelling fragments to %d segments" % len(np.unique(components)))

        relabelled = replace_values(fragments, self.nodes, components)

        self.segments[block.write_roi] = relabelled

        document = {
                'num_cpus': self.num_workers,
                'block_id': block.block_id,
                'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
                'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
                'start': start,
                'duration': time.time() - start
        }

        self.blocks_segmented.insert(document)

    def check_block(self, block):

        done = self.blocks_segmented.count({'block_id': block.block_id}) >= 1

        return done

    def requires(self):
        return [EmptyTask(global_config=self.global_config)]

if __name__ == "__main__":

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
            [
                {'task': SegmentationTask(global_config=global_config,
                **user_configs), 'request': None}
            ],
            global_config=global_config)

