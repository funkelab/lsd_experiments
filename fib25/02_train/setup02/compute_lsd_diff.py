from __future__ import print_function
from gunpowder import *
from lsd.gp import AddLocalShapeDescriptor
from funlib.segment.graphs import find_connected_components
from funlib.segment.arrays import replace_values
import json
import logging
import sys
import pymongo
import daisy
import numpy as np

class GetSegmentation(BatchFilter):

    def __init__(
            self,
            fragments,
            db_host,
            rag_db,
            edges_collection,
            threshold,
            segmentation):

        self.fragments = fragments
        self.db_host = db_host
        self.rag_db = rag_db
        self.edges_collection = edges_collection
        self.threshold = threshold
        self.segmentation = segmentation

    def setup(self):

        self.provides(
            self.segmentation,
            self.spec[self.fragments].copy())

        # open RAG DB
        self.rag_provider = daisy.persistence.MongoDbGraphProvider(
            self.rag_db,
            host=self.db_host,
            mode='r',
            position_attribute=['center_%s' % d for d in ['z', 'y', 'x']],
            edges_collection=self.edges_collection)

    def prepare(self, request):
        request[self.fragments] = request[self.segmentation].copy()

    def process(self, batch, request):

        fragments = batch[self.fragments]
        roi = fragments.spec.roi
        fragments_data = fragments.data

        segmentation_data = self.__get_segmentation(fragments_data, roi)

        segmentation = Array(
            segmentation_data,
            fragments.spec.copy())

        batch[self.segmentation] = segmentation

    def __get_segmentation(self, fragments, roi):

        segmentation = np.array(fragments)
        rag = self.rag_provider[roi]

        logger.info("Number of nodes in RAG: %d",len(rag.nodes()))
        logger.info("Number of edges in RAG: %d",len(rag.edges()))

        if len(rag.nodes()) == 0:
            logger.warning("No nodes found")
            return segmentation

        fragment_segment_lut = find_connected_components(
            rag,
            edge_score_attribute='merge_score',
            edge_score_relation='<=',
            edge_score_threshold=self.threshold)

        values_map = np.array([
            [fragment, fragment_segment_lut[fragment]]
            for fragment in rag.nodes()
        ], dtype=np.uint64)
        old_values = values_map[:,0]
        new_values = values_map[:,1]
        replace_values(segmentation, old_values, new_values, inplace=True)

        return segmentation

class ComputeDistance(BatchFilter):

    def __init__(self, a, b, diff):

        self.a = a
        self.b = b
        self.diff = diff

    def setup(self):

        self.provides(
            self.diff,
            self.spec[self.a].copy())

    def prepare(self, request):

        request[self.a] = request[self.diff].copy()
        request[self.b] = request[self.diff].copy()

    def process(self, batch, request):

        a_data = batch[self.a].data
        b_data = batch[self.b].data

        diff_data = np.sum((a_data - b_data)**2, axis=0)

        sum_a = np.sum(a_data[0:3,:,:,:], axis=0)
        nonzeros_a = sum_a != 0
        diff_data *= nonzeros_a

        batch[self.diff] = Array(
            diff_data,
            batch[self.a].spec.copy())

def compute_lsd_diff(
        lsds_file,
        lsds_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        edges_db_name,
        edges_collection,
        threshold,
        lsd_diffs_file,
        lsd_diffs_dataset,
        seg_dataset,
        **kwargs):

    fragments = ArrayKey('FRAGMENTS')
    segmentation = ArrayKey('SEGMENTATION')
    seg_lsds = ArrayKey('SEG_LSDS')
    pred_lsds = ArrayKey('PRED_LSDS')
    lsd_diff = ArrayKey('LSD_DIFF')

    chunk_request = BatchRequest()
    chunk_request[lsd_diff] = ArraySpec(
        roi=Roi((0, 0, 0), (1, 1, 1)))
    chunk_request[segmentation] = ArraySpec(
        roi=Roi((0, 0, 0), (1, 1, 1)))

    voxel_size = Coordinate((8,)*3)

    sources = (
        ZarrSource(
                fragments_file,
                datasets = {
                    fragments: fragments_dataset
                },
                array_specs = {
                    fragments: ArraySpec(voxel_size=voxel_size)
                }
            ),
        ZarrSource(
                lsds_file,
                datasets = {
                    pred_lsds: lsds_dataset
                },
                array_specs = {
                    pred_lsds: ArraySpec(voxel_size=voxel_size)
                }
            ) +
        Normalize(pred_lsds)
    )

    pipeline = sources + MergeProvider()

    pipeline += Pad(fragments, size=None)
    pipeline += Pad(pred_lsds, size=None)

    pipeline += GetSegmentation(
        fragments,
        db_host,
        edges_db_name,
        edges_collection,
        threshold,
        segmentation)

    pipeline += AddLocalShapeDescriptor(
        segmentation,
        seg_lsds,
        sigma=80,
        downsample=2)

    pipeline += ComputeDistance(
        seg_lsds,
        pred_lsds,
        lsd_diff)

    pipeline += ZarrWrite(
            dataset_names={
                segmentation: seg_dataset,
                lsd_diff: lsd_diffs_dataset
            },
            output_filename=lsd_diffs_file
        )
    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyRequestBlocks(
            chunk_request,
            roi_map={
                segmentation: 'write_roi',
                lsd_diff: 'write_roi'
            },
            num_workers=1)

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    compute_lsd_diff(**run_config)
