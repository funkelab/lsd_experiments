import daisy
import json
import logging
import lsd
import numpy as np
import os
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)

def dump_graph_in_block(block, rag_provider):

    logger.info("Reading RAG in block %d, ROI %s...", block.block_id, block.read_roi)
    start = time.time()

    rag_read = False
    for i in range(5):
        try:
            rag = rag_provider[block.read_roi]
            nodes = rag_provider.read_nodes(block.read_roi)
            rag_read = True
            break
        except:
            pass

    if not rag_read:
        raise RuntimeError("RAG not read after 5 retries...")

    logger.info("Read RAG in %.3fs", time.time() - start)

    start = time.time()

    with open('calyx_dump/edges_%d.csv'%block.block_id, 'w') as f_edges:
        for e, data in rag.edges.items():
            u, v = min(e[0], e[1]), max(e[0], e[1])
            # neo4j needs "weights" for CC analysis, i.e., 1 - score
            f_edges.write('%d,%f,%d,MERGES\n'%(
                    u, 1.0 - data['merge_score'], v
                )
            )

    with open('calyx_dump/nodes_%d.csv'%block.block_id, 'w') as f_nodes:
        for node in nodes:
            f_nodes.write('%d\n'%node['id'])

    logger.info("Dumped nodes and edges in %.3fs", time.time() - start)

def check_block(block):

    return (
        os.path.isfile('calyx_dump/edges_%d.csv'%block.block_id) and
        os.path.isfile('calyx_dump/nodes_%d.csv'%block.block_id))

def dump(
        db_host,
        db_name,
        edges_collection,
        roi_offset=None,
        roi_shape=None,
        num_workers=1,
        retry=0,
        **kwargs):

    roi = daisy.Roi(offset=roi_offset, shape=roi_shape)
    block_size = (4096, 4096, 4096)
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)

    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name=db_name,
        host=db_host,
        edges_collection=edges_collection,
        mode='r')

    # prepare CSV header files
    with open('calyx_dump/header_nodes.csv', 'w') as f:
        f.write('id:ID\n')
    with open('calyx_dump/header_edges.csv', 'w') as f:
        f.write(':START_ID,score,:END_ID,:TYPE\n')

    logger.info("Dumping node and edge list...")
    start = time.time()
    for i in range(retry + 1):
        if daisy.run_blockwise(
            roi,
            read_roi,
            write_roi,
            lambda b: dump_graph_in_block(
                b,
                rag_provider),
            check_block,
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logger.error("parallel read failed, retrying %d/%d", i + 1, retry)

    logger.info("Dumped node and edge list in %.3fs", time.time() - start)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    dump(**config)
