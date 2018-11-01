import multiprocessing
import dask
import daisy
import lsd
import os
import logging
import numpy as np
from pymongo import MongoClient

logging.basicConfig(level=logging.DEBUG)

def read_graph_in_block(block, rag_provider, shared_node_list, shared_edge_list):
    print('connecting...')
    rag_provider._MongoDbRagProvider__connect()
    rag_provider._MongoDbRagProvider__open_db()
    rag_provider._MongoDbRagProvider__open_collections()
    nodes = rag_provider._MongoDbRagProvider__read_nodes(block.read_roi)
    node_list = [
            (n['id'], rag_provider._MongoDbRagProvider__remove_keys(n, ['id']))
            for n in nodes
            ]
    shared_node_list += node_list
    
    node_ids = list([node[0] for node in node_list])
    edges = rag_provider.edges.find(
            {
                'u': { '$in': node_ids }
            })

    edge_list = [
        (e['u'], e['v'], rag_provider._MongoDbRagProvider__remove_keys(e, ['u', 'v']))
        for e in edges
        ]
    shared_edge_list += edge_list
    del nodes
    del edges
    print('done connecting')
    rag_provider._MongoDbRagProvider__disconnect()
    print("nodes: {0}".format(len(node_list)))
    print("edges: {0}".format(len(edge_list)))

def parallel_read_rag(
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        edges_collection,
        block_size,
        num_workers,
        retry):

    fragments = daisy.open_ds(fragments_file, fragments_dataset)
    total_roi = fragments.roi.copy()
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    rag_provider = lsd.persistence.MongoDbRagProvider(db_name=db_name, host=db_host, mode='r', edges_collection=edges_collection)
    node_list = multiprocessing.Manager().list()
    edge_list = multiprocessing.Manager().list()

    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: read_graph_in_block(
                b,
                rag_provider,
                node_list,
                edge_list),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel read failed, retrying %d/%d", i + 1, retry)

    graph = lsd.persistence.mongodb_rag_provider.MongoDbSubRag(db_name, db_host, 'r')
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list)
    return graph
