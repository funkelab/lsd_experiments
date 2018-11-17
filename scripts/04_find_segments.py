import daisy
import json
import logging
import lsd
import sys
import time
from parallel_read_rag import parallel_read_nodes_edges
from pymongo import MongoClient
from neo4j import GraphDatabase

try:
    import graph_tool
    import graph_tool.topology
    HAVE_GRAPH_TOOL = True
except:
    HAVE_GRAPH_TOOL = False

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def store_fragseg_lut(
        db_host,
        db_name,
        collection_name,
        lut):

    client = MongoClient(db_host)
    database = client[db_name]
    collection = database[collection_name]

    collection.insert_many(lut)

def extract_segmentation(
        db_host,
        db_name,
        edges_collection,
        thresholds,
        roi_offset=None,
        roi_shape=None,
        num_workers=1,
        **kwargs):

    driver = GraphDatabase.driver(
        'bolt://localhost:7687',
        auth=('neo4j', 'lsd'))


    for threshold in thresholds:

        with driver.session() as session:

            print("Getting CCs for threshold %.3f..."%threshold)
            start = time.time()
            res = session.run('''
                CALL algo.unionFind.stream('', 'MERGES', {threshold:%f, concurrency:1})
                YIELD nodeId,setId
                RETURN algo.getNodeById(nodeId).id as id, setId'''%(1.0 - threshold))
            print("%.3fs"%(time.time() - start))

            print("Creating fragment-segment LUT...")
            start = time.time()
            lut = list([
                {
                    'fragment': int(r['id']),
                    'segment': r['setId']
                }
                for r in res
            ])
            print("%.3fs"%(time.time() - start))

        print("Storing fragment-segment LUT...")
        start = time.time()
        store_fragseg_lut(
            db_host,
            db_name,
            'seg_%s_%d'%(edges_collection,int(threshold*100)),
            lut)
        print("%.3fs"%(time.time() - start))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
