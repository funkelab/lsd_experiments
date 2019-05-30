import pymongo
import daisy
import numpy as np
import networkx
import sys
import logging

# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

db_host ='mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke' 
db_name = 'synful'

roi = daisy.Roi(
        (158000, 121800, 403560),
        (76000, 52000, 64000))


def get_skeleton_from_db(component_id):

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='calyx_catmaid.nodes',
            edges_collection='calyx_catmaid.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                'calyx_neuropil_mask': ['masked'],
                'calyx_neuropil_components': ['component_id']
            })

    skeleton = skeletons_provider.get_graph(
            roi=roi,
            nodes_filter={'masked':True, 'neuron_id': component_id}
            )

    return skeleton

if __name__ == '__main__':

    component_id = 8302896

    get_skeleton_from_db(component_id)


