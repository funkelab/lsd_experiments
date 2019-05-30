import daisy
import pymongo
import numpy as np

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'mtlsd_6sample_noglia_autocontext'

if __name__ == "__main__":

    client = pymongo.MongoClient(db_host)
    fragment_db = client[db_name]
    nodes = fragment_db['nodes']

    scores_db = client['test_calyx_scores']
    scores = scores_db['scores']

    score = scores.find({'network_configuration': db_name})[0]

    unsplittable_fragments = score['unsplittable_fragments']

    fragments = list(nodes.find({'id': {'$in': unsplittable_fragments }}))
    print("Number of unsplittable fragments:", len(fragments))

    print("Centers of these fragments in CATMAID coordinates:")
    positions = [
        [
            int(f['center_x'])/4,
            int(f['center_y'])/4,
            int(f['center_z'])/40
        ]
        for f in fragments
    ]
    print(positions)
