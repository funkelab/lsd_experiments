import pymongo

def get_block_end(offset, shape):

    return [i+j for i,j in zip(offset, shape)]

def check_loc(start, end, loc):

    return all([i <= j <= k for i,j,k in zip(start, loc, end)])

def convert_vox_to_world(coord, voxel_size):

    return [i*j for i,j in zip(coord, voxel_size)][::-1]

if __name__ == '__main__':

    db_host =  "mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd"
    db_name = 'zebrafinch_auto_basic_163k_testing_masked'

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_extracted']

    voxel_size = [9,9,20]
    point = convert_vox_to_world([5350, 5450, 2850], voxel_size)
    print(point)

    for doc in collection.find():

        offset = doc['write_roi'][0]
        shape = doc['write_roi'][1]

        end = get_block_end(offset, shape)

        # print(offset, point, end, doc['block_id'])

        if check_loc(offset, end, point):
            print('block id: %s, read offset: %s, read shape: %s, write offset: %s, write shape: %s'%(
                      doc['block_id'],
                      doc['read_roi'][0],
                      doc['read_roi'][1],
                      doc['write_roi'][0],
                      doc['write_roi'][1]))
