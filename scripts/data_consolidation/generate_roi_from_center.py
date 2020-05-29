import json
import sys
import numpy as np

def get_center(start, end):

    return [int(round(((j-i)/2)+i)) for i,j in zip(start, end)]

def ng_to_world(coords, voxel_size):

    coords = [round(i)*j for i,j in zip(coords, voxel_size)]

    return np.flip(coords).astype(np.float32)

def center_roi(offset, size):

    offset = [int(i) for i in offset]

    return [i-j for i,j in zip(offset, [i/2 for i in size])]

def ensure_multiple(offset, voxel_size):
    return [j * round(i/j) for i,j in zip(offset, voxel_size[::-1])]

def create_json(
        config_file,
        container,
        offset,
        size):

    config = {
            'container': container,
            'offset': offset,
            'size': size
            }

    with open(config_file, 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':

    center = [87163,37384,4070]

    # print('Center: ', center)

    voxel_size = [4, 4, 40]

    size = [5000, 5000, 5000]

    offset = ensure_multiple(
                    center_roi(
                        ng_to_world(
                            center, voxel_size),
                        size),
                    voxel_size)

    print('Offset: %s, Size: %s'%(offset, size))
    print('\n')

     # create_json(out_file, sys.argv[1], offset, size)


