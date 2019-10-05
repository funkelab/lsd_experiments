import csv
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

def append_csv(path):

    l = []

    with open(path, mode='r') as f:
        reader = csv.reader(f)
        [l.append(i) for row in reader for i in row]

    return l

def convert_to_int(l):

    return [int(el) if not isinstance(el, l) else convert_to_int(el) for el in l]

def find_loc(string, start, end):

    return int(string[string.find(start) + len(start): string.rfind(end)])

def parse_locations(l):

    #this is a super specific lazy parsing for scott's urls lol

    locations = []

    xp = 'xp='
    yp = '&yp='
    zp = '&zp='
    end = '&tool'

    for row in l:
        x = find_loc(row, xp, yp)
        y = find_loc(row, yp, zp)
        z = find_loc(row, zp, end)

        locations.append([x, y, z])

    return locations

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

    centers = parse_locations(append_csv(sys.argv[1]))

    # enter voxel size and size of volume in xyz nm

    voxel_size = [4, 4, 40]

    size = [5000, 5000, 5000]

    for center in centers[0:5]:
        offset = ensure_multiple(
                    center_roi(
                        center, size),
                    voxel_size)

        print('Center: %s, Offset: %s, Size: %s'%(center[::-1], offset[::-1], size))
        print('\n')

    # create_json(sys.argv[1], sys.argv[2], offset, size)


