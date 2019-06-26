import daisy
import json
import logging
import numpy as np
import os
import sys
from skimage.io import imsave

logging.basicConfig(level=logging.INFO)

def open_ds(path, ds_name):
    try:
        logging.info('Loading dataset %s, converting to nd array...', ds_name)
        return daisy.open_ds(path, ds_name).to_ndarray()
    except KeyError:
        logging.info('Dataset %s was not loaded, ensure path is correct', ds_name)
        return None

def convert_labels_to_rgb(labels):

    max_label = np.max(labels)

    try:
        lookup = np.random.randint(low=0, high=255, size=(int(max_label+1),3), dtype=np.uint8)
    except MemoryError:
        logging.info('Data labels too large, try relabeling connected components first')
        return None

    lookup = np.append(lookup, np.zeros((int(max_label+1),1), dtype=np.uint8) + 255, axis=1)

    lookup[0] = 0

    return lookup[labels]

def create_out_dir(
        in_dataset,
        cube=False,
        neuron_id=None,
        lsds='Mean'):

    base_dir = '.'
    if cube:
        out_dir = os.path.join(base_dir, 'Cube')
    if neuron_id:
        out_dir = os.path.join(base_dir, 'Neuron_%s'%neuron_id)

    remove_volumes = in_dataset.strip('/volumes')

    if '/s0' in remove_volumes:
        remove_volumes = remove_volumes.strip('s0')

def save_image(name, array):
    return imsave(name, array, compress=5)

def extract_cube_sides(ds):

    if ds.dtype == 'uint64':

        logging.info('Data type is uint64, converting labels to rgb first...')
        ds = convert_labels_to_rgb(ds)

        neg_x = ds[:,:,-1]
        pos_x = ds[:,:,1]

        neg_y = ds[:,-1]
        pos_y = ds[:,1]

        neg_z = ds[-1]
        pos_z = ds[1]

    else:

        neg_x = ds[:,:,:,-1]
        pos_x = ds[:,:,:,1]

        neg_y = ds[:,:,-1]
        pos_y = ds[:,:,1]

        neg_z = ds[:,-1]
        pos_z = ds[:,1]

    save_image(name + '-x'), neg_x)
    save_image(name + 'x'), pos_x)

    save_image(name + '-y'), neg_y)
    save_image(name + 'y'), pos_y)

    save_image(name + '-z'), neg_z)
    save_image(name + 'z'), pos_z)


def extract_all_slices():
    pass


def extract_single_neuron():
    pass


def main(
    in_file,
    in_dataset,
    out_path,
    extract_cube_sides=False,
    extract_single_neuron=False):

    

 __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    main(**config)

