import daisy
import json
import logging
import numpy as np
import os
import sys
from skimage.io import imsave


def extract_slices(
        lsd_file,
        lsd_ds,
        seg_file,
        seg_ds,
        out_dir,
        channel='mean',
        neuron_ids=None):

    print('Converting lsds to np array...')
    lsds = daisy.open_ds(lsd_file, lsd_ds)
    lsds = lsds.to_ndarray()

    if channel == 'mean':
        print('Saving mean offset')
        lsds = lsds[0:3]
    elif channel == 'ortho':
        print('Saving orthogonals')
        lsds = lsds[3:6]
    elif channel == 'diag':
        print('Saving diagonals')
        lsds = lsds[6:9]
    elif channel == 'size':
        print('Saving size')
        lsds = lsds[9:10]

    print(lsds.shape)

    print('Converting seg to np array...')
    seg = daisy.open_ds(seg_file, seg_ds)
    seg = seg.to_ndarray()

    unique = np.unique(seg)

    if type(neuron_ids)==list:
        mask = np.zeros_like(seg, dtype=bool)
        for i in neuron_ids:
            assert i in unique, "Neuron id %i could not be found"%i
            mask[seg==i] = True
    else:
        assert neuron_ids in unique, "Neuron id %i could not be found"%neuron_ids
        mask = seg == neuron_ids

    inv_mask = np.logical_not(mask)

    print('Restricting lsds to neuron id mask...')
    lsds_mask = np.concatenate(lsds.shape[0] * [inv_mask[None]], axis=0)
    lsds[lsds_mask] = 0

    lsds *= 255
    # lsds = lsds.astype('uint8')

    out_path = os.path.join(out_dir,channel)

    if type(neuron_ids)!=list:
        out_path = os.path.join(out_path, str(neuron_ids))

    os.makedirs(out_path, exist_ok=True)

    print('Total slices: ', len(range(lsds.shape[1])))

    for z in range(lsds.shape[1]):
        print('Extracting lsds %s in section %i'\
                %(channel, z))

        name = os.path.join(out_path, '%s_section_%0.03i.tif'\
                %(channel,z))

        imsave(name, lsds[:, z], compress=5)

if __name__ == '__main__':

   #  config_file = sys.argv[1]

    # with open(config_file, 'r') as f:
        # config = json.load(f)

    # extract_slices(**config)

    lsd_file = sys.argv[1]
    seg_file = sys.argv[2]

    lsd_ds = 'volumes/lsds/s3'
    seg_ds = 'volumes/segmentation_30/s3'
    out_dir = sys.argv[3]
    channel = 'ortho'
    neuron_id = [
            77959531,
            2399956,
            78761685,
            143883913,
            79339143,
            169741673,
            120592431,
            152874406,
            161778032,
            114842732,
            79339143,
            113888349,
            76925608,
            118616469,
            84674362,
            86013599,
            2399956,
            157478892,
            86309614]

    extract_slices(
            lsd_file,
            lsd_ds,
            seg_file,
            seg_ds,
            out_dir,
            channel,
            neuron_id)
