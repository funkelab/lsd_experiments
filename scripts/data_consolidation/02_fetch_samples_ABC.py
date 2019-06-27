import h5py
import zarr
import numpy as np
import operator

data_dir = '/groups/funke/cremi/02_run/cremi_glial_masks'

# x,y: 1.5 times the context of setup04
#   z: a little more than context of setup04
PADDING = [ 12, 160, 160 ]

def offset_bb(gt, offset):

    orig_shape = gt.shape
    print("original shape: " + str(orig_shape))

    offset_bb = tuple(
            slice(
                int(offset[d]) - int(PADDING[d]), int(offset[d]) + int(orig_shape[d]) + int(PADDING[d])) for d in range(3)
                )


    print("offset bb: " + str(offset_bb))

    return offset_bb

def normal_bb(gt):

    orig_shape = gt.shape

    fg_indices = np.where(gt < np.uint64(-10))

    normal_bb = tuple(
            slice(
                max(0,             np.min(fg_indices[d]) - PADDING[d]),
                min(orig_shape[d], np.max(fg_indices[d]) + PADDING[d] + 1)
                )
            for d in range(3)
            )

    print("normal bb: " + str(normal_bb))

    return normal_bb

if __name__ == "__main__":

    for sample in [
            'sampleA',
            'sampleB',
            'sampleC'
            ]:

        print("Fetching %s"%sample)

        if sample == 'sampleA':
            f_in = h5py.File(data_dir + '/' + sample + '.merged.h5', mode='r')
        if sample == 'sampleB':
            f_in = h5py.File(data_dir + '/' + sample + '.merged.h5', mode='r')
        if sample == 'sampleC':
            f_in = h5py.File(data_dir + '/' + sample + '.merged.h5', mode='r')
        f_out = zarr.open(sample + '.n5', mode='w')

        print("Reading neuron IDs...")
        neuron_ids = f_in['/volumes/labels/merged_ids'][:]
        ids_offset = f_in['volumes/labels/merged_ids'].attrs['offset']

        resolution = f_in['volumes/labels/merged_ids'].attrs['resolution']

        offset = list(map(operator.truediv, ids_offset, resolution))
        offset = np.array(offset, dtype=np.int)
        new_offset = list(map(operator.sub, offset, PADDING))

        new_offset_world = list(map(operator.mul, new_offset, resolution))

        print("Finding BB...")
        offsetbb = offset_bb(neuron_ids, offset)
        normalbb = normal_bb(neuron_ids)

        print("Reading raw...")
        raw = f_in['/volumes/raw'][:]
        raw_shape = f_in['/volumes/raw'].shape
        raw_offset = f_in['/volumes/raw'].attrs['offset']

        new_raw_offset = list(map(operator.sub, raw_offset, PADDING))
        new_raw_offset_world = list(map(operator.mul, new_raw_offset, resolution))

        background_id = {
            'sampleA': 0,
            'sampleB': 864427,
            'sampleC': 148058
        }[sample]

        print('Changing background ids to (-3)...')
        neuron_ids = np.where(neuron_ids==background_id, np.uint64(-3), neuron_ids)

        print("Increasing background labels to size of raw...")
        new_neuron_ids = np.zeros_like(raw, dtype=np.uint64)
        new_neuron_ids[:] = np.uint64(-3)
        new_neuron_ids[offset[0]:offset[0]+neuron_ids.shape[0], offset[1]:offset[1]+neuron_ids.shape[1],offset[2]:offset[2]+neuron_ids.shape[2]] = neuron_ids

        print("Cropping...")
        raw = raw[offsetbb]
        new_neuron_ids = new_neuron_ids[offsetbb]

        missing_sections = [26, 86]
        if sample == 'sampleC':
            print('Replacing missing sections with (-2)...')
            for i in missing_sections:
                new_neuron_ids[i] = np.uint64(-2)

        print("Creating GT mask...")
        mask = (new_neuron_ids < np.uint64(-3)).astype(np.uint8)

        print("Creating glia mask...")
        glia_id = {
            'sampleA': 300118,
            'sampleB': 865226,
            'sampleC': 152161
        }[sample]
        glia = (new_neuron_ids == glia_id)

        print("Creating no-glia neuron IDs...")
        neuron_ids_noglia = np.array(new_neuron_ids)
        neuron_ids_noglia[glia] = 0

        glia = glia.astype(np.uint8)

        print("Creating no-neuron glia ID...")
        glia_noneurons = np.array(new_neuron_ids)
        glia_noneurons[np.logical_or(np.logical_not(glia), np.logical_not(mask))] = 0

        for ds_name, data in [
                ('/volumes/raw', raw),
                ('/volumes/labels/neuron_ids', new_neuron_ids),
                ('/volumes/labels/neuron_ids_noglia', neuron_ids_noglia),
                ('/volumes/labels/mask', mask),
                ('/volumes/labels/glia', glia),
                ('/volumes/labels/glia_noneurons', glia_noneurons)
                ]:

            print("Writing %s..."%ds_name)

            ds_out = f_out.create_dataset(
                ds_name,
                data=data,
                compressor=zarr.get_codec({'id': 'gzip', 'level': 5}))

            ds_out.attrs['offset'] = [0, 0, 0]
            ds_out.attrs['resolution'] = [4, 4, 40]


