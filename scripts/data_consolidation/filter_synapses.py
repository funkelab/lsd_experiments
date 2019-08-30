import json
import pandas as pd
import sys

def filter_synapses(
        in_file,
        roi_offset,
        roi_shape,
        confidence,
        out_file):

    print('Loading %s' %in_file)
    synapses = pd.read_csv(in_file)

    roi_end = [i+j for i,j in zip(roi_offset, roi_shape)]

    print('Filtering synapses...')
    filtered_z = synapses[(synapses['z'] >= roi_offset[0]/8) & (synapses['z'] <= roi_end[0]/8)]
    filtered_y = filtered_z[(filtered_z['y'] >= roi_offset[1]/8) & (filtered_z['y'] <= roi_end[1]/8)]
    filtered_x = filtered_y[(filtered_y['x'] >= roi_offset[2]/8) & (filtered_y['x'] <= roi_end[2]/8)]

    filtered_synapses = filtered_x[(filtered_x['conf'] >= confidence)]

    print(filtered_synapses)

    print('Writing to %s' %out_file)
    filtered_synapses.to_csv(out_file, index=False, header=True)

if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    filter_synapses(**config) 
