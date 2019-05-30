from __future__ import print_function

import neuroglancer
from funlib.show.neuroglancer import add_layer
import sys
import daisy

neuroglancer.set_server_bind_address('0.0.0.0')

f = sys.argv[1]
frags = daisy.open_ds(f, 'volumes/fragments')

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, frags, 'frags')

def my_action(s):
    print('Howdy Partner')
    print('  Mouse position: %s' % (s.mouse_voxel_coordinates,))
    print('  Layer selected values: %s' % (s.selected_values,))
viewer.actions.add('my-action', my_action)
with viewer.config_state.txn() as s:
    s.input_event_bindings.viewer['keyt'] = 'my-action'

print(viewer)
