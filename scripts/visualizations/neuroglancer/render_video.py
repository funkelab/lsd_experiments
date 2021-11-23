import daisy
import json
import neuroglancer
import os
import sys

from funlib.show.neuroglancer import add_layer, ScalePyramid, RenderArgs
from funlib.show.neuroglancer.video_tool import run_render

neuroglancer.set_server_bind_address('0.0.0.0')

f = sys.argv[1]

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

def create_viewer():
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_layer(s, raw, 'raw')
    return viewer

if __name__ == '__main__':

    render_params = RenderArgs()

    out_dir = os.path.join(os.getcwd(), 'test_video')

    os.makedirs(out_dir, exist_ok=True)

    render_params.script = sys.argv[2]
    render_params.output_directory = out_dir
    render_params.width=7680
    render_params.height=2160

    run_render(create_viewer, args=render_params)


