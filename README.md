LSD Experiments
===============

Experiments
-----------

* cremi
* fib19
* fib25

Each experiment has its own subdirectories.

Data
----

Raw data for each experiment is located in

```
<experiment>/01_data/<sample>
```

Where sample is a relative path to an HDF5, N5, or zarr container.

Training
--------

Training scripts are found in

```
<experiment>/02_train/<setup>
```

where `<setup>` is the name of a particular network and loss configuration.
Each setup directory contains a `mknet.py` to create the training and testing
networks, and a `train.py` to train to a given number of iterations. This
directory also contains `predict.py` to apply the trained network with a larger
input size to arbitrary samples.

All scripts should be runnable as they are using docker image
`funkey/lsd:v0.2`.

Prediction
----------

Given that each setup in each experiment has a `test_net.meta`,
`test_net_config.json` (as produced by `mknet.py`), a
`train_net_checkpoint_<iteration>`, and a `predict.py`, the prediction process
is the same for all of them.

Block-wise prediction can be performed using `scripts/predict_blockwise.py`.
See the comments inside this file for usage instructions.

This script will make sure that predictions are performed in a chunk-aligned
manner: A chunk corresponds both to the output size of the network and the size
of a chunk in the N5 container to produce. Parallel prediction inside
`predict.py` are only save if no two processes attempt to write to the same N5
chunk. `predict_blockwise.py` will read the chunk size from the network
configuration, create the output N5 container, and run prediction in blocks
that are multiples of the chunk size. This does currently put restrictions on
the area that can be predicted in a block-wise manner: Output volumes are
multiples of the block-size (which in turn are multiples of the chunk size).

The output dataset of `scripts/predict_blockwise.py` will be stored in

```
<experiment>/03_predict/<setup>/<iteration>/<sample>
```

Agglomeration
-------------

Similar to `scripts/predict_blockwise.py`, `scripts/agglomerate_blockwise.py`
performs block-wise agglomeration. See the script for usage instructions.
Agglomeration happens in two steps: First, _fragments_ are extracted in
parallel by performing watershed on predicted affinities. These fragments are
stored as a label volume along the N5 dataset holding the affinities and as
nodes in a candidate DB. Second, the fragments are agglomerated in parallel
using [waterz](https://github.com/funkey/waterz). This does not produce a
segmentation, instead, edges are stored in the candidate DB with a
`merge_score` attribute that correspond to the threshold under which each edge
got merged.

Affinities are assumed to be found in container
`<experiment>/03_predict/<setup>/<iteration>/<sample>` in dataset
`volumes/affs`.

Fragments will be stored in the same container in dataset `volumes/fragments`.

The candidate DB will (for now) be stored in
`<experiment>/03_predict/<setup>/<iteration>/<sample>.db`. This is an SQLite
DB, which is only intended for testing and won't scale to large volumes and
processing on several nodes on the cluster. A proper DB backend is about to
come soon...

The chunk restrictions mentioned earlier do not apply anymore. Block sizes can
freely be chosen, and will correspond to the chunk sizes in the fragments
dataset.

Segmentation
------------

Segmentations are implicitly stored in the candidate DB. To get a segmentation,
`scripts/extract_segmentations.py` can be used. This script will threshold
edges based on their `merge_score`, extract connected components, and create a
corresponding segmentation by merging fragments accordingly. This script does
not parallelize, yet. The resulting segmentation has to fit into memory.
