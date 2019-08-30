LSD Experiments
===============

Experiments
-----------

* cremi
* fib19
* fib25
* hemi
* zebrafinch
* L1
* vnc1 (WIP)
* zebrafish (WIP)

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

All scripts should be runnable as they are using singularity image
`funkey/lsd:v0.8`.

Post-processing
---------------

The post processing pipeline (inference, watershed, agglomeration, segmentation)
is split between scripts `scripts/{order}_{step}.py`. Every post processing script 
takes a config json as input. The required parameters for the jsons can be found in 
the main functions called in each script. A MongoDB backend is currently used for 
several steps in the pipeline. For inference, watershed, and agglomeration, we have
block check functions which log meta data for runs (i.e queue, number of workers,
start, end, block roi, etc) and also enable scripts to pick up where they left off
if cancelled or some kind of failure occurs. For watershed and agglomeration, we
also store nodes and edges, respectively.

Inference
---------

Given that each setup in each experiment has a `test_net.meta`, `test_net.json` 
(as produced by `mknet.py`), a `train_net_checkpoint_<iteration>`, and a `predict.py`,
the prediction process is the same for all of them.

Block-wise prediction can be performed using `scripts/01_predict_blockwise.py`.
See the comments inside this file for usage instructions.

This script will make sure that predictions are performed in a chunk-aligned
manner: A chunk corresponds both to the output size of the network and the size
of a chunk in the N5 container to produce. Parallel prediction inside
`predict.py` are only saved if no two processes attempt to write to the same N5/Zarr
chunk. `01_predict_blockwise.py` will read the chunk size from the network
configuration, create the output N5/Zarr container, and run prediction in blocks
that are multiples of the chunk size. This does currently put restrictions on
the area that can be predicted in a block-wise manner: Output volumes are
multiples of the block-size (which in turn are multiples of the chunk size).

Watershed
---------

`scripts/02_extract_fragments_blockwise.py` performs block-wise watershed. This
creates an initial supervoxel "oversegmentation" to be later agglomerated 
hierarchically. This script extracts _fragments_ in parallel by performing
watershed on predicted affinities. These fragments are stored as a label volume
inside the N5/Zarr dataset holding the affinities and as nodes in the database.
An optional _epsilon agglomeration_ can be performed to merge fragments into
larger supervoxels within block boundaries. This can be useful when dealing with 
large datasets in order to decrease the number of nodes in the database. The chunk
restrictions mentioned earlier do not apply anymore; block sizes can freely be chosen
but should be a multiple of the volume resolution

Agglomeration
-------------

`scripts/03_agglomerate_blockwise.py` performs block-wise agglomeration. The
fragments are agglomerated in parallel using [waterz](https://github.com/funkey/waterz).
This does not produce a segmentation, instead, edges are stored in the database 
with a `merge_score` attribute that correspond to the threshold under which 
each edge got merged. This results in a _region adjacency graph_ (RAG). It is good
practice to maintain a consistent block size with watershed.

Segmentation
------------

Segmentations are implicitly stored in the candidate DB. To get a segmentation,
`scripts/04_find_segments.py` and `scripts/04_extract_segmentation_from_lut` can be
used (these will likely be merged into one script upon release). These scripts use
several functions from [funlib.segment](https://github.com/funkelab/funlib.segment).
The first script reads the RAG and relabels connected components between nodes and 
edges given the underlying merge score and a threshold. The resulting node-component
lookup table (LUT) is saved in a compressed numpy array (.npz) inside a 
`luts/fragment_segment` folder next to `volumes` in the output N5/Zarr container. 
In order to extract a segmentation at a given threshold, the second script loads
the respective LUT and the fragments dataset and relabels the fragments with unique
segments, block-wise.
