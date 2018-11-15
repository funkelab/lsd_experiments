import dask
import dask.multiprocessing
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import daisy
import logging
import json
import sys

logging.basicConfig(level=logging.DEBUG)

def stats_in_block(
        vol,
        block,
        min_blocked,
        max_blocked,
        sum_blocked,
        hist_blocked,
        dtype):
    logging.info("Calculating statistics in {0}".format(block))
    data = vol[block.read_roi].to_ndarray()
    block_min = np.min(data)
    block_max = np.max(data)
    block_sum = np.sum(data)
    # block_hist is a tuple containing (histogram, bin edges)
    block_hist = np.histogram(data, bins=256, range=(0.0, 255.0)) if dtype == 'uint8' else np.histogram(data, bins=256, range=(0.0, 1.0))
    logging.debug("\nResult in block {0}\n min: {1} max: {2} sum: {3}".format(block, block_min, block_max, block_sum))
    min_blocked.append(block_min)
    max_blocked.append(block_max)
    sum_blocked.append(block_sum)
    hist_blocked.append(block_hist)

def combine_histograms(hist_blocked):
    # assuming all histograms have same edges as first histogram
    bin_edges = hist_blocked[0][1]
    bin_counts = np.zeros(bin_edges.shape[0])
    for i in range(len(hist_blocked)):
        # block_hist is a tuple containing (histogram, bin edges)
        block_hist = hist_blocked[i]
        bin_counts[0:block_hist[0].shape[0]] += block_hist[0]
    return (bin_counts, bin_edges)

def plot_histogram(bin_counts, bin_edges, dtype, plot_file):
    fig, ax = plt.subplots()
    width = 1.0 if dtype == 'uint8' else 1.0/256
    ax.bar(bin_edges, bin_counts, width=width, edgecolor='k', align='edge')
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)

def parallel_stats(
    """
    Calculates statistics (min, max, sum, mean) and plots a histogram of a
    given volume.

    Args:

        in_file (``string``):

            The input h5-like file containing a volume for which to calculate
            statistics.

        in_dataset (``string``):

            Name of dataset containing volume.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        num_workers (``int``):

            How many blocks to run in parallel.

        log_file (``string``):

            Name of log file in which results will be stored (should be
            prefix without extension).

        materialize (``bool``):

            Whether or not to load entire dataset into memory.

        retry (``int``):

            Number of repeat attempts if any tasks fail in first run.
    """
        in_file,
        in_dataset,
        block_size,
        num_workers,
        log_file='log',
        materialize=False,
        retry=2):
    vol = daisy.open_ds(in_file, in_dataset, mode='r')
    if materialize:
        vol.materialize()
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)

    m = mp.Manager()
    min_blocked = m.list()
    max_blocked = m.list()
    sum_blocked = m.list()
    hist_blocked = m.list()
    

    logging.info("Calculating statistics")

    for i in range(retry + 1):
        if daisy.run_blockwise(
            vol.roi,
            read_roi,
            write_roi,
            lambda b: stats_in_block(
                vol,
                b,
                min_blocked,
                max_blocked,
                sum_blocked,
                hist_blocked,
                vol.data.dtype),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel relabel failed, retrying %d/%d", i + 1, retry)

    logging.debug("Consolidating blockwise information")
    
    vol_min = np.float64(np.min(min_blocked)) 
    vol_max = np.float64(np.max(max_blocked))
    vol_sum = np.float64(np.sum(sum_blocked))
    vol_mean = vol_sum / np.prod(vol.shape)
    vol_hist = combine_histograms(hist_blocked)

    logging.info("\nFinal result\n min: {0} max: {1} sum: {2} mean: {3}".format(vol_min, vol_max, vol_sum, vol_mean))
    logging.info("Writing results to {0}.json and {0}.png".format(log_file))
    results = {}
    results['in_file'] = in_file
    results['in_dataset'] = in_dataset
    results['min'] = vol_min
    results['max'] = vol_max
    results['sum'] = vol_sum
    results['mean'] = vol_mean
    with open(log_file+".json", 'w') as f:
        json.dump(results, f)
    plot_histogram(*vol_hist, vol.data.dtype, log_file+".png")

if __name__=='__main__':
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    parallel_stats(**config)
