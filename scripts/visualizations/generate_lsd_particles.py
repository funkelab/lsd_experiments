import csv
import daisy
import numpy as np
import sys
import random
import time
import logging

def test_function(ds, sites, vs):

    offset = ds.roi.get_begin()
    ds = ds.to_ndarray()

    t = np.array([
            ds[:,
                int((site[0]-offset[0])/vs[0])-1,
                int((site[1]-offset[1])/vs[1])-1,
                int((site[2]-offset[2])/vs[2])-1
            ]
            for site in sites])

    return t

if __name__ == '__main__':

    lsds = daisy.open_ds(sys.argv[1], 'volumes/lsds/s3')
    seg = daisy.open_ds(sys.argv[2], 'volumes/segmentation_40/s3')

    roi = seg.roi
    voxel_size = seg.voxel_size

    lsds = lsds[roi]

    print('converting to numpy array...')
    lsds_np = lsds.to_ndarray()
    seg_np = seg.to_ndarray()

    i = 368275

    mask = seg_np == i

    inv_mask = np.logical_not(mask)

    lsds_mask = np.concatenate(10 * [inv_mask[None]],axis=0)

    lsds_np[lsds_mask] = 0

    print('creating daisy array...')

    lsds_daisy = daisy.Array(lsds_np, roi, voxel_size)

    points = []

    start = list(roi.get_begin())
    end = list(roi.get_end())

    count = 10000000

    start_time = time.time()

    for p in range(count):
        points.append([random.randint(i,j) for i,j in zip(start,end)])

    end_time = time.time()

    seconds = end_time - start_time
    print('Total time to append: %f seconds' % seconds)

    lsds_points = test_function(lsds_daisy,points,voxel_size)

    sites = []

    for i,j in zip(range(lsds_points.shape[0]), points):
        if np.sum(lsds_points[i]) != 0:
            sites.append([
                j[0]*0.0002,
                j[1]*0.0002,
                j[2]*0.0002,
                int(lsds_points[i][0]),
                int(lsds_points[i][1]),
                int(lsds_points[i][2]),
                int(lsds_points[i][3]),
                int(lsds_points[i][4]),
                int(lsds_points[i][5]),
                int(lsds_points[i][6]),
                int(lsds_points[i][7]),
                int(lsds_points[i][8]),
                int(lsds_points[i][9])
                ]
            )

    print(len(sites))

    with open("hemi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sites)
