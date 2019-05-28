import json
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time
import pymongo
import hashlib

import task_helper
from task_01_predict_blockwise import PredictTask

logging.basicConfig(level=logging.INFO)
logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)

class ExtractFragmentsTask(daisy.Task):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        affs_file,
        affs_dataset,

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    # pass

    experiment = daisy.Parameter()
    setup = daisy.Parameter()
    iteration = daisy.Parameter()
    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    num_workers = daisy.Parameter()
    queue = daisy.Parameter()

    mask_file = daisy.Parameter(default=None)
    mask_dataset = daisy.Parameter(default=None)

    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    fragments_in_xy = daisy.Parameter(default=False)

    epsilon_agglomerate = daisy.Parameter(default=0)

    def prepare(self):
        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        self.network_dir = os.path.join(self.experiment, self.setup, str(self.iteration))

        # prepare fragments dataset
        self.fragments = daisy.prepare_ds(
            self.fragments_file,
            self.fragments_dataset,
            self.affs.roi,
            self.affs.voxel_size,
            np.uint64,
            daisy.Roi((0, 0, 0), self.block_size),
            compressor={'id': 'zlib', 'level':5}
            )


        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        if 'blocks_extracted' not in db.list_collection_names():
                self.blocks_extracted = db['blocks_extracted']
                self.blocks_extracted.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
        else:
            self.blocks_extracted = db['blocks_extracted']

        context = daisy.Coordinate(self.context)
        total_roi = self.affs.roi.grow(context, context)
        read_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size).grow(context, context)
        write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

        logging.info("Scheduling...")

        self.schedule(
            total_roi,
            read_roi,
            write_roi,
            process_function=self.start_worker,
            check_function=self.check_block,
            num_workers=self.num_workers,
            read_write_conflict=False,
            fit='shrink')

    def start_worker(self):

        logging.info("worker started...")

        worker_id = daisy.Context.from_env().worker_id

        output_dir = os.path.join('.extract_fragments_blockwise', self.network_dir)

        try:
            os.makedirs(output_dir)
        except:
            pass

        log_out = os.path.join(output_dir, 'extract_fragments_blockwise_%d.out' %worker_id)
        log_err = os.path.join(output_dir, 'extract_fragments_blockwise_%d.err' %worker_id)

        config = {
                'affs_file': self.affs_file,
                'affs_dataset': self.affs_dataset,
                'mask_file': self.mask_file,
                'mask_dataset': self.mask_dataset,
                'block_size': self.block_size,
                'context': self.context,
                'db_host': self.db_host,
                'db_name': self.db_name,
                'num_workers': self.num_workers,
                'fragments_in_xy': self.fragments_in_xy,
                'fragments_file': self.fragments_file,
                'fragments_dataset': self.fragments_dataset,
                'epsilon_agglomerate': self.epsilon_agglomerate,
                'queue': self.queue
        }

        config_str = ''.join(['%s'%(v,) for v in config.values()])
        config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

        config_file = os.path.join(output_dir, '%d.config'%config_hash)

        with open(config_file, 'w') as f:
            json.dump(config, f)

        logging.info('Running block with config %s...'%config_file)

        logging.info("Calling extract fragments worker...")
        daisy.call([
            'run_lsf',
            '-c', '1',
            '-g', '0',
            '-q', self.queue,
            # '-b',
            '-s', 'funkey/lsd:v0.8',
            'python', 'extract_fragments_worker.py', config_file],
            log_out=log_out,
            log_err=log_err)

    def check_block(self, block):

        done = self.blocks_extracted.count({'block_id': block.block_id}) >= 1

        return done

    def requires(self):
        return [PredictTask(global_config=self.global_config)]

if __name__ == "__main__":

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
            [
                {'task': ExtractFragmentsTask(global_config=global_config,
                **user_configs), 'request': None}
            ],
            global_config=global_config)



