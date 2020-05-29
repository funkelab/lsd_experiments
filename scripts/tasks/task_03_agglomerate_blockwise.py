import json
import logging
import lsd
import numpy as np
import daisy
import sys
import pymongo
import hashlib
import os

import task_helper
from task_02_extract_fragments import ExtractFragmentsTask

logging.basicConfig(level=logging.INFO)

logging.getLogger('lsd.parallel_aff_agglomerate').setLevel(logging.DEBUG)

# logger = logging.getLogger(__name__)

class AgglomerateTask(daisy.Task):
    '''
    Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

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

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary below.
    '''

    experiment = daisy.Parameter()
    setup = daisy.Parameter()
    iteration = daisy.Parameter()
    affs_file = daisy.Parameter()
    affs_dataset = daisy.Parameter()
    fragments_file = daisy.Parameter()
    fragments_dataset = daisy.Parameter()
    block_size = daisy.Parameter()
    context = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    num_workers = daisy.Parameter()
    merge_function = daisy.Parameter()
    queue = daisy.Parameter()

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        self.network_dir = os.path.join(self.experiment, self.setup, str(self.iteration), self.merge_function)

        logging.info("Reading fragments from %s", self.fragments_file)
        self.fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset, mode='r')

        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        self.blocks_agglomerated = ''.join([
            'blocks_agglomerated_',
            self.merge_function])

        if ''.join(['blocks_agglomerated_', self.merge_function]) not in db.list_collection_names():
            self.blocks_agglomerated = db[self.blocks_agglomerated]
            self.blocks_agglomerated.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
        else:
            self.blocks_agglomerated = db[self.blocks_agglomerated]

        context = daisy.Coordinate(self.context)
        total_roi = self.affs.roi.grow(context, context)
        read_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size).grow(context, context)
        write_roi = daisy.Roi((0,)*self.affs.roi.dims(), self.block_size)

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

        worker_id = daisy.Context.from_env().worker_id

        output_dir = os.path.join('.agglomerate_blockwise', self.network_dir)

        try:
            os.makedirs(output_dir)
        except:
            pass

        log_out = os.path.join(output_dir, 'agglomerate_blockwise_%d.out' %worker_id)
        log_err = os.path.join(output_dir, 'agglomerate_blockwise_%d.err' %worker_id)

        config = {
                    'affs_file': self.affs_file,
                    'affs_dataset': self.affs_dataset,
                    'block_size': self.block_size,
                    'context': self.context,
                    'db_host': self.db_host,
                    'db_name': self.db_name,
                    'num_workers': self.num_workers,
                    'fragments_file': self.fragments_file,
                    'fragments_dataset': self.fragments_dataset,
                    'merge_function': self.merge_function,
                    'queue': self.queue
            }

        config_str = ''.join(['%s'%(v,) for v in config.values()])
        config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

        config_file = os.path.join(output_dir, '%d.config'%config_hash)

        with open(config_file, 'w') as f:
            json.dump(config, f)

        logging.info('Running block with config %s...'%config_file)

        base_dir = '/groups/funke/funkelab/sheridana/lsd_experiments'
        worker = 'scripts/tasks/workers/agglomerate_worker.py'

        daisy.call([
            'run_lsf',
            '-c', '1',
            '-g', '0',
            '-q', self.queue,
            '-b',
            '-s', 'funkey/lsd:v0.8',
            'python', os.path.join(base_dir, worker), config_file,
            '--run_worker'],
            log_out=log_out,
            log_err=log_err)

    def check_block(self, block):

        done = self.blocks_agglomerated.count({'block_id': block.block_id}) >= 1

        return done

    def requires(self):
        return [ExtractFragmentsTask(global_config=self.global_config)]

if __name__ == "__main__":

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
            [
                {'task': AgglomerateTask(global_config=global_config,
                **user_configs), 'request': None}
            ],
            global_config=global_config)

