import json
import logging
import numpy as np
import os
import sys
import pymongo
import daisy
import hashlib

# from task_helper import *
import task_helper
# logging.getLogger('daisy.blocks').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class PredictTask(daisy.Task):

    '''Run prediction in parallel blocks. Within blocks, predict in chunks.

    Parameters:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):
        raw_dataset (``string``):
        lsds_file (``string``):
        lsds_dataset (``string``):

            Paths to the input datasets. lsds can be None if not needed.

        out_file (``string``):
        out_dataset (``string``):

            Path to the output datset.

        block_size_in_chunks (``tuple`` of ``int``):

            The size of one block in chunks (not voxels!). A chunk corresponds
            to the output size of the network.

        num_workers (``int``):

            How many blocks to run in parallel.
    '''

    pass

    experiment = daisy.Parameter()
    setup = daisy.Parameter()
    iteration = daisy.Parameter()
    raw_file = daisy.Parameter()
    raw_dataset = daisy.Parameter()
    out_file = daisy.Parameter()
    num_workers = daisy.Parameter()
    db_host = daisy.Parameter()
    db_name = daisy.Parameter()
    queue = daisy.Parameter()
    auto_file = daisy.Parameter(None)
    auto_dataset = daisy.Parameter(None)

    def prepare(self):
        '''Daisy calls `prepare` for each task prior to scheduling
        any block.'''

        self.experiment_dir = '../' + self.experiment
        self.data_dir = os.path.join(self.experiment_dir, '01_data')
        self.train_dir = os.path.join(self.experiment_dir, '02_train')
        self.network_dir = os.path.join(self.experiment, self.setup, str(self.iteration))

        self.raw_file = os.path.abspath(self.raw_file)

        logger.info('Input file path: ' + self.raw_file)
        logger.info('Output file path: ' + self.out_file)

        self.setup = os.path.abspath(os.path.join(self.train_dir, self.setup))
        # from here on, all values are in world units (unless explicitly mentioned)

        # get ROI of source
        try:
            source = daisy.open_ds(self.raw_file, self.raw_dataset)
        except Exception:
            raise Exception("Raw dataset not found! "
                            "Please fix file path... "
                            "raw_file: {}".format(self.raw_file))
            # in_dataset = in_dataset + '/s0'
            # source = daisy.open_ds(in_file, in_dataset)

        logger.info("Source dataset has shape %s, ROI %s, voxel size %s"%(
            source.shape, source.roi, source.voxel_size))

        # load config
        with open(os.path.join(self.setup, 'config.json')) as f:
            logger.info("Reading setup config from %s"%os.path.join(self.setup, 'config.json'))
            net_config = json.load(f)
        outputs = net_config['outputs']

        # get chunk size and context
        net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
        net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
        context = (net_input_size - net_output_size)/2

        logger.info("Following sizes in world units:")
        logger.info("net input size  = %s" % (net_input_size,))
        logger.info("net output size = %s" % (net_output_size,))
        logger.info("context         = %s" % (context,))

        input_roi = source.roi.grow(context, context)
        output_roi = source.roi

        # create read and write ROI
        block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
        block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

        logger.info("Following ROIs in world units:")
        logger.info("Total input ROI  = %s" % input_roi)
        logger.info("Block read  ROI  = %s" % block_read_roi)
        logger.info("Block write ROI  = %s" % block_write_roi)
        logger.info("Total output ROI = %s" % output_roi)

        logging.info('Preparing output dataset')

        for output_name, val in outputs.items():
            out_dims = val['out_dims']
            out_dtype = val['out_dtype']
            self.out_dataset = 'volumes/%s'%output_name

            ds = daisy.prepare_ds(
                self.out_file,
                self.out_dataset,
                output_roi,
                source.voxel_size,
                out_dtype,
                write_roi=block_write_roi,
                num_channels=out_dims,
                compressor={'id': 'zlib', 'level': 5}
                )

        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        if 'blocks_predicted' not in db.list_collection_names():
            self.blocks_predicted = db['blocks_predicted']
            self.blocks_predicted.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
        else:
            self.blocks_predicted = db['blocks_predicted']

        self.succeeded = self.schedule(
                input_roi,
                block_read_roi,
                block_write_roi,
                process_function=self.predict_worker,
                check_function=self.check_block,
                read_write_conflict=False,
                fit='overhang',
                num_workers=self.num_workers
                )

        # if not self.succeeded:
            # raise RuntimeError("Prediction failed for (at least) one block")

    def predict_worker(self):

        setup_dir = os.path.join('..', self.experiment, '02_train', self.setup)
        predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

        if self.raw_file.endswith('.json'):
            with open(self.raw_file, 'r') as f:
                spec = json.load(f)
                self.raw_file = spec['container']

        worker_config = {
            'queue': self.queue,
            'num_cpus': 2,
            'num_cache_workers': 5,
            'singularity': 'funkey/lsd:v0.8'
        }

        config = {
            'iteration': self.iteration,
            'raw_file': self.raw_file,
            'raw_dataset': self.raw_dataset,
            'auto_file': self.auto_file,
            'auto_dataset': self.auto_dataset,
            'out_file': self.out_file,
            'out_dataset': self.out_dataset,
            'db_host': self.db_host,
            'db_name': self.db_name,
            'worker_config': worker_config
        }

        # get a unique hash for this configuration
        config_str = ''.join(['%s'%(v,) for v in config.values()])
        config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

        worker_id = daisy.Context.from_env().worker_id

        output_dir = os.path.join('.predict_blockwise', self.network_dir)

        try:
            os.makedirs(output_dir)
        except:
            pass

        config_file = os.path.join(output_dir, '%d.config'%config_hash)

        log_out = os.path.join(output_dir, 'predict_blockwise_%d.out'%worker_id)
        log_err = os.path.join(output_dir, 'predict_blockwise_%d.err'%worker_id)

        with open(config_file, 'w') as f:
            json.dump(config, f)

        logging.info('Running block with config %s...'%config_file)

        command = [
            'run_lsf',
            '-c', str(worker_config['num_cpus']),
            '-g', '1',
            '-q', worker_config['queue']
        ]

        if worker_config['singularity']:
            command += ['-s', worker_config['singularity']]

        command += [
            'python -u %s %s'%(
                predict_script,
                config_file
            )]

        daisy.call(command, log_out=log_out, log_err=log_err)

        logging.info('Predict worker finished')

        # if things went well, remove temporary files
        # os.remove(config_file)
        # os.remove(log_out)
        # os.remove(log_err)

    def check_block(self, block):

        done = self.blocks_predicted.count({'block_id': block.block_id}) >= 1

        return done


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    user_configs, global_config = task_helper.parseConfigs(sys.argv[1:])

    daisy.distribute(
        [
            {'task': PredictTask(global_config=global_config, **user_configs), 'request': None}
        ],
        global_config=global_config
    )

