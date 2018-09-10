import glob
import luigi
import os
import socket
import sys
import waterz
import itertools
import json
from redirect_output import *
from shared_resource import *
from targets import *
from subprocess import check_call

base_dir = '../../'
def set_base_dir(d):
    global base_dir
    base_dir = d
db_host = '10.40.4.51'

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_db_name(experiment, setup, iteration, sample):

    return '_'.join([
        experiment,
        setup.replace('setup', ''),
        str(iteration/1000) + 'k',
        sample.replace('testing/', '') # argh...
    ])

class TrainTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()

    def requires(self):
        if self.iteration == 10000:
            return []
        return TrainTask(self.experiment, self.setup, self.iteration - 10000)

    def run(self):

        log_base = os.path.join(base_dir, self.experiment, '02_train', str(self.setup), 'train_%d'%self.iteration)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '02_train', self.setup))

        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'run_lsf',
                    '-c', '5',
                    '-g', '1',
                    '-d', 'funkey/mala:v0.1-pre3', # TODO: update to lsd docker
                    'python -u train.py ' + str(self.iteration)
                ], stdout=o, stderr=e)

        # TODO: remove previous (-10k) checkpoint, unless a multiple of 100k

    def output(self):
        return FileTarget(self.output_filename())

    def output_filename(self):
        return os.path.join(
            base_dir,
            self.experiment,
            '02_train',
            str(self.setup),
            'train_net_checkpoint_%d.meta'%self.iteration)

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    predict_type = luigi.Parameter() # either 'affs' or 'lsd'

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def run(self):

        mkdirs(self.output_filename())
        output_base = os.path.join(self.output_dir(), self.sample) + '_predict'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump(f, {
                'experiment': self.experiment,
                'setup': self.setup,
                'iteration': self.iteration,
                'sample': self.sample,
                'out_dataset': 'volumes/' + self.predict_type,
                'out_dims': 3 if self.predict_type == 'affs' else 10, # TODO: add long range affs
                'block_size_in_chunks': [8, 8, 8], # == 512 chunks
                'num_workers': 8, # TODO: maybe make parameter of this task
                'raw_dataset': 'volumes/raw'): # TODO: for Dip, we'll need raw/s0
            })

        os.chdir(os.path.join(base_dir, 'scripts'))
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'python',
                    '-u',
                    'predict_blockwise.py',
                    config_filename
                ], stdout=o, stderr=e)

    def output(self):
        return N5DatasetTarget(self.output_filename(), 'volumes/' + self.predict_type)

    def output_dir(self):
        return os.path.join(base_dir, '03_predict', self.setup, str(self.iteration))

    def output_filename(self):
        return os.path.join(self.output_dir(), '%s.n5'%self.sample)

class ExtractFragments(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    block_size = luigi.Parameter()
    context = luigi.Parameter()
    fragments_in_xy = luigi.BoolParameter()
    mask_fragments = luigi.BoolParameter()

    def requires(self):

        return ProcessTask(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample,
            'affs')

    def run(self):

        output_base = os.path.join(self.output_dir(), self.sample) + '_extract'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump(f, {
                'experiment': self.experiment,
                'setup': self.setup,
                'iteration': self.iteration,
                'sample': self.sample,
                'block_size': self.block_size
                'context': self.context,
                'db_host': db_host,
                'db_name': db_name,
                'num_workers': 8,
                'fragments_in_xy': self.fragments_in_xy,
                'mask_fragments': self.mask_fragments
            })

        os.chdir(os.path.join(base_dir, 'scripts'))
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'python',
                    '-u',
                    'extract_fragments_blockwise.py',
                    config_filename
                ], stdout=o, stderr=e)

    def output(self):

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        return [
            N5DatasetTarget(self.output_filename(), 'volumes/fragments'),
            MongoDbCollectionTarget(db_name, db_host, 'nodes')
        ]

    def output_dir(self):
        return os.path.join(base_dir, '03_predict', self.setup, str(self.iteration))

    def output_filename(self):
        return os.path.join(self.output_dir(), '%s.n5'%self.sample)

class Agglomerate(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    block_size = luigi.Parameter()
    context = luigi.Parameter()
    fragments_in_xy = luigi.BoolParameter()
    mask_fragments = luigi.BoolParameter()
    # TODO: add merge function

    def requires(self):

        return ExtractFragments(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample,
            self.block_size,
            self.context,
            self.fragments_in_xy,
            self.mask_fragments)

    def run(self):

        output_base = os.path.join(self.output_dir(), self.sample) + '_agglomerate'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump(f, {
                'experiment': self.experiment,
                'setup': self.setup,
                'iteration': self.iteration,
                'sample': self.sample,
                'block_size': self.block_size
                'context': self.context,
                'db_host': db_host,
                'db_name': db_name,
                'num_workers': 8
            })

        os.chdir(os.path.join(base_dir, 'scripts'))
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'python',
                    '-u',
                    'agglomerate_blockwise.py',
                    config_filename
                ], stdout=o, stderr=e)

    def output(self):

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        return MongoDbCollectionTarget(db_name, db_host, 'edges')

    def output_dir(self):
        return os.path.join(base_dir, '03_predict', self.setup, str(self.iteration))

    def output_filename(self):
        return os.path.join(self.output_dir(), '%s.n5'%self.sample)

class Evaluate(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    block_size = luigi.Parameter()
    context = luigi.Parameter()
    fragments_in_xy = luigi.BoolParameter()
    mask_fragments = luigi.BoolParameter()
    border_threshold = luigi.IntParameter()
    thresholds = luigi.Parameter()

    def requires(self):
        return Agglomerate(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample,
            self.block_size,
            self.context,
            self.fragments_in_xy,
            self.mask_fragments)

    def run(self):

        output_base = os.path.join(self.output_dir(), self.sample) + '_evaluate'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump(f, {
                'experiment': self.experiment,
                'setup': self.setup,
                'iteration': self.iteration,
                'sample': self.sample,
                'border_threshold': self.border_threshold,
                'db_host': db_host,
                'db_name': db_name,
                'thresholds': self.thresholds
            })

        os.chdir(os.path.join(base_dir, 'scripts'))
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'run_docker',
                    '-d', 'funkey/lsd:v0.4',
                    'python -u evaluate.py ' + config_filename
                ], stdout=o, stderr=e)

    def output_dir(self):
        return os.path.join(base_dir, '03_predict', self.setup, str(self.iteration))

    def output(self):

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        return MongoDbCollectionTarget(db_name, db_host, 'scores') # TODO: store in global scores DB

class EvaluateCombinations(luigi.task.WrapperTask):

    # a dictionary containing lists of parameters to evaluate
    parameters = luigi.DictParameter()
    range_keys = luigi.ListParameter()

    def requires(self):

        for k in self.range_keys:
            assert len(k) > 0 and k[-1] == 's', ("Explode keys have to end in "
                                                 "a plural 's'")

        # get all the values to explode
        range_values = {
            k[:-1]: v
            for k, v in self.parameters.iteritems()
            if k in self.range_keys }

        other_values = {
            k: v
            for k, v in self.parameters.iteritems()
            if k not in self.range_keys }

        range_keys = range_values.keys()
        tasks = []
        for concrete_values in itertools.product(*list(range_values.values())):

            parameters = { k: v for k, v in zip(range_keys, concrete_values) }
            parameters.update(other_values)

            tasks.append(Evaluate(**parameters))

        return tasks
