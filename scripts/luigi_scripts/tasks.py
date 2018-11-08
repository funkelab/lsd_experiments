import luigi
import os
import itertools
import json
from targets import *
from daisy.processes import call

# where to find the experiment directories:
this_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(this_dir, '..', '..')
def set_base_dir(d):
    global base_dir
    base_dir = d

# the MongoDB host to use
db_host = '10.40.4.51'

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def truncate_sample_name(sample):
    return sample.replace('testing/', '')[:10].replace('.', '_').replace(':', '_')

def get_db_name(experiment, setup, iteration, sample):

    return '_'.join([
        experiment,
        setup.replace('setup', ''),
        str(int(iteration/1000)) + 'k',
        truncate_sample_name(sample)
    ])

class GenericParameter(luigi.Parameter):
    pass

class LsdTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    predict_path = luigi.Parameter(default=None)

    def input_data_dir(self):
        return os.path.join(
            base_dir,
            self.experiment,
            '01_data')

    def train_dir(self):
        return os.path.join(
            base_dir,
            self.experiment,
            '02_train',
            self.setup)

    def predict_dir(self):
        if predict_path is None:
            return os.path.join(
                base_dir,
                self.experiment,
                '03_predict',
                self.setup,
                str(self.iteration))
        else:
            return os.path.realpath(predict_path)

class TrainTask(LsdTask):

    def requires(self):
        if self.iteration == 10000:
            return []
        return TrainTask(self.experiment, self.setup, self.iteration - 10000)

    def run(self):

        log_base = os.path.join(self.train_dir(), 'train_%d'%self.iteration)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(self.train_dir())

        call([
                'run_lsf',
                '-c', '5',
                '-g', '1',
                '-d', 'funkey/lsd:v0.6',
                'python -u train.py ' + str(self.iteration)
            ],
            log_out,
            log_err)

        # remove previous checkpoint if not a multiple of 100k:
        if (self.iteration - 10000)%100000 != 0:
            os.remove(self.output_filename(self.iteration - 10000))

    def output(self):
        return FileTarget(self.output_filename())

    def output_filename(self, iteration=None):

        if iteration is None:
            iteration = self.iteration

        return os.path.join(
            self.train_dir(),
            'train_net_checkpoint_%d.meta'%iteration)

class PredictionTask(LsdTask):

    sample = luigi.Parameter()

    def prediction_filename(self):
        if self.sample.endswith('.json'):
            sample = self.sample.replace('.json', '.n5')
        else:
            sample = self.sample
        return os.path.join(self.predict_dir(), sample)

class PredictTask(PredictionTask):

    predict_type = luigi.Parameter() # either 'affs' or 'lsd'

    # maximum number of GPU workers for this task
    workers_per_task = 4

    # how many chunks to process in one block, i.e., on one GPU worker
    block_size_in_chunks = [2, 7, 7]

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def run(self):

        mkdirs(self.prediction_filename())
        output_base = self.prediction_filename() + '_predict'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump({
                'experiment': self.experiment,
                'setup': self.setup,
                'iteration': self.iteration,
                'in_file': self.input_filename(),
                'in_dataset': 'volumes/raw',
                'out_file': self.prediction_filename(),
                'out_dataset': 'volumes/' + self.predict_type,
                'block_size_in_chunks': PredictTask.block_size_in_chunks,
                'num_workers': PredictTask.workers_per_task
            }, f)

        os.chdir(os.path.join(base_dir, 'scripts'))
        call([
                'python',
                '-u',
                '01_predict_blockwise.py',
                config_filename
            ],
            log_out,
            log_err)

    def input_filename(self):
        return os.path.join(self.input_data_dir(), self.sample)

    def output(self):
        return N5DatasetTarget(self.prediction_filename(), 'volumes/' + self.predict_type)

class ExtractFragmentsTask(PredictionTask):

    block_size = GenericParameter()
    context = GenericParameter()
    fragments_in_xy = luigi.BoolParameter()
    epsilon_agglomerate = luigi.FloatParameter()
    mask_fragments = luigi.BoolParameter()

    # maximum number of workers for this task
    workers_per_task = 4

    def requires(self):

        return PredictTask(
            self.experiment,
            self.setup,
            self.iteration,
            self.predict_path,
            self.sample,
            'affs')

    def run(self):

        output_base = self.prediction_filename() + '_extract'
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump({
                'affs_file': self.prediction_filename(),
                'affs_dataset': 'volumes/affs',
                'fragments_file': self.prediction_filename(),
                'fragments_dataset': 'volumes/fragments',
                'block_size': self.block_size,
                'context': self.context,
                'db_host': db_host,
                'db_name': db_name,
                'num_workers': ExtractFragmentsTask.workers_per_task,
                'fragments_in_xy': self.fragments_in_xy,
                'epsilon_agglomerate': self.epsilon_agglomerate,
                'mask_fragments': self.mask_fragments,
                'mask_file': os.path.join(self.input_data_dir(), self.sample),
                'mask_dataset': 'volumes/labels/mask'
            }, f)


        os.chdir(os.path.join(base_dir, 'scripts'))
        call([
                'run_lsf',
                '-c', str(ExtractFragmentsTask.workers_per_task),
                '-g', '0',
                '-d', 'funkey/lsd:v0.6',
                'python -u 02_extract_fragments_blockwise.py ' + config_filename
            ],
            log_out,
            log_err)

    def output(self):

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        return [
                N5DatasetTarget(self.prediction_filename(), 'volumes/fragments'),
                MongoDbCollectionTarget(db_name, db_host, 'nodes')
            ]

class AgglomerateTask(PredictionTask):

    block_size = GenericParameter()
    context = GenericParameter()
    fragments_in_xy = luigi.BoolParameter()
    epsilon_agglomerate = luigi.FloatParameter()
    mask_fragments = luigi.BoolParameter()
    merge_function = luigi.Parameter()

    # maximum number of workers for this task
    workers_per_task = 4

    def requires(self):

        return ExtractFragmentsTask(
            self.experiment,
            self.setup,
            self.iteration,
            self.predict_path,
            self.sample,
            self.block_size,
            self.context,
            self.fragments_in_xy,
            self.epsilon_agglomerate,
            self.mask_fragments)

    def run(self):

        output_base = (
            self.prediction_filename() +
            '_agglomerate_' +
            self.merge_function)
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump({
                'affs_file': self.prediction_filename(),
                'affs_dataset': 'volumes/affs',
                'fragments_file': self.prediction_filename(),
                'fragments_dataset': 'volumes/fragments',
                'block_size': self.block_size,
                'context': self.context,
                'db_host': db_host,
                'db_name': db_name,
                'num_workers': AgglomerateTask.workers_per_task,
                'merge_function': self.merge_function
            }, f)

        os.chdir(os.path.join(base_dir, 'scripts'))
        call([
                'run_lsf',
                '-c', str(AgglomerateTask.workers_per_task),
                '-g', '0',
                '-d', 'funkey/lsd:v0.6',
                'python -u 03_agglomerate_blockwise.py ' + config_filename
            ],
            log_out,
            log_err)

    def output(self):

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        return MongoDbCollectionTarget(
            db_name,
            db_host,
            'edges_' + self.merge_function,
            require_nonempty=True)

class SegmentTask(PredictionTask):

    block_size = GenericParameter()
    context = GenericParameter()
    fragments_in_xy = luigi.BoolParameter()
    epsilon_agglomerate = luigi.FloatParameter()
    mask_fragments = luigi.BoolParameter()
    merge_function = luigi.Parameter()
    threshold = luigi.FloatParameter()

    def requires(self):

        return AgglomerateTask(
            self.experiment,
            self.setup,
            self.iteration,
            self.predict_path,
            self.sample,
            self.block_size,
            self.context,
            self.fragments_in_xy,
            self.epsilon_agglomerate,
            self.mask_fragments,
            self.merge_function)

    def run(self):

        output_base = (
            self.prediction_filename() +
            '_segment_' +
            self.merge_function +
            '_%.3f'%self.threshold)
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump({
                'fragments_file': self.prediction_filename(),
                'fragments_dataset': 'volumes/fragments',
                'out_file': self.prediction_filename(),
                'out_dataset': self.segmentation_dataset(),
                'db_host': db_host,
                'db_name': db_name,
                'edges_collection': 'edges_' + self.merge_function,
                'threshold': self.threshold
            }, f)

        os.chdir(os.path.join(base_dir, 'scripts'))
        call([
                'run_lsf',
                '-c', '1',
                '-g', '0',
                '-d', 'funkey/lsd:v0.6',
                'python -u 04_extract_segmentation.py ' + config_filename
            ],
            log_out,
            log_err)

    def segmentation_dataset(self):
        return (
            'volumes/segmentation_' +
            self.merge_function +
            '_%.3f'%self.threshold)

    def output(self):

        return N5DatasetTarget(
            self.prediction_filename(),
            self.segmentation_dataset())

class EvaluateTask(LsdTask):

    sample = luigi.Parameter()
    block_size = GenericParameter()
    context = GenericParameter()
    fragments_in_xy = luigi.BoolParameter()
    epsilon_agglomerate = luigi.FloatParameter()
    mask_fragments = luigi.BoolParameter()
    merge_function = luigi.Parameter()
    border_threshold = luigi.IntParameter()
    thresholds_minmax = GenericParameter()
    thresholds_step = luigi.FloatParameter()

    def requires(self):

        return AgglomerateTask(
            self.experiment,
            self.setup,
            self.iteration,
            self.predict_path,
            self.sample,
            self.block_size,
            self.context,
            self.fragments_in_xy,
            self.epsilon_agglomerate,
            self.mask_fragments,
            self.merge_function)

    def run(self):

        output_base = self.input_filename() + '_evaluate_' + self.merge_function
        log_out = output_base + '.out'
        log_err = output_base + '.err'

        db_name = get_db_name(
            self.experiment,
            self.setup,
            self.iteration,
            self.sample)

        config_filename = output_base + '.json'
        with open(config_filename, 'w') as f:
            json.dump({
                'gt_file': self.gt_filename(),
                'gt_dataset': 'volumes/labels/neuron_ids',
                'fragments_file': self.input_filename(),
                'fragments_dataset': 'volumes/fragments',
                'border_threshold': self.border_threshold,
                'db_host': db_host,
                'rag_db_name': db_name,
                'edges_collection': 'edges_' + self.merge_function,
                'scores_db_name': self.scores_db_name(),
                'thresholds_minmax': self.thresholds_minmax,
                'thresholds_step': self.thresholds_step,
                'configuration': self.get_configuration()
            }, f)

        os.chdir(os.path.join(base_dir, 'scripts'))
        call([
                'run_lsf',
                '-c', '1',
                '-g', '0',
                '-d', 'funkey/lsd:v0.6',
                'python -u 05_evaluate.py ' + config_filename
            ],
            log_out,
            log_err)

    def gt_filename(self):
        return os.path.join(self.input_data_dir(), self.sample)

    def input_filename(self):
        return os.path.join(self.predict_dir(), self.sample)

    def scores_db_name(self):
        return self.experiment + '_' + truncate_sample_name(self.sample)

    def get_configuration(self):
        return {
            'experiment': self.experiment,
            'setup': self.setup,
            'iteration': self.iteration,
            'sample': self.sample,
            'block_size': self.block_size,
            'context': self.context,
            'fragments_in_xy': self.fragments_in_xy,
            'epsilon_agglomerate': self.epsilon_agglomerate,
            'mask_fragments': self.mask_fragments,
            'merge_function': self.merge_function,
            'border_threshold': self.border_threshold,
        }

    def output(self):

        return MongoDbDocumentTarget(
            self.scores_db_name(),
            db_host,
            'scores',
            self.get_configuration())

class EvaluateCombinations(luigi.task.WrapperTask):

    # a dictionary containing lists of parameters to evaluate
    parameters = luigi.DictParameter()
    range_keys = luigi.ListParameter()
    cls = GenericParameter(default=EvaluateTask)

    def requires(self):

        for k in self.range_keys:
            assert len(k) > 0 and k[-1] == 's', ("Explode keys have to end in "
                                                 "a plural 's'")

        # get all the values to explode
        range_values = {
            k[:-1]: v
            for k, v in self.parameters.items()
            if k in self.range_keys }

        other_values = {
            k: v
            for k, v in self.parameters.items()
            if k not in self.range_keys }

        range_keys = range_values.keys()
        tasks = []
        for concrete_values in itertools.product(*list(range_values.values())):

            parameters = { k: v for k, v in zip(range_keys, concrete_values) }
            parameters.update(other_values)

            tasks.append(self.cls(**parameters))

        return tasks
