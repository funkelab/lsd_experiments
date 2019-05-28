import datetime
import json
import logging
import multiprocessing
import hashlib
import subprocess
import os
import collections
import pymongo

import daisy
import ast

logger = logging.getLogger(__name__)

def parseConfigs(args, aggregate_configs=True):
    global_configs = {}
    user_configs = {}
    hierarchy_configs = collections.defaultdict(dict)

    # first load default configs if avail
    try:
        config_file = "task_defaults.json"
        with open(config_file, 'r') as f:
            global_configs = {**json.load(f), **global_configs}
    except Exception:
        logger.info("Default task config not loaded")
        pass

    for config in args:
        print(config)
        if "=" in config:
            key, val = config.split('=')
            if '.' in key:
                task, param = key.split('.')
                hierarchy_configs[task][param] = val
            else:
                user_configs[key] = ast.literal_eval(val)
        else:
            with open(config, 'r') as f:
                print("\nhelper: loading %s" % config)
                new_configs = json.load(f)
                keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
                for k in keys:
                    if k in global_configs:
                        if k in new_configs:
                            global_configs[k].update(new_configs[k])
                    else:
                        global_configs[k] = new_configs[k]
                print(list(global_configs.keys()))

    print("\nhelper: final config")
    print(global_configs)
    print(hierarchy_configs)
    global_configs = {**hierarchy_configs, **global_configs}
    if aggregate_configs:
        aggregateConfigs(global_configs)
    return (user_configs, global_configs)


def aggregateConfigs(configs):

    input_config = configs["GlobalInput"]
    network_config = configs["Network"]

    today = datetime.date.today()
    parameters = {}
    parameters['experiment'] = input_config['experiment']
    parameters['year'] = today.year
    parameters['month'] = '%02d' % today.month
    parameters['day'] = '%02d' % today.day
    parameters['network'] = network_config['setup']
    parameters['iteration'] = network_config['iteration']

    for config in input_config:
        if isinstance(input_config[config], str):
            input_config[config] = input_config[config].format(**parameters)

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config['raw_file'] = input_config['raw_file']
        config['raw_dataset'] = input_config['raw_dataset']
        config['experiment'] = input_config['experiment']
        config['db_host'] = input_config['db_host']
        config['db_name'] = input_config['db_name']

        if 'out_file' not in config:
            config['out_file'] = input_config['out_file']

        config['setup'] = network_config['setup']
        config['iteration'] = network_config['iteration']

    if "ExtractFragmentsTask" in configs:
        config = configs["ExtractFragmentsTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['out_file']

        if 'fragments_file' not in config:
            config['fragments_file'] = input_config['out_file']
        config['db_name'] = input_config['db_name']
        config['db_host'] = input_config['db_host']
        config['experiment'] = input_config['experiment']
        config['setup'] = network_config['setup']
        config['iteration'] = network_config['iteration']

    # if "AgglomerateTask" in configs:
        # config = configs["AgglomerateTask"]
        # if 'affs_file' not in config:
            # config['affs_file'] = input_config['output_file']
        # config['fragments_file'] = input_config['output_file']
        # config['db_name'] = input_config['db_name']
        # config['db_host'] = input_config['db_host']
        # config['log_dir'] = input_config['log_dir']

    # if "SegmentationTask" in configs:
        # config = configs["SegmentationTask"]
        # config['fragments_file'] = input_config['output_file']
        # config['out_file'] = input_config['output_file']
        # config['db_name'] = input_config['db_name']
        # config['db_host'] = input_config['db_host']
        # config['log_dir'] = input_config['log_dir']

