# =========================================================================
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict
import h5py


def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())

class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

    def get_metrics(self):
        return list(self.kv_pairs.keys())

def load_h5(data_path, verbose=0):
    if verbose == 0:
        logging.info('Loading data from h5: ' + data_path)
    data_dict = dict()
    with h5py.File(data_path, 'r') as hf:
        for key in hf.keys():
            data_dict[key] = hf[key][:]
    return data_dict


def save_attention_matrix(attn, path, fields):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(attn, interpolation='nearest', aspect='auto', cmap='Blues')
    plt.colorbar(format='%.3f')
    plt.xticks(ticks=list(range(len(fields))),labels=fields, rotation=45, ha='right', rotation_mode="anchor")
    plt.yticks(ticks=list(range(len(fields))),labels=fields, rotation=45, ha='right', rotation_mode="anchor")
    fig.savefig(f'{path}_attn.png', bbox_inches='tight',dpi=500)
    plt.close(fig)

def save_rel_score_relation(rel_score, path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    rel_score = rel_score.reshape(1, -1)[:,::-1]
    plt.imshow(rel_score, interpolation='spline16', aspect='auto', cmap='Blues')
    plt.colorbar(format='%.2f')
    plt.gca().yaxis.set_visible(False)
    plt.xlabel("Positions")
    fig.savefig(f'{path}_rel_score.png', bbox_inches='tight')
    plt.close(fig)

def save_embedding_dim_exp_results(auc, logloss, ylim11, ylim12, ylim21, ylim22):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 4))

    color1 = 'tab:red'
    color2 = 'tab:blue'

    ax1.set_xlabel('#Dimension')
    ax1.set_ylabel('AUC')
    lns1 = ax1.plot(auc[0][0], auc[0][1], color=color1, label="Ele.me", marker = 'D')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(ylim11)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax12.set_ylabel('AUC')
    lns12 = ax12.plot(auc[1][0], auc[1][1], color=color2, label="Bundle", marker = '*', markersize=15)
    ax12.tick_params(axis='y', labelcolor=color2)
    ax12.set_ylim(ylim12)
    ax12.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.setp([ax1,ax12],xticks=auc[0][0],xticklabels=auc[0][0])

    leg = lns1 + lns12
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, bbox_to_anchor=(0.9, 0.4))

    ax2.set_xlabel('#Dimension')
    ax2.set_ylabel('Logloss')
    lns2 = ax2.plot(logloss[0][0], logloss[0][1], color=color1, label="Ele.me", marker = 'D')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.set_ylim(ylim21)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax22 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    ax22.set_ylabel('Logloss')
    lns22 = ax22.plot(logloss[1][0], logloss[1][1], color=color2, label="Bundle", marker = '*', markersize=15)
    ax22.tick_params(axis='y', labelcolor=color2)
    ax22.set_ylim(ylim22)
    ax22.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.setp([ax2,ax22],xticks=logloss[0][0],xticklabels=logloss[0][0])

    leg = lns2 + lns22
    labs = [l.get_label() for l in leg]
    ax2.legend(leg, labs, bbox_to_anchor=(0.9, 0.4))

    fig.tight_layout()  
    plt.savefig(f'embed_dim.png', bbox_inches='tight')
    plt.close(fig)