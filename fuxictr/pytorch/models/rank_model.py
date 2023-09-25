# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import print_to_json, Monitor, save_attention_matrix, save_rel_score_relation
from tqdm import tqdm
import tensorflow as tf
import datetime
from pathlib import Path
        


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 save_attn_matrix=False,
                 save_rel_score_relation=False, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 params=None,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.time_keys = kwargs.get("time_keys", [])
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._save_attn_matrix = save_attn_matrix
        self._save_rel_score_relation = save_rel_score_relation
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self._dataset_id = kwargs["dataset_id"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        # Hide GPU from visible devices
        log_dir = f"log/{self._dataset_id}/{self.model_id}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        if params:
            result_filename = os.path.join(log_dir, "config.log")
            with open(result_filename, 'w') as fw:
                fw.write(print_to_json(params))
        self._total_steps = 0

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.current_lr = lr
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if feature in self.time_keys:
                pass
            else:
                if (feature_source is not None) and (spec["source"] not in feature_source):
                    continue
                if spec["type"] == "meta":
                    continue
                X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[:, self.feature_map.get_column_index(labels[0])].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[:, self.feature_map.get_column_index(self.feature_map.group_id)]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        with self.train_summary_writer.as_default():
            tf.summary.scalar("monitor_value", monitor_value, step=self._total_steps)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                self.current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(self.current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch=={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_loss', loss.item(), step=self._total_steps)
                tf.summary.scalar('learning_rate', self.current_lr, step=self._total_steps)

            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None, test=False):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if test and self._save_attn_matrix:
                attn_matrixs = []

            if test and self._save_rel_score_relation:
                rel_score = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if test and self._save_attn_matrix:
                    attn_matrixs.extend(return_dict["attn"][0].data.cpu().numpy())
                if test and self._save_rel_score_relation:    
                    rel_score.append(np.mean(return_dict["rel_score"].data.cpu().numpy(), axis=0))           
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            if test and self._save_attn_matrix:
                attn_matrixs = np.array(attn_matrixs, np.float64)
                mean_attn = np.mean(attn_matrixs, axis=0)
                # print(mean_attn)
                save_attention_matrix(mean_attn, self.model_id, self.target_field+self.default_field)

            if test and self._save_rel_score_relation: 
                rel_score = np.array(rel_score, np.float64)
                mean_rel_score = np.mean(rel_score, axis=0)
                save_rel_score_relation(mean_rel_score, self.model_id)

            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            with self.train_summary_writer.as_default():
                if test:
                    for k, v in val_logs.items():
                        tf.summary.scalar(f"Test_{k}", v, step=self._total_steps)
                else:
                    for k, v in val_logs.items():
                        tf.summary.scalar(k, v, step=self._total_steps)
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

