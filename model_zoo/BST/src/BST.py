# =========================================================================
# Copyright (C) 2022. The FuxiCTR Authors. All rights reserved.
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


""" This model implements the paper: Chen et al., Behavior Sequence Transformer
    for E-commerce Recommendation in Alibaba, DLP-KDD 2021.
    [PDF] https://arxiv.org/pdf/1905.06874v1.pdf
"""


import torch
from torch import nn
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from torch.nn import MultiheadAttention
import copy


class BST(BaseModel):
    def __init__(self, 
                 feature_map, 
                 params,
                 model_id="BST", 
                 gpu=-1, 
                 dnn_hidden_units=[256, 128, 64],
                 dnn_activations="ReLU",
                 num_heads=2,
                 stacked_transformer_layers=1,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 layer_norm=True,
                 use_residual=True,
                 bst_target_field=[("item_id", "cate_id")],
                 bst_sequence_field=[("click_history", "cate_history")],
                 seq_pooling_type="mean", # ["mean", "sum", "target", "concat"]
                 use_position_emb=True,
                 use_causal_mask=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(BST, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  params=params,
                                  **kwargs)
        if type(bst_target_field) != list:
            bst_target_field = [bst_target_field]
        self.bst_target_field = bst_target_field
        if type(bst_sequence_field) != list:
            bst_sequence_field = [bst_sequence_field]
        self.bst_sequence_field = bst_sequence_field
        assert len(self.bst_target_field) == len(self.bst_sequence_field), \
               "len(self.bst_target_field) != len(self.bst_sequence_field)"
        self.use_causal_mask = use_causal_mask
        self.seq_pooling_type = seq_pooling_type
        self.feature_map = feature_map
        self.target_time_key = self.time_keys[0]
        self.sequence_time_key = self.time_keys[1]
        
        self.num_time_embeddings = kwargs["num_time_embeddings"]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim, self.time_keys)  # without time embedding layer

        self.time_embedding_layer = nn.Embedding(self.num_time_embeddings, embedding_dim, padding_idx=self.num_time_embeddings-1)
        self.transformer_encoders = nn.ModuleList()
        seq_out_dim = 0
        sequence_field = self.bst_sequence_field
        if type(sequence_field) == list:
            model_dim = embedding_dim * (int(use_position_emb) + len(sequence_field)) # concat position emb
            seq_len = feature_map.features[sequence_field[0]]["max_len"] + 1 # add target item
        else:
            model_dim = embedding_dim * (1 + int(use_position_emb))
            seq_len = feature_map.features[sequence_field]["max_len"] + 1
        seq_out_dim += self.get_seq_out_dim(model_dim, seq_len, sequence_field, embedding_dim)

        self.transformer_encoders.append(
            BehaviorTransformer(seq_len=seq_len,
                                model_dim=model_dim,
                                num_heads=num_heads,
                                stacked_transformer_layers=stacked_transformer_layers,
                                attn_dropout=attention_dropout,
                                net_dropout=net_dropout,
                                position_dim=embedding_dim,
                                use_position_emb=use_position_emb,
                                layer_norm=layer_norm,
                                use_residual=use_residual))
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim_wo_time(self.time_keys) + seq_out_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_seq_out_dim(self, model_dim, seq_len, sequence_field, embedding_dim):
        num_seq_field = len(sequence_field) if type(sequence_field) == list else 1
        if self.seq_pooling_type == "concat":
            seq_out_dim = seq_len * model_dim - num_seq_field * embedding_dim
        else:
            seq_out_dim = model_dim - num_seq_field * embedding_dim
        return seq_out_dim
        
    def forward(self, inputs):
        X = self.get_inputs(inputs) # without time embedding
        feature_emb_dict = self.embedding_layer(X)
        target_time = inputs[:, self.feature_map.get_column_index(self.target_time_key)]
        sequence_time = inputs[:, self.feature_map.get_column_index(self.sequence_time_key)] 
        concat_time = torch.cat([sequence_time, target_time.unsqueeze(1)], dim=1)

        delta_times = torch.unsqueeze(target_time, dim=sequence_time.dim()-1) - concat_time

        delta_times = torch.where(delta_times > self.num_time_embeddings -2, self.num_time_embeddings -2, delta_times)
        delta_times = torch.where(delta_times < 0, 0, delta_times)
        delta_times = delta_times.to(self.device).long()

        target_field = copy.deepcopy(self.bst_target_field)
        sequence_field = copy.deepcopy(self.bst_sequence_field)
                                            
        target_emb = self.concat_embedding(target_field, feature_emb_dict)
        sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
        concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1)], dim=1)
        seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
        padding_mask, attn_mask = self.get_mask(seq_field, X)

        time_embs = self.time_embedding_layer(delta_times)  #  time embedding
        transformer_out = self.transformer_encoders[0](concat_seq_emb, time_embs, attn_mask) # b x len x emb
        pooling_emb = self.sequence_pooling(transformer_out, padding_mask)
        feature_emb_dict[f"attn_{0}"] = pooling_emb
        sequence_field += target_field
        for field in flatten([sequence_field]):
            feature_emb_dict.pop(field, None) # delete old embs
        concat_emb = torch.cat(list(feature_emb_dict.values()), dim=-1)
        y_pred = self.dnn(concat_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_mask(self, seq_field, x):
        """ padding_mask: 1 for masked positions
            attn_mask: 1 for masked positions in nn.MultiheadAttention
        """
        if self.feature_map.features[seq_field]["dtype"] == "str":
            padding_mask = x[seq_field].long() == 0  # padding_idx = 0 required
        elif self.feature_map.features[seq_field]["dtype"] == "float":
            padding_mask = x[seq_field].float() == self.feature_map.features[seq_field]["padding_idx"]
        padding_mask = torch.cat([padding_mask, torch.zeros(x[seq_field].size(0), 1, dtype=torch.bool, device=x[seq_field].device)], dim=-1)
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len * self.num_heads, 1).view(-1, seq_len, seq_len)
        diag_zeros = (1 - torch.eye(seq_len, device=x[seq_field].device)).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x[seq_field].device), 1).bool() \
                               .unsqueeze(0).expand_as(attn_mask)
            attn_mask = attn_mask | causal_mask
        return padding_mask, attn_mask

    def sequence_pooling(self, transformer_out, mask):
        mask = (1 - mask.float()).unsqueeze(-1) # 0 for masked positions
        if self.seq_pooling_type == "mean":
            return (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            return (transformer_out * mask).sum(dim=1)
        elif self.seq_pooling_type == "target":
            return transformer_out[:, -1, :]
        elif self.seq_pooling_type == "concat":
            return transformer_out.flatten(start_dim=1)
        else:
            raise ValueError("seq_pooling_type={} not supported.".format(self.seq_pooling_type))

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == list:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]


class BehaviorTransformer(nn.Module):
    def __init__(self,
                 seq_len=1,
                 model_dim=64,
                 num_heads=8,
                 stacked_transformer_layers=1,
                 attn_dropout=0.0,
                 net_dropout=0.0,
                 use_position_emb=True,
                 position_dim=4,
                 layer_norm=True,
                 use_residual=True):
        super(BehaviorTransformer, self).__init__()
        self.position_dim = position_dim
        self.use_position_emb = use_position_emb
        self.transformer_blocks = nn.ModuleList(TransformerBlock(model_dim=model_dim,
                                                                 ffn_dim=model_dim,
                                                                 num_heads=num_heads, 
                                                                 attn_dropout=attn_dropout, 
                                                                 net_dropout=net_dropout,
                                                                 layer_norm=layer_norm,
                                                                 use_residual=use_residual)
                                                for _ in range(stacked_transformer_layers))


    def forward(self, x, time_embs, attn_mask=None):
        # input b x len x dim
        if self.use_position_emb:
            x = torch.cat([x, time_embs], dim=-1)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, attn_mask=attn_mask)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(model_dim,
                                            num_heads=num_heads, 
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout1 = nn.Dropout(net_dropout)
        self.dropout2 = nn.Dropout(net_dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim) if layer_norm else None
        self.layer_norm2 = nn.LayerNorm(model_dim) if layer_norm else None

    def forward(self, x, attn_mask=None):
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        s = self.dropout1(attn)
        if self.use_residual:
            s += x
        if self.layer_norm1 is not None:
            s = self.layer_norm1(s)
        out = self.dropout2(self.ffn(s))
        if self.use_residual:
            out += s
        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)
        return out
