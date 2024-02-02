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


import torch
from torch import nn
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from torch.nn import MultiheadAttention
import copy


class TiSASRec(BaseModel):
    def __init__(self, 
                 feature_map, 
                 params,
                 model_id="TiSASRec", 
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
                 use_abs_time=True,
                 target_field=[("item_id", "cate_id")],
                 sequence_field=[("click_history", "cate_history")],
                 seq_pooling_type="mean", # ["mean", "sum", "target", "concat"]
                 use_causal_mask=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(TiSASRec, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  params=params,
                                  **kwargs)
        if type(target_field) != list:
            target_field = [target_field]
        self.target_field = target_field
        if type(sequence_field) != list:
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        assert len(self.target_field) == len(self.sequence_field), \
               "len(self.target_field) != len(self.sequence_field)"
        self.use_causal_mask = use_causal_mask
        self.seq_pooling_type = seq_pooling_type
        self.feature_map = feature_map
        self.target_time_key = self.time_keys[0]
        self.sequence_time_key = self.time_keys[1]
        
        self.num_pos_embeddings = kwargs["num_pos_embeddings"]
        self.num_rel_time_embeddings = kwargs["num_rel_time_embeddings"]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_abs_time = use_abs_time
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim, self.time_keys)  # without time embedding layer

        self.transformer_encoders = nn.ModuleList()
        seq_out_dim = 0
        sequence_field = self.sequence_field
        if type(sequence_field) == list:
            model_dim = embedding_dim * len(sequence_field)
            seq_len = feature_map.features[sequence_field[0]]["max_len"] + 1 # add target item
        else:
            model_dim = embedding_dim
            seq_len = feature_map.features[sequence_field]["max_len"] + 1
        seq_out_dim += self.get_seq_out_dim(model_dim, seq_len, sequence_field, embedding_dim)

        self.pos_value_embedding_layer = nn.Embedding(self.num_pos_embeddings, model_dim, padding_idx=self.num_pos_embeddings-1)
        self.pos_key_embedding_layer = nn.Embedding(self.num_pos_embeddings, model_dim, padding_idx=self.num_pos_embeddings-1)

        time_padding_idx = self.num_rel_time_embeddings-1
        self.rel_time_value_embedding_layer = nn.Embedding(self.num_rel_time_embeddings, model_dim, padding_idx=time_padding_idx)

        self.rel_time_key_embedding_layer = nn.Embedding(self.num_rel_time_embeddings, model_dim, padding_idx=time_padding_idx)

        self.transformer_encoders.append(
            TimeAwareTransformerEncoder(seq_len=seq_len,
                                model_dim=model_dim,
                                num_heads=num_heads,
                                stacked_transformer_layers=stacked_transformer_layers,
                                attn_dropout=attention_dropout,
                                net_dropout=net_dropout,
                                position_dim=embedding_dim,
                                use_position_emb=True,
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

        target_field = copy.deepcopy(self.target_field)
        sequence_field = copy.deepcopy(self.sequence_field)
                                            
        target_emb = self.concat_embedding(target_field, feature_emb_dict)
        sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)

        concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1)], dim=1)

        position_ids = torch.arange(
            start=concat_seq_emb.size(1)-1, end=-1, step=-1, dtype=torch.long, device=self.device
        )

        target_time = inputs[:, self.feature_map.get_column_index(self.target_time_key)] 
        sequence_time = inputs[:, self.feature_map.get_column_index(self.sequence_time_key)] 
        concat_time = torch.cat([sequence_time, target_time.unsqueeze(1)], dim=1)

        rel_time_matrix = self.time2rel_matrix(concat_time)

        rel_time_key_embs = self.rel_time_key_embedding_layer(rel_time_matrix)
        rel_time_value_embs = self.rel_time_value_embedding_layer(rel_time_matrix)

        seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
        padding_mask, attn_mask = self.get_mask(seq_field, X)

        pos_key_embs = self.pos_key_embedding_layer(position_ids)  
        pos_value_embs = self.pos_value_embedding_layer(position_ids)  

        transformer_out = self.transformer_encoders[0](concat_seq_emb, pos_key_embs, pos_value_embs, rel_time_key_embs, rel_time_value_embs, attn_mask) # b x len x emb
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
        
    def time2rel_matrix(self, concat_time: torch.Tensor):
        relation_matrix = concat_time.unsqueeze(2) - concat_time.unsqueeze(1)
        min_rel_time = torch.min(torch.min(torch.where(relation_matrix <= 0, torch.inf, relation_matrix), dim=1).values, dim=1).values
        relation_matrix = torch.div(relation_matrix, min_rel_time.view(-1, 1, 1))
        if self.use_abs_time:
            relation_matrix = torch.abs(relation_matrix)
        max_rel_time = self.num_rel_time_embeddings-2 if self.use_abs_time else int((self.num_rel_time_embeddings-2)/2)
        relation_matrix = relation_matrix.long()
        relation_matrix = torch.where(torch.abs(relation_matrix) > max_rel_time, max_rel_time,relation_matrix)
        if not self.use_abs_time:
            relation_matrix += int((self.num_rel_time_embeddings-2)/2)
        return relation_matrix.to(self.device)


class TimeAwareTransformerEncoder(nn.Module):
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
        super(TimeAwareTransformerEncoder, self).__init__()
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
        if self.use_position_emb:
            self.position_emb = nn.Parameter(torch.Tensor(seq_len, position_dim))
            self.reset_parameters()

    def reset_parameters(self):
        seq_len = self.position_emb.size(0)
        pe = torch.zeros(seq_len, self.position_dim)
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_dim, 2).float() * (-np.log(10000.0) / self.position_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_emb.data = pe

    def forward(self, x, pos_key_embs, pos_value_embs, rel_time_key_embs, rel_time_value_embs, attn_mask=None):
        # input b x len x dim
        batch_size = x.size(0)
        pos_key_embs = pos_key_embs.unsqueeze(0).repeat(batch_size, 1, 1)
        pos_value_embs = pos_value_embs.unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, pos_key_embs, pos_value_embs, rel_time_key_embs, rel_time_value_embs, attn_mask=attn_mask)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(TransformerBlock, self).__init__()
        self.attention = TimeAwareMultiheadAttention(model_dim,
                                            num_heads=num_heads, 
                                            dropout=attn_dropout)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout1 = nn.Dropout(net_dropout)
        self.dropout2 = nn.Dropout(net_dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim) if layer_norm else None
        self.layer_norm2 = nn.LayerNorm(model_dim) if layer_norm else None

    def forward(self, x, pos_key_embs, pos_value_embs, rel_time_key_embs, rel_time_value_embs, attn_mask=None):
        attn = self.attention(x, x, x, pos_key_embs, pos_value_embs, rel_time_key_embs, rel_time_value_embs, attn_mask=attn_mask)
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

class TimeAwareMultiheadAttention(nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, num_heads, dropout):
        super(TimeAwareMultiheadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout_rate = dropout

    def forward(self, queries, keys, values, pos_K, pos_V, time_matrix_K, time_matrix_V, attn_mask):
        bsz, tgt_len, embed_dim = queries.shape
        _, src_len, _ = keys.shape
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(values)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)

        pos_K_ = torch.cat(torch.split(pos_K, self.head_size, dim=2), dim=0)
        pos_V_ = torch.cat(torch.split(pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        paddings = torch.ones(attn_weights.shape) * (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(attn_weights.device)
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs