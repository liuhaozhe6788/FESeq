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
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block
from torch.nn import MultiheadAttention
from model_zoo.FESeq.src.interaction_layer import InteractionLayer
from model_zoo.FESeq.src.pooling_layer import PoolingLayer
import copy
import einops
import math



class FESeq(BaseModel):
    def __init__(self, 
                 feature_map, 
                 params,
                 model_id="FESeq", 
                 gpu=-1, 
                 dnn_hidden_units=[256, 128, 64],
                 dnn_activations="ReLU",
                 num_heads=2,
                 stacked_transformer_layers=1,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 net_dropout=0,
                 fi_dropout=0,
                 batch_norm=False,
                 layer_norm=True,
                 use_residual=True,
                 default_field=["user_id"],
                 target_field=[("item_id", "cate_id")],
                 target_item_field=[("item_id", "cate_id")],  
                 sequence_field=[("click_history", "cate_history")],
                 seq_pooling_type="mean", # ["mean", "sum", "target", "concat"]
                 seq_pooling_attn_type="bilinear_attention",
                 use_pooling_attn_scale=True,
                 rel_score_hidden_dim=128,
                 use_position_emb=True, 
                 use_time_emb=True,
                 use_causal_mask=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FESeq, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  params=params,
                                  **kwargs)
        if type(default_field) != list:
            default_field = [default_field]
        self.default_field = default_field
        if type(target_field) != list:
            target_field = [target_field]
        self.target_field = target_field
        if type(sequence_field) != list:
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        assert len(self.target_field) == len(self.sequence_field), \
               "len(self.target_field) != len(self.sequence_field)"
        if type(target_item_field) != list:
            target_item_field = [target_item_field]
        self.target_item_field = target_item_field
        self.use_causal_mask = use_causal_mask
        self.seq_pooling_type = seq_pooling_type
        self.feature_map = feature_map

        self.target_time_key = self.time_keys[0]
        self.sequence_time_key = self.time_keys[1]

        self.interaction_layer_name = kwargs["interaction_layer_name"]
        self.use_seq_feature_interaction = kwargs["use_seq_feature_interaction"]
        self.seq_feature_interaction_layers = kwargs["seq_feature_interaction_layers"]
        self.use_item_feature_interaction = kwargs.get("use_item_feature_interaction", False)
        self.item_feature_interaction_layers = kwargs.get("item_feature_interaction_layers", 0)
        
        self.num_time_embeddings = kwargs["num_time_embeddings"]
        self.time_log_base = kwargs["time_log_base"]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim, self.time_keys)  # without time embedding layer

        self.use_pos_emb = use_position_emb
        self.use_time_emb = use_time_emb

        if use_time_emb:
            self.time_embedding_layer = nn.Embedding(self.num_time_embeddings, embedding_dim, padding_idx=self.num_time_embeddings-1)

        if self.use_pos_emb:
            self.num_pos_embeddings = feature_map.features[sequence_field[0]]["max_len"] + 2
            self.pos_embedding_layer = nn.Embedding(self.num_pos_embeddings, embedding_dim, padding_idx=self.num_pos_embeddings-1)
        self.transformer_encoders = nn.ModuleList()
        seq_out_dim = 0
        sequence_field = self.sequence_field
        if type(sequence_field) == list and type(default_field) == list:
            model_dim = embedding_dim * (int(use_position_emb) + int(use_time_emb) + len(sequence_field)+len(default_field)) # concat time emb
            seq_len = feature_map.features[sequence_field[0]]["max_len"] + 1 # add target item
        else:
            model_dim = embedding_dim * (1 + int(use_position_emb) + int(use_time_emb))
            seq_len = feature_map.features[sequence_field]["max_len"] + 1
        seq_out_dim += self.get_seq_out_dim(model_dim, seq_len, sequence_field, embedding_dim)

        if type(target_item_field) == list:        
            target_item_dim = embedding_dim * len(target_item_field)
        else:
            target_item_dim = embedding_dim

        if self.use_seq_feature_interaction:
            self.seq_feature_interaction = InteractionLayer(embedding_dim * (len(sequence_field)+len(default_field)) if self.interaction_layer_name.lower() not in ["autoint", "destine"] else embedding_dim, self.seq_feature_interaction_layers, self.interaction_layer_name, net_dropout=fi_dropout, save_attn_matrix=self._save_attn_matrix)

        if self.use_item_feature_interaction and self.seq_pooling_type == "weighted_sum":
            self.item_feature_interaction = InteractionLayer(embedding_dim * len(target_item_field) if self.interaction_layer_name.lower() not in ["autoint", "destine"] else embedding_dim, self.item_feature_interaction_layers, self.interaction_layer_name, net_dropout=fi_dropout)

        self.transformer_encoders.append(
            TransformerEncoder(seq_len=seq_len,
                                model_dim=model_dim,
                                num_heads=num_heads,
                                stacked_transformer_layers=stacked_transformer_layers,
                                attn_dropout=attention_dropout,
                                net_dropout=net_dropout,
                                position_dim=embedding_dim,
                                layer_norm=layer_norm,
                                use_residual=use_residual))
        self.pooling_layer = PoolingLayer(seq_pooling_type=self.seq_pooling_type,
                                          attention_type=seq_pooling_attn_type,
                                          seq_model_dim=model_dim,
                                          item_dim=target_item_dim,
                                          net_dropout=net_dropout,
                                          attn_dropout=attention_dropout,
                                          attn_dim=rel_score_hidden_dim,
                                          use_scale=use_pooling_attn_scale)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim_wo_time(self.time_keys) + seq_out_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        if self.seq_pooling_type == "weighted_sum":
            self.output_emb_layer = nn.ModuleDict() # output vocab embedding
            for feature in flatten([self.target_item_field]):
                feature_spec = feature_map.features[feature]
                if feature_spec["type"] == "categorical":
                    self.output_emb_layer[feature] = nn.Embedding(feature_spec["vocab_size"], 
                                                            embedding_dim, 
                                                            padding_idx=feature_spec["padding_idx"])
                elif feature_spec["type"] == "numeric":
                    self.output_emb_layer[feature] = nn.Linear(1, embedding_dim, bias=False)

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
    

    def get_out_embedding(self, field, target_field, X):
        emb_list = []
        for input_name, emb_name in zip(flatten([field]), flatten([target_field])):
            if self.feature_map.features[emb_name]["type"] == "categorical" or (self.feature_map.features[emb_name]["type"] == "sequence" and self.feature_map.features[emb_name]["dtype"] == "str"):
                emb = self.output_emb_layer[emb_name](X[input_name].long())
            elif self.feature_map.features[emb_name]["type"] == "numeric" or (self.feature_map.features[emb_name]["type"] == "sequence" and self.feature_map.features[emb_name]["dtype"] == "float"):
                inp = X[input_name].float() 
                if inp.dim() == 2:
                    batch_size, seq_len = inp.size()
                    inp = inp.view(batch_size* seq_len, 1)
                    emb = self.output_emb_layer[emb_name](inp)
                    emb = emb.view(batch_size, seq_len, -1)
                elif inp.dim() == 1:
                    batch_size = inp.size()[0]
                    inp = inp.view(batch_size, 1)          
                    emb = self.output_emb_layer[emb_name](inp)
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)
        
    def forward(self, inputs):
        X = self.get_inputs(inputs) # without time embedding
        feature_emb_dict = self.embedding_layer(X)

        # generate input embeddings
        target_field = copy.deepcopy(self.target_field)
        sequence_field = copy.deepcopy(self.sequence_field)
        default_field = copy.deepcopy(self.default_field)
                                            
        target_emb = self.concat_embedding(target_field, feature_emb_dict)
        sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
        default_emb = self.concat_embedding(default_field, feature_emb_dict) if len(default_field) != 0 else None
        
        if self.seq_pooling_type == "weighted_sum":
            target_item_emb = self.get_out_embedding(self.target_item_field, self.target_item_field, X)

        concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1)], dim=1)

        sequence_field += target_field 
        for field in flatten([sequence_field]):
            feature_emb_dict.pop(field, None) # delete old embs 

        if default_emb != None:
            default_emb = einops.repeat(default_emb, 'b n -> b l n', l=concat_seq_emb.size(1))   
            concat_seq_emb = torch.cat([concat_seq_emb, default_emb], dim=-1)

        if self.use_time_emb:
            # generate time embedding
            target_time = inputs[:, self.feature_map.get_column_index(self.target_time_key)]
            sequence_time = inputs[:, self.feature_map.get_column_index(self.sequence_time_key)] 
            concat_time = torch.cat([sequence_time, target_time.unsqueeze(1)], dim=1)

            delta_times = torch.unsqueeze(target_time, dim=sequence_time.dim()-1) - concat_time

            delta_times = torch.where(delta_times < 0, 0, delta_times)
            if self.time_log_base != -1:
                delta_times = torch.log(delta_times+1)
                delta_times = torch.div(delta_times, math.log(self.time_log_base))

            delta_times = torch.where(delta_times > self.num_time_embeddings -2, self.num_time_embeddings -2, delta_times)
            delta_times = delta_times.to(self.device).long()

            time_embs = self.time_embedding_layer(delta_times)  #  time embedding

        if self.use_pos_emb:
            # generate position embedding
            position_ids = torch.arange(
                start=concat_seq_emb.size(1)-1, end=-1, step=-1, dtype=torch.long, device=self.device
            )

            pos_embs = self.pos_embedding_layer(position_ids)  #  pos embedding


        if self.use_seq_feature_interaction:
            batch_size, seq_len, total_dim = concat_seq_emb.shape
            if self.interaction_layer_name.lower() not in ["autoint", "destine"]:
                concat_seq_emb = concat_seq_emb.view((batch_size*seq_len, total_dim))
            else:
                concat_seq_emb = concat_seq_emb.view((batch_size*seq_len, total_dim//self.embedding_dim, self.embedding_dim))
            if self._save_attn_matrix and self.interaction_layer_name.lower() in ["autoint", "destine"]:
                concat_seq_emb, seq_attn_matrix = self.seq_feature_interaction(concat_seq_emb)
            else:
                concat_seq_emb = self.seq_feature_interaction(concat_seq_emb)
            concat_seq_emb = concat_seq_emb.view((batch_size, seq_len, total_dim))

        if self.use_item_feature_interaction and self.seq_pooling_type == "weighted_sum":
            batch_size, total_dim = target_item_emb.shape
            if self.interaction_layer_name.lower() not in ["autoint", "destine"]:
                target_item_emb = target_item_emb.view((batch_size, total_dim))
            else:
                target_item_emb = target_item_emb.view((batch_size, total_dim//self.embedding_dim, self.embedding_dim))
            target_item_emb = self.item_feature_interaction(target_item_emb)
            target_item_emb = target_item_emb.view((batch_size, total_dim))

        # generate mask        
        seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field
        padding_mask, attn_mask = self.get_mask(seq_field, X)

        if self.use_pos_emb:
            pos_embs = einops.repeat(pos_embs, 'm n -> k m n', k=concat_seq_emb.size(0))
            concat_seq_emb = torch.cat([concat_seq_emb, pos_embs], dim=-1)   
        if self.use_time_emb:
            concat_seq_emb = torch.cat([concat_seq_emb, time_embs], dim=-1)  

        transformer_out = self.transformer_encoders[0](concat_seq_emb, attn_mask) # b x len x emb
        if self.seq_pooling_type != "weighted_sum":
            pooling_emb = self.pooling_layer(transformer_out, padding_mask)  
        else:
            pooling_emb, rel_score_attn = self.pooling_layer(transformer_out, padding_mask, target_item_emb)            
        feature_emb_dict[f"attn_{0}"] = pooling_emb
        
        concat_emb = torch.cat(list(feature_emb_dict.values()), dim=-1)           

        y_pred = self.dnn(concat_emb)

        return_dict = dict()
        return_dict["y_pred"] = y_pred
 
        if self._save_attn_matrix and \
            self.interaction_layer_name.lower() == "autoint" and \
            self.use_seq_feature_interaction:
            return_dict["attn"]=seq_attn_matrix   
        if self._save_rel_score_relation:
            return_dict["rel_score"]=rel_score_attn                 
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

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == list:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1) 
        else:
            return feature_emb_dict[field]


class TransformerEncoder(nn.Module):
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
        super(TransformerEncoder, self).__init__()
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


    def forward(self, x, attn_mask=None):
        # input b x len x dim
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
    