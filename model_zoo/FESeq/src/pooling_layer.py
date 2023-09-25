import torch
from torch import nn
from fuxictr.pytorch.layers import MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class PoolingLayer(nn.Module):

    def __init__(self,
                 seq_pooling_type,
                 attention_type="bilinear_attention",
                 seq_model_dim=None,
                 item_dim=None,
                 attn_dim=128,
                 num_heads=1,
                 net_dropout=0.1,
                 attn_dropout=0.1,
                 use_scale=True):
        super(PoolingLayer, self).__init__()
        self.seq_pooling_type = seq_pooling_type
        self.attention_type = attention_type
        if self.seq_pooling_type == "weighted_sum":
            self.activation = get_activation("prelu", 1)
            self.head_dim = attn_dim // num_heads
            self.num_heads = num_heads
            self.W_q = nn.Linear(item_dim, attn_dim, bias=False)
            self.W_k = nn.Linear(seq_model_dim, attn_dim, bias=False)
            self.W_v = nn.Linear(seq_model_dim, attn_dim, bias=False)
            self.ffn = nn.Sequential(nn.Linear(attn_dim, seq_model_dim),
                        self.activation)
  
            self.scale = attn_dim** 0.5 if use_scale else None
            if attention_type == "bilinear_attention":
                self.W_kernel = nn.Parameter(torch.eye(self.head_dim))
            elif attention_type == "din_attention":
                self.attn_mlp = MLP_Block(input_dim=self.head_dim * 4,
                                        output_dim=1,
                                        hidden_units=[512, 256, 128],
                                        hidden_activations="dice",
                                        output_activation=None, 
                                        dropout_rates=attn_dropout,
                                        batch_norm=False)
            self.net_dropout = nn.Dropout(net_dropout) if net_dropout > 0 else None
            self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(self, transformer_out, mask, target_item_emb=None):
        mask = (1 - mask.float()).unsqueeze(-1) # 0 for masked positions
        if self.seq_pooling_type == "mean":
            return (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            return (transformer_out * mask).sum(dim=1), 
        elif self.seq_pooling_type == "target":
            return transformer_out[:, -1, :]
        elif self.seq_pooling_type == "concat":
            return transformer_out.flatten(start_dim=1)
        elif self.seq_pooling_type == "weighted_sum":
            user_behavior_key_emb = self.W_k(transformer_out)
            user_behavior_value_emb = self.W_v(transformer_out)
            target_item_emb = self.W_q(target_item_emb)
            user_behavior_key_emb = self.activation(user_behavior_key_emb)
            user_behavior_value_emb = self.activation(user_behavior_value_emb)
            target_item_emb = self.activation(target_item_emb)

            batch_size = transformer_out.size(0)
            target_item_emb = target_item_emb.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            user_behavior_key_emb = user_behavior_key_emb.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            user_behavior_value_emb = user_behavior_value_emb.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.net_dropout is not None:
                user_behavior_key_emb = self.net_dropout(user_behavior_key_emb)
                user_behavior_value_emb = self.net_dropout(user_behavior_value_emb)
                target_item_emb = self.net_dropout(target_item_emb)

            seq_len = transformer_out.size(1)
            if self.attention_type == "dot_attention":
                attn_score = (user_behavior_key_emb @ target_item_emb.transpose(-1, -2))
                if self.scale:
                   attn_score = attn_score/self.scale
            elif self.attention_type == "bilinear_attention":
                attn_score = ((user_behavior_key_emb @ self.W_kernel) @ target_item_emb.transpose(-1, -2))
                if self.scale:
                   attn_score = attn_score/self.scale
            elif self.attention_type == "din_attention":
                target_item_emb = target_item_emb.expand(batch_size, self.num_heads, seq_len, -1)
                din_concat = torch.cat([target_item_emb, user_behavior_key_emb, target_item_emb - user_behavior_key_emb, 
                                        target_item_emb * user_behavior_key_emb], dim=-1)
                attn_score = self.attn_mlp(din_concat.view(-1, 4 * target_item_emb.size(-1)))
            attn_score = attn_score.view(batch_size, self.num_heads, seq_len)
            if mask is not None:
                mask = mask.transpose(-1, -2).expand(-1, self.num_heads, -1).view_as(attn_score)
                attn_score = attn_score.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
            attn_score = attn_score.softmax(dim=-1)
            if self.attn_dropout is not None:
                attn_score = self.attn_dropout(attn_score)
            output = torch.sum(attn_score.unsqueeze(-1) * user_behavior_value_emb, dim=2)
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim).squeeze(1)
            return self.ffn(output), attn_score
        else:
            raise ValueError("seq_pooling_type={} not supported.".format(self.seq_pooling_type))