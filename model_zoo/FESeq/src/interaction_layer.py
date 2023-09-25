import torch
from torch import nn
from fuxictr.pytorch.layers import CrossNetV2, CrossNet, CrossNetMix
from model_zoo.AutoInt.src import MultiHeadSelfAttention 
from model_zoo.DESTINE.src import DisentangledSelfAttention
from model_zoo.FINAL.model import FinalBlock

class InteractionLayer(nn.Module):

    def __init__(self, input_dim, num_layers, layer_name, net_dropout=0.2, num_heads=1, save_attn_matrix=False):
        super(InteractionLayer, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.layer_name = layer_name
        if self.layer_name.lower() == "crossnetv2":
            self.feat_interaction_layers = CrossNetV2(self.input_dim, self.num_layers)
        elif self.layer_name.lower() == "crossnet":
            self.feat_interaction_layers = CrossNet(self.input_dim, self.num_layers)   
        elif self.layer_name.lower() == "crossnet_mix":
            self.feat_interaction_layers = CrossNetMix(self.input_dim, self.num_layers)                  
        elif self.layer_name.lower() == "autoint":
            self.feat_interaction_layers = nn.Sequential(
            *[MultiHeadSelfAttention(input_dim,
                                     attention_dim=input_dim, 
                                     dropout_rate=net_dropout,
                                     num_heads=num_heads,
                                     save_attn_matrix=save_attn_matrix,
                                     last_layer=(i==num_layers-1)) \
             for i in range(num_layers)])
        elif self.layer_name.lower() == "destine":
            self.feat_interaction_layers = nn.ModuleList([
                DisentangledSelfAttention(input_dim,
                                        attention_dim=input_dim, 
                                        num_heads=num_heads,
                                        dropout_rate=net_dropout,
                                        use_scale=True,
                                        relu_before_att=False,
                                        save_attn_matrix=save_attn_matrix,
                                        last_layer=(i==num_layers-1)) \
                for i in range(num_layers)])
        elif self.layer_name.lower() == "final":
            self.feat_interaction_layers = FinalBlock(input_dim,
                                     hidden_units=[input_dim]*num_layers,
                                     hidden_activations="ReLU",
                                     dropout_rates=[net_dropout]*num_layers)
        else:
            raise ValueError("layer_name={} not supported.".format(self.layer_name))

    def forward(self, inputs):
        if self.layer_name.lower() == "destine":
            for self_attn in self.feat_interaction_layers:
                inputs = self_attn(inputs, inputs, inputs)
            return inputs
        else:
            return self.feat_interaction_layers(inputs)

    

    