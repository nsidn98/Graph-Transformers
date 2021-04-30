from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool

import os
import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import store_args

class naiveTransformerNet(torch.nn.Module):
    @store_args    
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int, 
                output_dim:int, heads:int=1, concat:bool=True, global_aggr:str='add',
                beta:bool=False, dropout:float=0.0, edge_dim:Optional[int]=None):
        """
            A transformer graph convolutional neural network
            Aggregates only with attention weights evaluated on 
            neighbouring nodes
            NOTE: Uses ReLU as non-linear activations after each layer
            Parameters:
            –––––––––––
            in_channels: int
                Node feature dimensions
            hidden_dim: int
                Hidden layer dimension size; NOTE will be same throughout all layers
            num_layers: int
                Number of hidden transformer layers
            output_dim: int
                Output dimension; eg: can be number of classes
            heads: int
                Number of heads in the attention layers; Default = 1
            concat: bool
                Whether to concatenate head outputs or average; Default = True
            global_aggr: str
                Global aggregation after conv layers; Default = 'add'
                Choices: 'add', 'mean'
            beta: bool
                Whether to add skip connections in the transformer layers; Default = False
            dropout: float
                Dropout probability for transformer and linear layers; Default = True
            edge_dim: int (Optional)
                Edge dimension
                Having this will include edge features in the attention evaluation
        """
        super(naiveTransformerNet, self).__init__()
        # TODO make for loop with nn.ModuleList()
        assert global_aggr in ['add', 'mean'], "Choose global aggr layer from 'mean' or 'add'"
        self.convs = nn.ModuleList()
        # add input layer
        self.convs.append(self.addTransformerConvLayer(in_channels, hidden_dim))
        # append conv layers
        for i in range(num_layers-1):
            self.convs.append(self.addTransformerConvLayer(self.getInChannels(hidden_dim), hidden_dim))
        
        self.linears = nn.Sequential(
                        nn.Linear(self.getInChannels(hidden_dim), hidden_dim),
                        nn.Dropout(dropout), ReLU(),
                        nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.node_feature, data.edge_index, data.edge_feature, data.batch

        # forward pass conv layers
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))
        
        # aggregate from all nodes
        if self.global_aggr == 'add':
            x = global_add_pool(x, batch)
        elif self.global_aggr == 'mean':
            x = global_mean_pool(x, batch)
        
        # forward pass linear layers
        x = self.linears(x)
        return x

    def addTransformerConvLayer(self, in_channels:int, out_channels:int):
        return TransformerConv(in_channels=in_channels, out_channels=out_channels,
                                heads=self.heads, concat=self.concat, beta=self.beta,
                                dropout=self.dropout, edge_dim=self.edge_dim
                                )

    def getInChannels(self, out_channels):
        """
            Given the out_channels of the 
            previous layer return in_channels
            for the next layer
            This depends on the number of heads
            and whether we are concatenating
            the head outputs
        """
        return out_channels + (self.heads-1)*self.concat*(out_channels)

if __name__ == "__main__":
    # NOTE run from the Graph-Transformers folder
    from dataloaders.snapDataloaders import DataLoaderSnap
    dl = DataLoaderSnap('ENZYMES', batch_size=2)
    trainLoader = dl.trainLoader
    data = next(iter(trainLoader))
    num_node_feats = data.num_node_features
    net = naiveTransformerNet(in_channels=num_node_feats, hidden_dim=32, 
                                num_layers=3, output_dim=2, heads=4)
    print(net)
    print(net(data).shape)