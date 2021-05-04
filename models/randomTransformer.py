"""
    For every node in every graph, add random nodes in edge_index
    to calculate the attention weights.
    So new_edge_index = neighbours U randomly_sample_nodes
"""
from typing import Union, Tuple, Optional

import random
import numpy as np
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

class randomTransformerNet(torch.nn.Module):
    @store_args    
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int, 
                output_dim:int, num_random_nodes:int=10, heads:int=1, 
                concat:bool=True, global_aggr:str='add', beta:bool=False, 
                dropout:float=0.0, edge_dim:Optional[int]=None):
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
            num_random_nodes: int
                Number of random nodes to connect to for each node `i`
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
        super(randomTransformerNet, self).__init__()
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
        x, edge_attr, batch = data.node_feature, data.edge_feature, data.batch
        edge_index = self.add_random_edges(data)

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
    
    def add_random_edges(self, data):
        """
            Connect `self.num_random_nodes` number of 
            random nodes for each node `i` in a graph
            data: deepsnap.batch.Batch
                example: Batch(G=[5], batch=[126], edge_index=[2, 470], 
                                edge_label_index=[2, 470], graph_label=[5], 
                                node_feature=[126, 3], node_label_index=[126], 
                                task=[5])
        """
        graphList = data.G  # list of graphs in the batch
        edge_index = data.edge_index.cpu()
        num_nodes_cumsum = np.cumsum([g.number_of_nodes() for g in graphList])
        min_node_idx = 0
        new_edge_index = edge_index.clone() # keeping a clone so that while appending, the search space doesn't increase
        for graph_num in range(len(graphList)):
            max_node_idx = num_nodes_cumsum[graph_num]
            nodes_set = set(np.arange(min_node_idx, max_node_idx))                      # set of all nodes in the current graph
            for node_i in range(min_node_idx, max_node_idx):
                src_idx  = torch.where(edge_index[0] == node_i)                         # get the idx where 'node_i' is the source node
                connected_nodes = set(edge_index[1][src_idx].numpy())                   # the nodes to which 'node_i' is connected
                non_connected_nodes = nodes_set - connected_nodes                       # set of nodes not connected to 'node_i'
                # https://stackoverflow.com/questions/6494508/how-do-you-pick-x-number-of-unique-numbers-from-a-list-in-python
                num_nodes_sample = min(self.num_random_nodes, len(non_connected_nodes)) # number of points should not exceed nodes available
                sampled_nodes = random.sample(non_connected_nodes, num_nodes_sample)    # list of length 'num_nodes_sample'
                src_tensor  = (torch.ones((1,len(sampled_nodes))) * node_i).long()      # shape [1, num_nodes_sample]
                dest_tensor = torch.LongTensor(sampled_nodes).unsqueeze(0)              # shape [1, num_nodes_sample]
                add_edge_index = torch.cat([src_tensor, dest_tensor], 0)                # shape [2, num_nodes_sample]
                add_edge_index.to(edge_index.device)                                    # add to the same device
                new_edge_index = torch.cat([new_edge_index, add_edge_index], 1)
            min_node_idx = num_nodes_cumsum[graph_num]                                  # important to update min_node_idx

        return new_edge_index


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
    net = randomTransformerNet(in_channels=num_node_feats, hidden_dim=32, 
                                num_layers=3, output_dim=2, heads=4)
    print(net)
    print(net(data).shape)