from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool

import os
import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import store_args

class naiveTransformerNet(torch.nn.Module):
    @store_args    
    def __init__(self, in_channel:int,
                heads:int=1, concat:bool=True, beta:bool=False, 
                dropout:float=0.0, edge_dim:Optional[int]=None):
        super(naiveTransformerNet, self).__init__()
        # TODO make for loop with nn.ModuleList()
        self.conv1 = TransformerConv(in_channels=in_channel, out_channels=10, 
                                    heads=heads,concat=concat, beta=beta, 
                                    dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels=self.getInChannels(10), out_channels=5, 
                                    heads=heads, concat=concat, beta=beta, 
                                    dropout=dropout, edge_dim=edge_dim)
        self.lin1 = nn.Linear(self.getInChannels(5), 10)
        self.lin2 = nn.Linear(10, 3)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)  # aggregate from all nodes
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

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
    # run from the Graph-Transformers folder
    from dataloaders.dataloaderBase import DataLoaderBase
    dl = DataLoaderBase('ENZYMES', batch_size=2)
    trainLoader = dl.trainLoader
    data = next(iter(trainLoader))
    num_node_feats = data.num_node_features
    net = naiveTransformerNet(in_channel=num_node_feats)
    print(net)
    print(net(data).shape)