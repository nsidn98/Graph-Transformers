import random
import numpy as np
import torch
from torch_sparse import coalesce
import torch_geometric
from torch_geometric.utils import add_self_loops

class AddRandomEdges(object):
    """
            Given PyG data graph,
            sample a set of random nodes from the graph
            and connect them to all the nodes
            data: torch_geometric.data.data.Data
            NOTE: this will be passed as a 'transform'
            argument to the dataset
        """
    def __init__(self, K:int, seed:int):
        self.K = K
        random.seed(seed)

    def __call__(self, data:torch_geometric.data.data.Data):
        num_nodes = data.num_nodes
        node_set = set(np.arange(num_nodes))
        K = min(self.K, num_nodes)  # NOTE some graphs may have less than K nodes
        random_nodes = torch.LongTensor(random.sample(node_set, K)).unsqueeze(0)
        edge_index = data.edge_index
        src_node_idx = torch.arange(num_nodes).long().repeat(K).unsqueeze(0)        # shape [1, num_nodes*K]
        dest_node_idx = random_nodes.repeat((num_nodes,1)).T.flatten().unsqueeze(0) # shape [1, num_nodes*K]
        rand_edge_idx = torch.cat([src_node_idx, dest_node_idx], 0) # shape [2, num_nodes*K]
        new_edge_index = torch.cat([edge_index, rand_edge_idx], 1)  # shape [2, num_nodes*K + num_edges]     
        c_edge_index, _ = coalesce(new_edge_index, None, num_nodes, num_nodes)  # remove double counted edges
        data.edge_index = c_edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class AddSelfLoop(object):
    """
            Given PyG data graph,
            sample a self loops to nodes
            data: torch_geometric.data.data.Data
            NOTE: this is different from torch_geometric.transforms.AddSelfLoops
            where the PyG version refuses to add selfloops when edge_attr are present
            NOTE: this will be passed as a 'transform' argument to the dataset
        """
    def __call__(self, data:torch_geometric.data.data.Data):
        N = data.num_nodes
        edge_index = data.edge_index

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        edge_index, _ = coalesce(edge_index, None, N, N)
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)