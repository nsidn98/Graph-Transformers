"""
    Dataloader as used in snap-stanford/deepsnap
    Using this because their method seems to be a 
    more standard way of loading graph datasets from
    Pytorch Geometric
    NOTE: Install deepsnap using `pip install deepsnap`
    Reference:
    • Deepsnap example for graph classification:
    https://github.com/snap-stanford/deepsnap/blob/master/examples/graph_classification/graph_classification_TU.py
    • TUDataset
    https://chrsmrrs.github.io/datasets/docs/datasets/
    ________________________________________________________________________________________________________
    |   Name    |   Source      |  Graphs  |  Classes  | Node Labels | Edge Labels | Node Attr | Edge Attr |
    ________________________________________________________________________________________________________
    • ENZYMES   |   TUDataset   |   600    |     6     |      +      |     -       |   +(18)   |    -      |
    • PROTEINS  |   TUDataset   |   1113   |     2     |      +      |     -       |   +(1)    |    -      |
    • MUTAG     |   TUDataset   |   188    |     2     |      +      |     +       |    -      |    -      |
    • NCl1      |   TUDataset   |   4110   |     2     |      +      |     +       |    -      |    -      |
    • AIDS      |   TUDataset   |   2000   |     2     |      +      |     +       |   +(4)    |    -      |
    • Cora      |   Planetoid   |     1    |     7     |
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""
from typing import List
import torch

# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset, KarateClub, Coauthor, \
                                    Amazon, MNISTSuperpixels, PPI, QM7b

import os
import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import store_args

# TODO dict to store what dataset can be used for what task
task_dataset_mapping = {
                        
}
multi_graph_datasets = ['MNIST', 'PPI', 'QM7b'] # 'TU' datasets are also in this

single_graph_datasets = ['Cora', 'Citeseer', 'PubMed', 'Karate', 
                        'Coauthor_CS', 'Coauthor_Physics', 
                        'Amazon_Computers', 'Amazon_Photo']

class DataLoaderMaster():
    @store_args
    def __init__(self, dataset_name:str, task='graph', transform=None,
                train_test_val_split:List=[0.8, 0.1, 0.1], 
                batch_size:int=1, shuffle:bool=False, num_workers:int=0, 
                seed:int=0, **kwargs):
        """
            Load the dataset according to the name
            Supported datasets:
            –––––––––––––––––––
            • Planetoid: 'Cora', 'Citeseer', 'PubMed'
            • TU_Dataset: 'TU_datasetName'; eg: 'TU_ENZYMES', 'TU_IMDB'
            • 'Karate'
            • 'Coauthor_CS', 'Coauthor_Physics'
            • 'Amazon_Computers', 'Amazon_Photo'
            • 'MNIST'
            • 'PPI'
            • 'QM7b' 
            Parameter:
            ––––––––––
            dataset_name: str
                Dataset name.
            task:  str
                The task for which the dataset is used for
                task = 'node or 'edge' or 'link_pred' or 'graph'.
            transform: 
                'transform' to transform the graphs
            train_test_val_split: List
                Fractions to split dataset in training, testing and validation
                Should sum to 1. NOTE: Used only for TUDataset as Planetoid just
                has a single graph
                Default: [0.8, 0.1, 0.1]
            batch_size: int
                How many samples per batch to load.
                Default: 1
            shuffle: bool
                If set to True, the data will be reshuffled at every epoch.
                Default: False
            num_workers: int
                Number of workers for dataloader
                Default: 0
            seed: int
                Random seed for reproducing randomness in shuffling
                Default: 0
            **kwargs: Dict
                Other arguments for TUDataset/Planetoid dataset
                TUDataset: {'use_node_attr': True, 'use_edge_attr': True}
                Planetoid: {'split'="public", 'num_train_per_class'=20, 'num_val'=500, 'num_test'=1000}
            Usage:
            ––––––
            dl = DataLoaderSnap('ENZYMES', batch_size=32)
            trainloader = dl.trainLoader
            for batch in trainloader:
                batch.to(device)
                label = batch.graph_label
                out = model(batch)  
                # x, edge_index, batch = data.node_feature, data.edge_index, data.batch
                ...
            transductive: bool 
                Whether the learning is transductive   
                (`True`) or inductive (`False`). Inductive split is   
                always used for the graph-level task : task=='graph'
        """
        assert task in ['node', 'edge', 'link_pred', 'graph'], "Task not supported"

        if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
            dataset_raw = Planetoid(root='data/', name=dataset_name, transform=transform, **kwargs)
        elif dataset_name[:3] == 'TU_':
            # TU_IMDB doesn't have node features
            if dataset_name[3:] == 'IMDB':
                name = 'IMDB-MULTI'
                dataset_raw = TUDataset(root='data/', name=name, transform=transform, **kwargs)
            else:
                dataset_raw = TUDataset(root='data/', name=dataset_name[3:], transform=transform, **kwargs)
        elif dataset_name == 'Karate':
            dataset_raw = KarateClub(transform=transform)
        elif 'Coauthor' in dataset_name:
            if 'CS' in dataset_name:
                dataset_raw = Coauthor(root='data/', name='CS', transform=transform)
            else:
                dataset_raw = Coauthor(root='data/', name='Physics', transform=transform)
        elif 'Amazon' in dataset_name:
            if 'Computers' in dataset_name:
                dataset_raw = Amazon(root='data/', name='Computers', transform=transform)
            else:
                dataset_raw = Amazon(root='data/', name='Photo', transform=transform)
        elif dataset_name == 'MNIST':
            dataset_raw = MNISTSuperpixels(root='data/', transform=transform)
        elif dataset_name == 'PPI':
            dataset_raw = PPI(root='data/', transform=transform)
        elif dataset_name == 'QM7b':
            dataset_raw = QM7b(root='data/', transform=transform)
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')
        
        if 'TU' in dataset_name or dataset_name in multi_graph_datasets:
            # shuffle dataset
            len_dataset = len(dataset_raw)
            torch.manual_seed(seed)
            perm = torch.randperm(len_dataset)
            # using this instead of dataset.shuffle() because I do not 
            # know which seed has to be set for its reproducibility
            self.dataset_raw = dataset_raw[perm]
            ##################
            # split into train-test-val
            train_idx = int(len_dataset * train_test_val_split[0])
            test_idx  = train_idx + int(len_dataset * train_test_val_split[1])
            train_dataset = self.dataset_raw[:train_idx]
            test_dataset  = self.dataset_raw[train_idx: test_idx]
            val_dataset   = self.dataset_raw[test_idx:]
            ##################
            # convert to dataloaders
            self.trainLoader = DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)
            self.testLoader = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)
            self.valLoader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)
            ##################
            self.get_output_dim()    
            self.get_feat_dims()
        elif dataset_name in single_graph_datasets:
            # TODO: add compatibility for single graph datasets
            # TODO: add train_test_val split and dataloader
            # use torch_geometric.transoforms.AddTrainValTestMask
            pass

    def get_feat_dims(self):
        data = self.dataset_raw[0]
        self.num_node_features = data.num_node_features
        self.num_edge_features = data.num_edge_features if data.num_edge_features > 0 else None

    def get_output_dim(self):
        """
            If task is graph classification, 
                return number of graph labels
            If task is node classifications,
                return number of node labels
            TODO: add all other task output dims
            right now just supports graph classification
        """
        if self.task == 'graph':
            self.output_dim = self.dataset_raw.num_classes

if __name__ == "__main__":
    from torch_geometric.transforms import Compose, AddSelfLoops
    from dataloaders.transforms import AddRandomEdges, AddSelfLoop
    data_list = ['TU_MUTAG']
    for data_name in data_list:
        # for a,e in zip(na,ne):
        # params = {'use_node_attr': False, 'use_edge_attr': False}
        params = {}
        # transforms = Compose([AddRandomEdges(10), AddSelfLoop()])  # NOTE can use selfloops only when no edge_attr
        transforms = Compose([])  # NOTE can use selfloops only when no edge_attr
        dl = DataLoaderMaster(data_name, batch_size=2, task='graph', transform=transforms, **params)
        print('_'*50)
        # print(f'Data:{data_name}, Node feats: {a}, Edge Feats: {e}')
        print(f'Data:{data_name}')
        print(dl.num_node_features, dl.num_edge_features, dl.output_dim)
        print('_'*50)
