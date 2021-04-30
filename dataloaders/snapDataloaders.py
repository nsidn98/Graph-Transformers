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

import networkx as netlib
import torch_geometric.transforms as T

from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset, KarateClub, Coauthor, \
                                    Amazon, MNISTSuperpixels, PPI, QM7b
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

import os
import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import store_args

# dict to store what dataset can be used for what task
task_dataset_mapping = {
                        ''
}

class DataLoaderSnap():
    @store_args
    def __init__(self, dataset_name:str, task='graph', 
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
            dataset_raw = Planetoid(root='data/', name=dataset_name, **kwargs)
        elif dataset_name[:3] == 'TU_':
            # TU_IMDB doesn't have node features
            if dataset_name[3:] == 'IMDB':
                name = 'IMDB-MULTI'
                dataset_raw = TUDataset(root='data/', name=name, transform=T.Constant(), **kwargs)
            else:
                dataset_raw = TUDataset(root='data/', name=dataset_name[3:], **kwargs)
        elif dataset_name == 'Karate':
            dataset_raw = KarateClub()
        elif 'Coauthor' in dataset_name:
            if 'CS' in dataset_name:
                dataset_raw = Coauthor(root='data/', name='CS')
            else:
                dataset_raw = Coauthor(root='data/', name='Physics')
        elif 'Amazon' in dataset_name:
            if 'Computers' in dataset_name:
                dataset_raw = Amazon(root='data/', name='Computers')
            else:
                dataset_raw = Amazon(root='data/', name='Photo')
        elif dataset_name == 'MNIST':
            dataset_raw = MNISTSuperpixels(root='data/')
        elif dataset_name == 'PPI':
            dataset_raw = PPI(root='data/')
        elif dataset_name == 'QM7b':
            dataset_raw = QM7b(root='data/')
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')
        
        graphs = GraphDataset.pyg_to_graphs(dataset_raw, netlib=netlib)
        min_node = self.filter_graphs()
        dataset = GraphDataset(graphs, task=task, minimum_node_per_graph=min_node)
        transductive = False if self.task=='graph' else True # refer docstring
        datasets = {}
        datasets['train'], datasets['val'], datasets['test'] = dataset.split(
                            transductive=transductive, split_ratio=train_test_val_split)
        dataloaders = {split: DataLoader(
                                    dataset, collate_fn=Batch.collate(), 
                                    batch_size=batch_size, shuffle=shuffle, 
                                    num_workers=num_workers
                                    )
                                    for split, dataset in datasets.items()}
        self.trainLoader = dataloaders['train']
        self.testLoader  = dataloaders['test']
        self.valLoader   = dataloaders['val']
        self.output_dim = self.get_output_dim(dataset)

    def get_feat_dims(self):
        data = next(iter(self.testLoader))
        num_node_features = data.num_node_features
        num_edge_features = data.num_edge_features if data.num_edge_features > 0 else None
        return num_node_features, num_edge_features

    def filter_graphs(self):
        """
            Filter graphs by min number of nodes
        """
        if self.task == 'graph':
            return 0
        else:
            return 5

    def get_output_dim(self, dataset):
        """
            If task is graph classification, 
                return number of graph labels
            If task is node classifications,
                return number of node labels
            TODO: add all other task output dims
            right now just supports graph classification
        """
        if self.task == 'graph':
            return dataset.num_graph_labels

if __name__ == "__main__":
    data_list = ['TU_ENZYMES']
    for data_name in data_list:
        # for a,e in zip(na,ne):
        # params = {'use_node_attr': a, 'use_edge_attr': e}
        params = {}
        dl = DataLoaderSnap(data_name, batch_size=2, task='graph', **params)
        print('_'*50)
        # print(f'Data:{data_name}, Node feats: {a}, Edge Feats: {e}')
        print(f'Data:{data_name}')
        print(dl.get_feat_dims())
