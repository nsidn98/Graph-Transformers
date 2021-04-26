"""
    Code for loading the following datasets into 
    torch geometric compatible dataloaders:
    TUDataset: https://chrsmrrs.github.io/datasets/docs/datasets/
    _______________________________________________________________________________________________________________
    |   Name    |   Source      |      |  Graphs  |  Classes  | Node Labels | Edge Labels | Node Attr | Edge Attr |
    _______________________________________________________________________________________________________________
    • ENZYMES   |   TUDataset   |   ✔︎  |   600    |     6     |      +      |     -       |   +(18)   |    -      |
    • PROTEINS  |   TUDataset   |   ✔︎  |   1113   |     2     |      +      |     -       |   +(1)    |    -      |
    • MUTAG     |   TUDataset   |   ✔︎  |   188    |     2     |      +      |     +       |    -      |    -      |
    • NCl1      |   TUDataset   |   x  |   4110   |     2     |      +      |     +       |    -      |    -      |
    • AIDS      |   TUDataset   |   ✔︎  |   2000   |     2     |      +      |     +       |   +(4)    |    -      |
    • Cora      |   Planetoid   |   x  |     1    |     7     |
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    • OGB       |   NOTE: Check this https://ogb.stanford.edu/docs/dataset_overview/
"""
from typing import List
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset

# dict to manage where to load dataset from in torch_geometric
dataset_sources = {
                    'ENZYMES': TUDataset,
                    'PROTEINS': TUDataset,
                    'MUTAG': TUDataset,
                    'NCl1': TUDataset,
                    'AIDS': TUDataset,
                    'Cora': Planetoid,
    }

class DataLoaderBase():
    def __init__(self, dataset_name:str, train_test_val_split:List=[0.7, 0.2, 0.1], 
                batch_size:int=1, shuffle:bool=False, num_workers:int=0, 
                seed:int=0, **kwargs):
        """
            Load the dataset according to the name
            Currently supports:
                _____________________________
                • ENZYMES   |   TUDataset   |
                • PROTEINS  |   TUDataset   |
                • MUTAG     |   TUDataset   |
                • NCl1      |   TUDataset   |
                • AIDS      |   TUDataset   |
                • Cora      |   Planetoid   |
                –––––––––––––––––––––––––––––
            Parameter:
            ––––––––––
            dataset_name: str
                Dataset name. Currently supports:
                ENZYMES, PROTEINS, MUTAG, NCl1, AIDS, Cora
            train_test_val_split: List
                Fractions to split dataset in training, testing and validation
                Should sum to 1. NOTE: Used only for TUDataset as Planetoid just
                has a single graph
                Default: [0.7, 0.2, 0.1]
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
        """
        self.dataset = dataset_sources[dataset_name](root='data/', name=dataset_name, **kwargs,)
        len_dataset = len(self.dataset)
        # Cora has only one graph
        if dataset_name != 'Cora':
            # shuffle dataset
            torch.manual_seed(seed)
            perm = torch.randperm(len_dataset)
            # using this instead of dataset.shuffle() because I do not 
            # know which seed has to be set for its reproducibility
            self.dataset = self.dataset[perm]
            ##################
            # split into train-test-val
            train_idx = int(len_dataset * train_test_val_split[0])
            test_idx  = train_idx + int(len_dataset * train_test_val_split[1])
            self.train_dataset = self.dataset[:train_idx]
            self.test_dataset  = self.dataset[train_idx: test_idx]
            self.val_dataset   = self.dataset[test_idx:]
            ##################
            # convert to dataloaders
            self.trainLoader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers)
            self.testLoader  = DataLoader(self.test_dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers)
            self.valLoader   = DataLoader(self.val_dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers)
            ##################
        elif dataset_name == 'Cora':
            # TODO add Cora compatibility
            print('_'*50)
            print("Currently doesn't support Cora")
            print('_'*50)
            pass

    def get_feat_dims(self):
        data = next(iter(self.testLoader))
        num_node_features = data.num_node_features
        num_edge_features = data.num_edge_features if data.num_edge_features > 0 else None
        return num_node_features, num_edge_features

if __name__ == "__main__":
    na = [True, True, False, False]
    ne = [True, False, True, False]
    data_list = ['ENZYMES','PROTEINS','MUTAG','AIDS']
    for data_name in data_list:
        for a,e in zip(na,ne):
            params = {'use_node_attr': a, 'use_edge_attr': e}
            dl = DataLoaderBase(data_name, batch_size=2, **params)
            print('_'*50)
            print(f'Data:{data_name}, Node feats: {a}, Edge Feats: {e}')
            print(dl.get_feat_dims())
