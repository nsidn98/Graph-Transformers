# Information about the dataloaders:
* The code we are using currently can be found in `dataloaderMaster.py`
* We have defined the transforms to be performed on the datasets in `transforms.py` which includes addition of self_loops and addition of random edges in the graph.

## Usage example:
```python
# execute from main folder
from torch_geometric.transforms import Compose, AddSelfLoops
from dataloaders.transforms import AddRandomEdges, AddSelfLoop
from dataloaders.dataloaderMaster import DataLoaderMaster

params = {}
transforms = Compose([AddRandomEdges(10)])  # add 10 random edges per node
dl = DataLoaderMaster('TU_MUTAG', batch_size=2, task='graph', transform=transforms, **params)
print(dl.num_node_features, dl.num_edge_features, dl.output_dim)
```
