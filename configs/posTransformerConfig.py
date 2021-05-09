import argparse
import warnings
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Argument parser for naive transformer')

##################### Experimental values #####################
parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs for training')
##################################################################

##################### data #####################
parser.add_argument('--dataset_name', type=str, default='TU_MUTAG',
                    help='Dataset to use')
parser.add_argument('--train_test_val_split', type=float, nargs='+', default=[0.7,0.2,0.1],
                    help='Splitting of data into train-test-val')
parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch Size for dataloader')
parser.add_argument('--shuffle', type=lambda x:bool(strtobool(x)), default=True,
                    help='Shuffle the dataloader or not')
parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
parser.add_argument('--use_edge_attr', type=lambda x:bool(strtobool(x)), default=True,
                    help='To include edge features in transformerConv')
parser.add_argument('--use_node_attr', type=lambda x:bool(strtobool(x)), default=True,
                    help='To load extra node feats from dataset')
parser.add_argument('--classification_task', type=lambda x:bool(strtobool(x)), default=True,
                    help='Whether the dataset is a classification task or regression')
parser.add_argument('--task', type=str, default='graph', choices=['graph', 'edge', 'link_pred', 'node'],
                    help='The task for which the dataset is used for')
parser.add_argument('--add_self_loops', type=lambda x:bool(strtobool(x)), default=True,
                    help='Add self loops in edge_index; NOTE: cannot use edge_attr with this')
parser.add_argument('--add_random_edges', type=lambda x:bool(strtobool(x)), default=False,
                    help='Add random edges in edge_index')
parser.add_argument('--naive_type', type=str, default='naive', choices=['naive', 'random'],
                    help='Type of naive transformer to use; Naive or Random')
parser.add_argument('--num_random_edges', type=int, default=10,
                    help='Number of random edges to add')
##################################################################

##################### network architecture #####################
parser.add_argument('--hidden_dim', type=int, default=32, 
                    help='Hidden layer dimensions for transformerConv and Linear layers')
parser.add_argument('--num_layers', type=int, default=5,
                    help='Number of transformerConv layers')
parser.add_argument('--global_aggr', type=str, default='add', choices=['add', 'mean'],
                    help='Global aggregation after conv layers')
parser.add_argument('--heads', type=int, default=1, help='Number of heads for attention layers')
parser.add_argument('--concat_heads', type=lambda x:bool(strtobool(x)), default=True,
                    help='Whether to concatenate heads or average over')
parser.add_argument('--beta_heads', type=lambda x:bool(strtobool(x)), default=True,
                    help='Use the skip information thingy in transformerConv')
parser.add_argument('--root_wt', type=lambda x:bool(strtobool(x)), default=False,
                    help='Whether to add root node feature to transformed node features\
refer: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html\
                        for more info')
parser.add_argument('--dropout', type=float, default=0, help='To dropout in posTransformerConv')
parser.add_argument('--num_pos_filters', type=int, default=10, 
                    help='Number of positional encoding filters')
##################################################################

##################### optimizer #####################
parser.add_argument('--opt', type=str, default='Adam', help='Optimizer choice.\
                    Has to be equal to pytorch class name for optimizer in torch.optim')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty) on weights')
##################################################################

##################### book keeping #####################
parser.add_argument('--project', type=str, default='naive-gcn', help='wandb project space')
parser.add_argument('--exp_name', type=str, help='Name for experiment')
parser.add_argument('--save_freq', type=int, default=100, 
                    help='Frequency of train steps to save the weights of the network')
parser.add_argument('--seed', type=int, default=0, help='Seed for all randomness')
parser.add_argument('-d','--dryrun', type=lambda x:bool(strtobool(x)), default=False, 
                    help='If just testing the code and do not want WandbLogger')
##################################################################

def print_box(string, num_dash: int = 50):
    """
        print the given string as:
        _____________________
        string
        _____________________
    """
    print('_' * num_dash)
    print(string)
    print('_' * num_dash)

def config_check(args:argparse.Namespace):
    """
        Check if the arguments are compatible
    """
    print_box('Checking Configs')
    # if dryrun the don't run for default number epochs
    if args.dryrun:
        args.epoch = 5
    
    # self loops can only be added when no edge_attr are used
    if args.add_self_loops:
        args.use_edge_attr = False
        print_box("Setting 'use_edge_attr' to False because adding self loops", 60)
    # warn for root_wt and add_self_loops both being True
    if args.root_wt and args.add_self_loops:
        warnings.warn('\nRoot weight and self loops are both True; May not work as intended\n', UserWarning)
    if args.naive_type == 'naive':
        print('_'*50)
        print("Naive Transformer, hence overriding:")
        print("'add_random_edges' to False")
        print("'num_random_edges' to 0")
        print('_'*50)
        args.add_random_edges = False
        args.num_random_edges = 0
    if args.naive_type == 'random':
        print_box("Random Naive Transformer, hence overriding: 'add_random_edges' to True", 100)
        args.add_random_edges = True
    return args

args = parser.parse_args()
args = config_check(args)