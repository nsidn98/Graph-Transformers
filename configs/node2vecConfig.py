import argparse
import warnings
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Argument parser for naive transformer')

##################### Experimental values #####################
parser.add_argument('-e', '--epoch', type=int, default=500, help='Number of epochs for training')
##################################################################

##################### data #####################
parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                    help='Dataset to use')
parser.add_argument('-bs','--batch_size', type=int, default=32, 
                    help='Batch Size for dataloader')
parser.add_argument('--num_workers', type=int, default=0, 
                    help='Number of dataloader workers')
parser.add_argument('--use_edge_attr', type=lambda x:bool(strtobool(x)), default=False,
                    help='To include edge features in dataset')
parser.add_argument('--use_node_attr', type=lambda x:bool(strtobool(x)), default=True,
                    help='To load extra node feats from dataset')
##################################################################

##################### network architecture #####################
parser.add_argument('--node_feat_embed_dim', type=int, default=32, 
                    help='Hidden layer dimension for node features')
parser.add_argument('--embedding_dim', type=int, default=32, 
                    help='Embedding table dimension for nodes')
parser.add_argument('--use_node_feat', type=lambda x:bool(strtobool(x)), default=False,
                    help='Whether we want to use the node features or not.\
                    If False, the equivalent to vanilla node2vec')
##################################################################

##################### node2vec  #####################
parser.add_argument('--walk_length', type=int, default=20, 
                    help='The walk length.')
parser.add_argument('--context_size', type=int, default=10, 
                    help='The actual context size which is considered for\
                    positive samples. This parameter increases the effective\
                    sampling rate by reusing samples across different source nodes.')
parser.add_argument('--walks_per_node', type=int, default=20, 
                    help='The number of walks to sample for each node.')
parser.add_argument('--num_negative_samples', type=int, default=1, 
                    help='The number of negative samples to use for each positive sample')
parser.add_argument('--p', type=int, default=1, 
                    help='Embedding table dimension for nodes')
parser.add_argument('--q', type=int, default=1, 
                    help='Embedding table dimension for nodes')
##################################################################

##################### optimizer #####################
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
##################################################################

##################### book keeping #####################
# parser.add_argument('--project', type=str, default='node2vec_proteins', help='wandb project space')
parser.add_argument('--exp_name', type=str, help='Name for experiment')
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
    return args

args = parser.parse_args()
args = config_check(args)