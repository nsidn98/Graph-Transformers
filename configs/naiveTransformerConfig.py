import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Argument parser for naive transformer')

##################### Experimental values #####################
parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs for training')
##################################################################

##################### data #####################
parser.add_argument('--dataset_name', type=str, default='MUTAG', 
                    choices=['ENZYMES', 'PROTEINS', 'MUTAG', 'AIDS'],
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
##################################################################

##################### network architecture #####################
parser.add_argument('--heads', type=int, default=1, help='Number of heads for attention layers')
parser.add_argument('--concat_heads', type=lambda x:bool(strtobool(x)), default=True,
                    help='Whether to concatenate heads or average over')
parser.add_argument('--beta_heads', type=lambda x:bool(strtobool(x)), default=True,
                    help='Use the skip information thingy in transformerConv')
parser.add_argument('--dropout', type=float, default=0, help='To dropout in transformerConv')
##################################################################

##################### optimizer #####################
parser.add_argument('--opt', type=str, default='Adam', help='Optimizer choice.\
                    Has to be equal to pytorch class name for optimizer in torch.optim')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty) on weights')
##################################################################

##################### book keeping #####################
parser.add_argument('--project', type=str, default='naive-gcn', help='wandb project space')
parser.add_argument('--save_freq', type=int, default=100, 
                    help='Frequency of train steps to save the weights of the network')
parser.add_argument('--seed', type=int, default=0, help='Seed for all randomness')
parser.add_argument('-d','--dryrun', type=lambda x:bool(strtobool(x)), default=False, 
                    help='If just testing the code and do not want WandbLogger')
##################################################################

def config_check(args:argparse.Namespace):
    """
        Check if the arguments are compatible
    """
    # if dryrun the don't run for default number epochs
    if args.dryrun:
        args.epoch = 2

    return args

args = parser.parse_args()
args = config_check(args)