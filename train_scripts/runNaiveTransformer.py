"""
    Script to train a Naive Graph Transformer model
    This uses the transformerConv layers
"""
from tqdm import trange
import argparse
import numpy as np
import torch
import torch.nn as nn

from typing import Dict
import os
import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))

from dataloaders.snapDataloaders import DataLoaderSnap
from models.naiveTransformer import naiveTransformerNet
from utils.utils import print_args, print_box

class Trainer():
    def __init__(self, args:argparse.Namespace, device:str, logger=None):
        self.args = args
        self.logger = logger

        self.device = device
        ### seeds ###
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ####################

        ### dataloader ###
        # NOTE hardcoding HERE; make this compatible with Cora
        print_box('Loading datasets')
        data_params = {'use_node_attr': args.use_node_attr, 'use_edge_attr': args.use_edge_attr}
        self.dl = DataLoaderSnap(dataset_name=args.dataset_name, task=args.task,
                                train_test_val_split=args.train_test_val_split, 
                                batch_size=args.batch_size, shuffle=args.shuffle, 
                                num_workers=args.num_workers, seed=args.seed, **data_params)
        self.trainLoader = self.dl.trainLoader
        self.valLoader   = self.dl.valLoader
        node_dim, edge_dim = self.dl.get_feat_dims()    # get node and edge feat dims
        output_dim = self.dl.output_dim     # get the output dimension depending on the task
        ####################

        ### network init ###
        self.network = naiveTransformerNet(in_channels=node_dim, hidden_dim=args.hidden_dim, 
                                        num_layers=args.num_layers, output_dim=output_dim,
                                        heads=args.heads, concat=args.concat_heads, 
                                        global_aggr=args.global_aggr, beta=args.beta_heads, 
                                        dropout=args.dropout, edge_dim=edge_dim).to(self.device)
        print_box(self.network, num_dash=80)
        print_box(f'Is the Network on CUDA?: {next(self.network.parameters()).is_cuda}')
        ####################

        self.optimizer = getattr(torch.optim, self.args.opt)(self.network.parameters(), 
                                                            lr=self.args.lr, 
                                                            weight_decay=self.args.weight_decay)

        if args.classification_task:
            self.loss_function = nn.CrossEntropyLoss()
            print_box('Using Cross Entropy Loss')
        else:
            self.loss_function = nn.SmoothL1Loss()
            print_box('Using Smooth L1 Loss')


    def train_epoch(self, train_step:int):
        # train one epoch 
        correct = 0; total = 0; epoch_loss = 0
        for batch_idx, data in enumerate(self.trainLoader):
            # if dryrun then only run for 100 batches
            if self.args.dryrun and batch_idx == 100:
                break
            ############
            data = data.to(self.device)
            target = data.graph_label   # TODO make this more general

            # forward pass
            output = self.network(data)
            loss = self.loss_function(output, target)
            epoch_loss += loss.item()
            ##############

            # accuracy prediction
            if self.args.classification_task:
                _, predicted = torch.max(output.data,1)
                correct += (predicted == target).sum().item()
                total   += predicted.shape[0]
            ##############

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ##############

            if (train_step % self.args.save_freq) and self.logger:
                self.save_checkpoint(self.logger.weight_save_path, train_step)

            train_step += 1
            # prog_bar.set_description(f'Epoch={epoch} Loss (loss={loss.item():.3f})')
        return epoch_loss, correct, total, train_step

    @torch.no_grad()
    def val_epoch(self):
        # perform one epoch of validation
        self.network.eval()
        val_correct = 0; val_total = 0; val_epoch_loss = 0
        for j, val_data in enumerate(self.valLoader):
            # if dryrun then end early
            if self.args.dryrun and j == 100:
                break
            val_data = val_data.to(self.device)
            val_target = val_data.graph_label   # TODO make this more general

            val_output = self.network(val_data)
            loss = self.loss_function(val_output, val_target)
            val_epoch_loss += loss.item()
            # accuracy prediction
            _, val_predicted = torch.max(val_output.data, 1)
            val_correct += (val_predicted == val_target).sum().item()
            val_total += val_predicted.shape[0]
        # important to set network back to training mode
        self.network.train()
        return val_correct, val_total, val_epoch_loss

    
    def train(self):
        train_step = 0
        # prog_bar = trange(self.args.epoch)
        for epoch in range(self.args.epoch):

            # train for one epoch
            epoch_loss, correct, total, train_step = self.train_epoch(train_step)
                        
            # validation
            val_correct, val_total, val_epoch_loss = self.val_epoch()
            
            if epoch % 10:
                self.print_metrics(epoch, epoch_loss, val_epoch_loss, correct, val_correct, total, val_total)
            if self.logger:
                self.logger.writer.add_scalar('Epoch loss', epoch_loss/len(self.trainLoader), epoch)
                if self.args.classification_task:
                    self.logger.writer.add_scalar('Train Accuracy', correct/total, epoch)
                    self.logger.writer.add_scalar('Val Accuracy', val_correct/val_total, epoch)
            # prog_bar.update(1)

    def print_metrics(self, epoch, train_loss, val_loss, train_correct, 
                            val_correct, train_total, val_total):
        """
            Print the metrics in a pretty format
        """
        train_loss = train_loss/len(self.trainLoader)
        val_loss = val_loss/len(self.valLoader)
        val_acc = val_correct/val_total
        train_acc = train_correct/train_total
        print_str = f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}'
        if self.args.classification_task:
            print_str += f', Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}'
        print_box(print_str, 100)

    def save_checkpoint(self, path:str, train_step_num: int):
        """
            Saves the model in the wandb experiment run directory
            This will store the 
                * model state_dict
                * args:
                    Will save this as a Dict as well as argparse.Namespace
            param:
                path: str
                    path to the wandb run directory
                    Example: wandb.run.dir
                train_step_num: int
                    The train step number at which model is getting saved
        """
        checkpoint = {}
        checkpoint['args'] = self.args
        checkpoint['args_dict'] = vars(self.args)
        checkpoint['state_dict'] = self.network.state_dict()
        checkpoint['train_step_num'] = train_step_num
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path:str):
        """
            Load the trained model weights
            param:
                path: str
                    path to the saved weights file
        """
        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")
        checkpoint_dict = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint_dict['state_dict']) 

if __name__ == "__main__":
    from configs.naiveTransformerConfig import args
    from utils.logger import WandbLogger

    print_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_box(f'Device: {device}') 

    if args.dryrun:
        logger = None
    else:
        logger = WandbLogger(experiment_name=args.exp_name, save_folder='NT', 
                            project='Graph Transformer', entity='graph_transformers', 
                            args=args)

    trainer = Trainer(args, device, logger)
    trainer.train()