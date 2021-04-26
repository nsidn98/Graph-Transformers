import os
import glob
import argparse
import time
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import connected_to_internet

class WandbLogger:
    def __init__(self, experiment_name:str, save_folder:str, project:str, entity:str, args:argparse.Namespace, **kwargs):
        """
            Wandb Logger Wrapper
            Parameters:
            –––––––––––
            experiment_name: str
                Name for logging the experiment on Wandboard
                Will save logs with the name {experiment_name}_{start_time}
            save_folder: str
                Name of the folder to store wandb run files
                Will save all the logs in a folder with name wandb_{save_folder}
            project: str
                Project name for wandboard
                Example: 'My Repo Name'
                This is for the Wandboard. 
                Your logs will get logged to this project name on the wandb cloud.
            entity: str
                Entity/username for wandboard
                Example: 'nsidn98'
                Your wandb userid
            args: argparse.Namespace
                Experiment arguments to save
                The arguments you want to log for the experiment
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
            Usage:
            ––––––
            ##### args #####
            import argparse
            parser = argparse.ArgumentParser(description='Traffic prediction args')
            parser.add_argument('--num_nodes', type=int, default=50, help='Number of nodes in the graph')
            parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs for training')
            args = parser.parse_args()
            ###############
            # init the logger and save the arguments used in the experiment stored in args
            logger = WandbLogger(experiment_name='myExpName', save_folder='trafficGNN', project='trafficPrediction', entity='nsidn98', args=args)
            ##### run the training loop #####
            for epoch_num in range(args.epoch):
                loss = model(x,y)
                # log the epoch loss
                logger.writer.add_scalar('Epoch Loss', loss, epoch_num)
                # can use any of the methods compatible with torch.utils.tensorboard.SummaryWriter
                # https://pytorch.org/docs/stable/tensorboard.html
                val_loss = model(x_val, y_val)
                logger.writer.add_scalar('Val Epoch Loss', val_loss, epoch_num)
                logger.save_checkpoint(network=model,train_step_num=epoch_num)
        """
        self.args = args
        # check if internet is available; if not then change wandb mode to dryrun
        if not connected_to_internet():
            import json
            # save a json file with your wandb api key in your home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser('~')+'/keys.json') as json_file: 
                key = json.load(json_file)
                my_wandb_api_key = key['my_wandb_api_key'] # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key # my Wandb api key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ['WANDB_SAVE_CODE'] = "true"

        start_time = time.strftime("%H_%M_%S-%d_%m_%Y", time.localtime())
        experiment_name = f"{experiment_name}_{start_time}"
        
        print('_'*50)
        print('Creating wandboard...')
        print('_'*50)
        wandb_save_dir = os.path.join(os.path.abspath(os.getcwd()),f"wandb_{save_folder}")
        if not os.path.exists(wandb_save_dir):
            os.makedirs(wandb_save_dir)
        wandb.init(project=project, entity=entity, sync_tensorboard=True,\
                    config=vars(args), name=experiment_name,\
                    save_code=True, dir=wandb_save_dir, **kwargs)
        
        code = wandb.Artifact('project-source', type='code')
        for path in glob.glob('*.py'):
            print(path)
            code.add_file(path)
        
        wandb.run.log_artifact(code)
        print('_'*50)
        print("Saved files used in this experiment")
        print('_'*50)

        self.writer = SummaryWriter(f"{wandb.run.dir}/{experiment_name}")
        self.weight_save_path = os.path.join(wandb.run.dir, "model.ckpt")
    
    def save_checkpoint(self, network, path:str=None, epoch:int=0):
        """
            Saves the model in the wandb experiment run directory
            This will store the 
                • model state_dict
                • args:
                    Will save this as a Dict as well as argparse.Namespace
            Parameters:
            –––––––––––
            network: nn.Module
                The network to be saved
            path: str
                path to the wandb run directory
                Example: wandb.run.dir
            epoch: int
                The epoch number at which model is getting saved
            ––––––––––––––––––––––––––––––––––––––––––––
        """
        if path is None:
            # this will set the path to be the same folder 
            # as the one where other logs are stored
            path = self.weight_save_path
        checkpoint = {}
        checkpoint['args'] = self.args
        checkpoint['args_dict'] = vars(self.args)
        checkpoint['state_dict'] = network.state_dict()
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, path)