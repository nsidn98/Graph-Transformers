import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid, TUDataset

def main(node2vec, data, args:argparse.Namespace, wt_path:str, data_idx:int,
        num_node_feats:int, device:str, logger=None, plot:bool=False):
    model = node2vec(data.x, data.edge_index, embedding_dim=args.embedding_dim, 
                    num_node_feats=num_node_feats, walk_length=args.walk_length, 
                    context_size=args.context_size, walks_per_node=args.walks_per_node,
                    node_feat_embed_dim=args.node_feat_embed_dim, p=args.p, q=args.q,
                    num_negative_samples=args.num_negative_samples, sparse=False).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        # NOTE: don't need this; Converged when loss has stabilised
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
        return acc

    @torch.no_grad()
    def plot_points():
        colors = ['#ffc0cb', '#bada55', '#008080', '#420420', 
                    '#7fe5f0', '#065535', '#ffd700']
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    for epoch in range(1, args.epoch+1):
        # print(f'Epoch: {epoch}')
        loss = train()
        if logger is not None:
            logger.writer.add_scalar(f'Loss/graph_{data_idx}', loss, epoch)
        # acc = test()
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    
    if logger is not None:
        logger.save_checkpoint(network=model, path=wt_path)

    if plot:
        plot_points()

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.getcwd()))   # add the main folder in path

    # custom imports
    from node_embedding.node2vecNode import Node2Vec
    from configs.node2vecConfig import args
    from utils.logger import WandbLogger
    from utils.utils import print_args

    # load dataset
    dataset = TUDataset('data/', args.dataset_name, 
                        use_node_attr=args.use_node_attr, 
                        use_edge_attr=args.use_edge_attr)
    num_node_feats = 0
    data = dataset[0]   # TODO change idx here

    if args.use_node_feat:
        num_node_feats = data.x.shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    logger, wt_path = None, None
    if not args.dryrun:
        save_folder = 'node2vec_' + args.dataset_name
        logger = WandbLogger(experiment_name=args.exp_name, save_folder=save_folder, 
                            project='Node Embeddings', entity='graph_transformers', args=args)
    
    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print_args(args)

    # make folder to save weights
    if logger is not None:
        wt_folder_path = os.path.join(logger.wandb_dir, 'weights/')
        if not os.path.exists(wt_folder_path):
            os.makedirs(wt_folder_path)

    # iterate through all graphs to learn embeddings
    for IDX in range(len(dataset)):
        data = dataset[IDX]
        print(f'Learning for graph {IDX+1} with {data.num_nodes} nodes')
        if logger is not None:
            wt_path = os.path.join(wt_folder_path, f'model_{IDX}.ckpt')
        main(Node2Vec, data, args, wt_path=wt_path, data_idx=IDX, 
            num_node_feats=num_node_feats, device=device, logger=logger)
