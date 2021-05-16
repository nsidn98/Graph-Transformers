"""
    Another alternative to connect random nodes to each node
    For each node, this finds the non-connected nodes then samples
    random nodes from them and then connects them
    Kind of slow with larger batch sizes!
    Storing this on GitHub in case we ever want to use this later in the experiments
    Either add this in transforms or in the forward method for the networks
"""

def add_random_edges(data, num_random_nodes):
    graphList = data.G  # list of graphs in the batch
    edge_index = data.edge_index.cpu()
    num_nodes_cumsum = np.cumsum([g.number_of_nodes() for g in graphList])
    min_node_idx = 0
    new_edge_index = edge_index.clone() # keeping a clone so that while appending, the search space doesn't increase
    for graph_num in range(len(graphList)):
        max_node_idx = num_nodes_cumsum[graph_num]
        nodes_set = set(np.arange(min_node_idx, max_node_idx))                      # set of all nodes in the current graph
        for node_i in range(min_node_idx, max_node_idx):
            src_idx  = torch.where(edge_index[0] == node_i)                         # get the idx where 'node_i' is the source node
            connected_nodes = set(edge_index[1][src_idx].numpy())                   # the nodes to which 'node_i' is connected
            non_connected_nodes = nodes_set - connected_nodes                       # set of nodes not connected to 'node_i'
            # https://stackoverflow.com/questions/6494508/how-do-you-pick-x-number-of-unique-numbers-from-a-list-in-python
            num_nodes_sample = min(num_random_nodes, len(non_connected_nodes)) # number of points should not exceed nodes available
            print(node_i, num_nodes_sample)
            sampled_nodes = random.sample(non_connected_nodes, num_nodes_sample)    # list of length 'num_nodes_sample'
            src_tensor  = (torch.ones((1,len(sampled_nodes))) * node_i).long()      # shape [1, num_nodes_sample]
            dest_tensor = torch.LongTensor(sampled_nodes).unsqueeze(0)              # shape [1, num_nodes_sample]
            add_edge_index = torch.cat([src_tensor, dest_tensor], 0)                # shape [2, num_nodes_sample]
            add_edge_index.to(edge_index.device)                                    # add to the same device
            new_edge_index = torch.cat([new_edge_index, add_edge_index], 1)
        min_node_idx = num_nodes_cumsum[graph_num]                                  # important to update min_node_idx

    return new_edge_index