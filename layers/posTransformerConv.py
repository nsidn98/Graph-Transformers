"""
    Positional transformer Conv
    • d(u,v)  = || x_u - x_v ||
    • P_{u,v} = [cos(a_i . d(u,v)); sin(a_i . d(u,v))] for i \in [1, ..., K]
    • A_{u,v} = MLP(P_{u,v}).view(d,d)
    • α_{u,v} = σ( ((W_u.x_u)^T . A_{u,v} . (W_v.x_v)) / sqrt(d))
"""

import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

import os, sys
# this adds the Graph-Transformers folder in path
# so the imports from different folders will work
sys.path.append(os.path.abspath(os.getcwd()))
from layers.positionalEncoding import PositionalEncoding


class PosTransformerConv(MessagePassing):
    r"""
        The graph transformer operator from the `"Masked Label Prediction:
        Unified Message Passing Model for Semi-Supervised Classification"
        with the positional encoding

        Args:
            in_channels (int or tuple): Size of each input sample. A tuple
                corresponds to the sizes of source and target dimensionalities.
            out_channels (int): Size of each output sample.
            heads (int, optional): Number of multi-head-attentions.
                (default: :obj:`1`)
            concat (bool, optional): If set to :obj:`False`, the multi-head
                attentions are averaged instead of concatenated.
                (default: :obj:`True`)
            beta (bool, optional): If set, will combine aggregation and
                skip information via

                .. math::
                    \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                    (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                    \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

                with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
                [ \mathbf{x}_i, \mathbf{m}_i, \mathbf{x}_i - \mathbf{m}_i ])`
                (default: :obj:`False`)
            dropout (float, optional): Dropout probability of the normalized
                attention coefficients which exposes each node to a stochastically
                sampled neighborhood during training. (default: :obj:`0`)
            edge_dim (int, optional): Edge feature dimensionality (in case
                there are any). Edge features are added to the keys after
                linear transformation, that is, prior to computing the
                attention dot product. They are also added to final values
                after the same linear transformation. The model is:

                .. math::
                    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                    \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                    \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                    \right),

                where the attention coefficients :math:`\alpha_{i,j}` are now
                computed via:

                .. math::
                    \alpha_{i,j} = \textrm{softmax} \left(
                    \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                    (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                    {\sqrt{d}} \right)

                (default :obj:`None`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            root_weight (bool, optional): If set to :obj:`False`, the layer will
                not add the transformed root node features to the output and the
                option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                heads: int = 1, concat: bool = True, beta: bool = False,
                dropout: float = 0., edge_dim: Optional[int] = None,
                bias: bool = True, root_weight: bool = True, 
                num_pos_filters: int = 10, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PosTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.num_pos_filters = num_pos_filters

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.lin_A = Linear(2*num_pos_filters, heads * out_channels * out_channels)
        self.posEnc = PositionalEncoding(self.num_pos_filters)  # NOTE this inits weights to Unif(0,1)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_A.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        """"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # distance between node embeddings
        # • d(u,v)  = || x_u - x_v ||
        d_ij = torch.linalg.norm(x_i - x_j, dim=-1).unsqueeze(1)    # shape [batch, 1]
        # • P_{u,v} = [cos(a_i . d(u,v)); sin(a_i . d(u,v))] for i \in [1, ..., K]
        P_ij = self.posEnc(d_ij)    # shape [batch, 2K]
        # • A_{u,v} = MLP(P_{u,v}).view(d,d)
        A_ij = self.lin_A(P_ij).view(-1, self.heads, self.out_channels, self.out_channels)  # shape [batch, heads, d, d]
        # get K and Q
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                    self.out_channels)
            key += edge_attr

        # alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        # perform α_{u,v} = σ( ((W_u.x_u)^T . A_{u,v} . (W_v.x_v)) / sqrt(d))
        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        # https://discuss.pytorch.org/t/how-to-implement-4d-tensor-multiplication/108476/5
        AQ = torch.einsum('bhij, bhj -> bhi', A_ij, query) # NOTE @nsidn98, @octavian-ganea check if it is 'bhij, bhi -> bhj'
        # AQ is of shape [batch, heads, d]
        alpha = (key * AQ).sum(dim=-1) / math.sqrt(self.out_channels)   # shape [batch, heads]
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out
    

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels, self.heads)