# my implementation based on paper: "Identity-aware Graph Neural Networks"

import numpy as np
import random

from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch
import torch.nn as nn
# from torch.functional import F
# from torch_geometric.nn import GATConv, GINConv, ChebConv, SAGEConv, HypergraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric import utils as pygutils
from torch import Tensor
from torch_sparse import SparseTensor, matmul
# import torch_sparse
import models


class IDGINConv(MessagePassing):
    def __init__(self, pre_nn0: Callable, pre_nn1: Callable, eps: float = 0., 
                train_eps: bool = False, device='cuda:0',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.pre_nn0 = pre_nn0
        self.pre_nn1 = pre_nn1
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = torch.Tensor([eps])
        self.eps = self.eps.to(device)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_index_opt: Adj=None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
            
        out = self.propagate(edge_index, x=x, size=size)
            
        if edge_index_opt is not None:
            out_opt = self.propagate(edge_index_opt, x=x, size=size)
            out = torch.cat([out, out_opt], dim=-1)
                
        if edge_index_opt is None:
            out += (1 + self.eps) * x[0]
        else:
            out += (1 + self.eps) * torch.cat([x[0], x[1]], dim=-1)

        return out

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor, idx: int) -> Tensor:
                              
        adj_t = adj_t.set_value(None, layout=None)
        N = len(adj_t[0])

        # NOTE: repeat K times:
        h0 = self.pre_nn0(x)
        h1 = self.pre_nn1(x)
        hk = torch.cat([h0, h1], dim=0)

        hj = x
        k_nodes, _, _, _ = pygutils.k_hop_subgraph(idx, self.K, adj_t)
        for j in k_nodes:
            neighbors = adj_t.index_select(dim=0, idx=torch.tensor([j]))
            neighbors = neighbors.to_torch_sparse_coo_tensor()._indices()[1]
            # NOTE: if n == i then xx[n + N] else: xx[n]
            neighbors[neighbors==idx] += N
            hj[j] = torch.sum(hk[neighbors, :], dim=0) # NOTE: aggregate or other agg op.

        return hj


class IDGNN(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, layer_num, dropout=0.6):
        super(IDGNN, self).__init__()
        
        self.args = args
        
        self.convs = nn.ModuleList()
        
        ln_0 = models.MLP(2*in_dim, hid_dim, hid_dim, 2)
        ln_1 = models.MLP(2*in_dim, hid_dim, hid_dim, 2)
        self.convs.append(IDGINConv(ln_0, ln_1))
        
        for _ in range(layer_num - 2):
            ln_mid0 = models.MLP(2 * hid_dim, hid_dim, hid_dim, 2)
            ln_mid1 = models.MLP(2 * hid_dim, hid_dim, hid_dim, 2)
            self.convs.append(IDGINConv(ln_mid0, ln_mid1))
            
        ln_last0 = models.MLP(2 * hid_dim, hid_dim, out_dim, 2)
        ln_last1 = models.MLP(2 * hid_dim, hid_dim, out_dim, 2)

        self.convs.append(IDGINConv(ln_last0, ln_last1))
        
    def forward(self, x, adj1, adj2=None, graphs:models.BaseGraph=None):
        N = len(adj1[0])
        out = torch.empty_like(x[0])
        for i in range(N):
            hj = x
            for conv in self.convs:
                hj = conv(hj, adj1, adj2, idx=i)
            out[i] = hj[i]
            
        return out
    