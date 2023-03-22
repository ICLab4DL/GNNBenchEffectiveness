import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool, BatchNorm


class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = [config['hidden_units'][0]] + config['hidden_units']
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config['train_eps']
        if config['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool
        elif config['aggregation'] == 'max':
            self.pooling = global_max_pool

        # self.batch0 = BatchNorm1d(dim_features)
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2
                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        # TODO: batch normalization:
        # x = self.batch0(x)
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                # NOTE: residual connection
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out
    




import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv
from torch_geometric.utils import degree

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class EGINConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr="add")

        self.mol = mol

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(
            2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # edge_embedding = self.edge_encoder(edge_attr)
        # out = self.mlp((1 + self.eps) * x +
        #                self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=None))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return F.relu(x_j)
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGCNConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        super(EGCNConv, self).__init__(aggr='add')

        self.mol = mol

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()
        
        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(EGNN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = config['hidden_units']
        self.num_layers = config['layer_num']

        # self.node_encoder = AtomEncoder(self.embeddings_dim)
        self.ln = nn.Linear(dim_features, self.embeddings_dim)
        self.mol = True
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(
                EGINConv(self.embeddings_dim, True))

            self.bns.append(torch.nn.BatchNorm1d(self.embeddings_dim))

        self.out = nn.Linear(self.embeddings_dim, dim_target)

    def reset_parameters(self):
        # if self.mol:
        #     for emb in self.node_encoder.atom_embedding_list:
        #         nn.init.xavier_uniform_(emb.weight.data)
        # else:
        #     nn.init.xavier_uniform_(self.node_encoder.weight.data)

        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

        self.ln.reset_parameters()
        self.out.reset_parameters()

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        # x = x.long()
        # h = self.node_encoder(x)
        h = self.ln(x.float())
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            # print('h3 shape:', h.shape)

            h = F.relu(h)
            
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_attr)

        if not self.mol:
            h = self.bns[-1](h)

        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)

        h = self.out(h)

        return h