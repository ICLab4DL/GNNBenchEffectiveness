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
                x = self.first_h(x.float())
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

        # if self.mol:
        #     self.edge_encoder = BondEncoder(emb_dim)
        # else:
        #     self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        # if self.mol:
        #     for emb in self.edge_encoder.bond_embedding_list:
        #         nn.init.xavier_uniform_(emb.weight.data)
        # else:
        #     self.edge_encoder.reset_parameters()

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

# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))
    
full_atom_feature_dims = get_atom_feature_dims()

class MyAtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(MyAtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            print('dim:', dim)
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """
        
        x shape: torch.Size([898, 9])
        x0: tensor([5, 0, 4, 5, 3, 0, 2, 0, 0], device='cuda:1')
        
        atom x: torch.Size([1029, 9])
        atom x0: tensor([7, 0, 1, 4, 0, 0, 3, 0, 0], device='cuda:0')
        
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class EGNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(EGNN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = config['hidden_dim']
        self.num_layers = config['layer_num']

        self.node_encoder = MyAtomEncoder(self.embeddings_dim)
        # self.ln = nn.Linear(dim_features, self.embeddings_dim)
        self.mol = True
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.ln_degree = Linear(1, self.embeddings_dim)

        for i in range(self.num_layers):
            self.convs.append(
                EGINConv(self.embeddings_dim, True))

            self.bns.append(torch.nn.BatchNorm1d(self.embeddings_dim))

        self.out = nn.Linear(self.embeddings_dim, dim_target)
        self.reset_parameters()

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)

        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

        self.out.reset_parameters()

    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is not None:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_encoder(x[:, :9])
        if x.shape[-1] > 9:
            h = self.ln_degree(x[:, 9:]) + h

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, edge_attr)
            
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_attr)

        if not self.mol:
            h = self.bns[-1](h)

        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)

        h = self.out(h)

        return h