import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, BatchNorm
from models.graph_classifiers.GIN import GIN


class ModelAdapter(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(ModelAdapter, self).__init__()

        hid_out_dim = config['hidden_units'][-1]
        
        self.gin1 = GIN(1, hid_out_dim, config)
        self.gin2 = GIN(dim_features-1, hid_out_dim, config)
        self.ln = Linear(2*hid_out_dim, dim_target)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # NOTE: seperate features:
        out1 = self.gin1(x=x[..., -1:], edge_index=edge_index, batch=batch) # degree
        out2 = self.gin2(x=x[..., :-1], edge_index=edge_index, batch=batch) # attribute
        
        out = self.ln(torch.cat([out1, out2], dim=-1))
        return out