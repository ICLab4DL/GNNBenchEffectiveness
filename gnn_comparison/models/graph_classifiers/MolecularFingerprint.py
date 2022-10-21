import torch
from torch.nn import ReLU
from torch import dropout, nn
from torch_geometric.nn import global_add_pool


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']
        dropout = config['dropout'] if 'dropout' in config else 0.6
        print(dim_features)
        print(hidden_dim)
        print(dim_target)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), ReLU(),
                                       nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_dim, dim_target), ReLU())

    def forward(self, data):
        # TODO: use graph-wise feature: g_x 
        if 'g_x' in data:
            print('using g_x:', data['g_x'][0])
            result = self.mlp(data['g_x'])
            print('result: ', result)
            return result
                
        return self.mlp(global_add_pool(data.x, data.batch))
