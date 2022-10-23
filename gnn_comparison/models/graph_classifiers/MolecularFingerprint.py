import torch
from torch.nn import ReLU
from torch import dropout, nn
from torch_geometric.nn import global_add_pool


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']
        dropout = config['dropout'] if 'dropout' in config else 0.4
        print('dim_features: ', dim_features)
        print('hidden_dim: ', hidden_dim)
        print('dim_target: ', dim_target)
        print('dropout:', dropout)
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), nn.Sigmoid(),
                                       nn.Dropout(dropout),
                                    #    torch.nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
                                    #    nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_dim, dim_target), nn.Sigmoid())

    def forward(self, data):
        # TODO: use graph-wise feature: g_x 
        if 'g_x' in data:
            # print('using g_x:', data['g_x'][:20],data['g_x'][20:-1] )
            result = self.mlp(data['g_x'])
            # print('result: ', result)
            return result
                
        return self.mlp(global_add_pool(data.x, data.batch))
