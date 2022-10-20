import torch
from torch.nn import ReLU
from torch_geometric.nn import global_add_pool


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']
        print(dim_features)
        print(hidden_dim)
        print(dim_target)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), ReLU(),
                                       torch.nn.Linear(hidden_dim, dim_target), ReLU())

    def forward(self, data):
        # TODO: use graph-wise feature: g_x 
        if 'g_x' in data:
            # print('using g_x, shape:', data['g_x'].shape)
            return self.mlp(data['g_x'])
                        
        return self.mlp(global_add_pool(data.x, data.batch))
