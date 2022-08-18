# MLP model
import models
import torch
import torch.nn as nn

from models import BaseGraph, BaseGraphUtils



def mirror_adj(a: torch.Tensor):
    upper_tri = torch.triu(a)
    a1 = (upper_tri+upper_tri.T).fill_diagonal_(1.0)
    lower_tri = torch.tril(a)
    a2 = (lower_tri+lower_tri.T).fill_diagonal_(1.0)
    return a1, a2
    

class DirectModelAdapter(nn.Module):
    def __init__(self, model, pooling, out_dim, node_fea=None):
        super(DirectModelAdapter, self).__init__()
        self.model = model
        self.pooling = pooling
        self.node_fea = node_fea
        self.ln = nn.Linear(model.out_dim, out_dim)
    
    def forward(self, graphs:BaseGraph):
        
        node_x, adj = graphs.get_node_features(), graphs.A
        if self.node_fea is not None:
            node_x = self.node_fea
            
        # print(graphs.graph_type)
        if graphs.graph_type == 'pyg':
            dense_a = graphs.pyg_graph.edge_index.to_dense()
            dense_a1, dense_a2 = mirror_adj(dense_a)
            # TODO: dense to sparse.
            coo1 = BaseGraphUtils.dense_to_coo(dense_a1)
            coo2 = BaseGraphUtils.dense_to_coo(dense_a2)
            edge_index1 = coo1.indices()
            edge_index2 = coo2.indices()
            
        else:
            adj1 = []
            adj2 = []
            for a in adj:
                a1, a2 = mirror_adj(a)
                adj1.append(a1)
                adj2.append(a2)
                
            edge_index1 = torch.stack(adj1, dim=0)
            edge_index2 = torch.stack(adj2, dim=0)
        
        out = self.model(node_x, edge_index1, edge_index2, graphs)
        out = self.pooling(out)
        out = self.ln(out)
        return out

        
class GNNModelAdapter(nn.Module):
    def __init__(self, model, pooling, out_dim, node_fea=None, mid_fea=False):
        super(GNNModelAdapter, self).__init__()
        self.model = model
        
        self.mid_fea = mid_fea
        self.pooling = pooling
        self.node_fea = node_fea
        self.ln = nn.Linear(model.out_dim, out_dim)
        
    def forward(self, graphs:BaseGraph):
        node_x, adj = graphs.get_node_features(), graphs.A
        if self.node_fea is not None:
            node_x = self.node_fea
        mid_feature = self.model(node_x, adj, graphs)
        out = self.pooling(mid_feature) if self.pooling is not None else mid_feature
        out = self.ln(out)
        if self.mid_fea:
            return out, mid_feature
        else:
            return out

def choose_model(args, name, node_fea_dim=1, graph_fea_dim=1, class_num=6, node_num=20,
                 out_mid_fea=False, node_wise=False):
    import importlib
    
    import baseline_models.gnn_lspe.nets.OGBMOL_graph_classification.gatedgcn_net as lspe_net
    import baseline_models.gnn_lspe.layers.gatedgcn_layer as lspe_layers
    from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
    
    importlib.reload(models)
    importlib.reload(lspe_net)
    importlib.reload(lspe_layers)
    
    
    N = node_num
    if name =='mlp':
        my_model = models.ClassPredictor(N*N, 64, class_num, 3, dropout=0.6)
    elif name == 'cnn':
        my_model = models.SimpleCNN(graph_fea_dim, 64, class_num, dropout=0.6)
    elif name == 'cnn_big':
        my_model = models.SimpleCNN(graph_fea_dim, 64, class_num, dropout=0.6, kernelsize=(11, 11))
    elif name == 'gnn':
        # pool =  models.GateGraphPooling(None, N = N)
        pool = global_mean_pool
        layer_num = 2
        my_model = GNNModelAdapter(models.MultilayerGNN(layer_num, node_fea_dim, 64, 32, dropout=0.6), pool, class_num)
    elif name == 'gin':
        # pool =  models.GateGraphPooling(None, N = N)
        if node_wise:
            pool = None
        else:
            pool = models.MeanPooling()
            # pool = global_mean_pool
        layer_num = 3
        my_model = GNNModelAdapter(models.GINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6), pool, class_num, mid_fea=out_mid_fea)
    elif name == 'gin_direc':
        pool =  models.GateGraphPooling(None, N = N)
        layer_num = 3
        di_model = models.DiGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        my_model = DirectModelAdapter(di_model, pool, class_num)
    elif name == 'lsd_gin':
        pool =  models.GateGraphPooling(None, N = N)
        layer_num = 3
        di_model = models.LSDGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        my_model = DirectModelAdapter(di_model, pool, class_num)
        
    elif name == 'lspe':
        pe_init = 'lap_pe'
        pos_enc_dim = 8
        in_dim = node_fea_dim
        hid_dim = 64
        out_dim = 32
        layer_num= 3
        lspe_model = lspe_net.ReGatedGCNNet(pe_init, pos_enc_dim, in_dim,
                                       hid_dim,out_dim,layer_num=layer_num)
        
        pool =  models.GateGraphPooling(None, N = N)
        my_model = GNNModelAdapter(lspe_model, pool, class_num)
        
    my_model.cuda()
    return my_model

