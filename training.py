import utils
import models
from models import *

# import baseline_models.gnn_lspe.nets.OGBMOL_graph_classification.gatedgcn_net as lspe_net
# import baseline_models.gnn_lspe.layers.gatedgcn_layer as lspe_layers

import torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
import torch
from torch import nn
import torch.optim as optim

import os
import time
from collections import Counter, defaultdict


import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm

def training(epochs, trainer, train_evaluator, test_evaluator, train_dataloader, test_dataloader, cuda=True):
    for e in range(epochs):
        for x, y in train_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            loss, pred_y = trainer.train(x, y)
            train_evaluator.evaluate(pred_y, y, loss)
        
        for x, y in test_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
                
            loss, pred_y = trainer.eval(x, y)
            test_evaluator.evaluate(pred_y, y, loss)
        
        train_evaluator.statistic(e)
        test_evaluator.statistic(e)
        
    train_evaluator.eval_on_test()
    test_evaluator.eval_on_test()
        
      
def plot_confuse_matrix(preds, labels, save_file_path=None):
    sns.set()
    fig = plt.figure(figsize=(3, 1.5),tight_layout=True, dpi=150)
    ax = fig.gca()
    gts = [int(l) for l in labels]
    preds = [int(l) for l in preds]
    
    label_names = list(set(gts))
    label_names.sort()
    C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

    # from confusion to ACC, micro-F1, macro-F1, weighted-f1.
    print('Confusion:', C2)
    font_size = 6
    p = sns.heatmap(C2, cbar=False, annot=True, ax=ax, cmap="YlGnBu", square=False, annot_kws={"size":font_size},
        yticklabels=label_names,xticklabels=label_names)
    
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    plt.tight_layout()
    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path)

            

def plot_loss(loss, save_file_path=None):
    fig = plt.figure(figsize=(3, 1.5), tight_layout=True, dpi=150)
    plt.plot(loss)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path)

class CELossCal(nn.Module):
    def __init__(self, weights=None):
        super(CELossCal, self).__init__()
        self.crite = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, pred_y, y):
        return self.crite(pred_y, y)
 

class GraphEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, preds, labels, loss):
        raise NotImplementedError
    
    def statistic(self, epoch):
        raise NotImplementedError


class SimpleEvaluator(GraphEvaluator):
    def __init__(self, args, is_regression=False):
        super(SimpleEvaluator, self).__init__()
        self.args = args
        self.metrics = defaultdict(list)
        self.preds = []
        self.labels = []
        self.epoch_loss = []
        self.mean_metrics = defaultdict(list)
        self.total_metrics = {}
        self.is_regression = is_regression
        
    def evaluate(self, preds, labels, loss, null_val=0.0):
        
        if self.is_regression:
            preds_b = preds
            if np.isnan(null_val):
                mask = ~torch.isnan(labels)
            else:
                mask = (labels != null_val)
                mask = mask.float()
                mask /= torch.mean(mask)
                # handle all zeros.
                mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
                mse = (preds - labels) ** 2
                mae = loss
                mape = mae / labels
                mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
                rmse = torch.sqrt(mse)
                
                self.metrics['mae'].append(mae.item())
                self.metrics['mape'].append(mape.item())
                self.metrics['rmse'].append(rmse.item())
                self.metrics['loss'].append(loss)
            
        else:
            num = preds.size(0)
            # print('evl preds shape:', preds.shape, labels.shape)
            preds_b = preds.argmax(dim=1).squeeze()
            labels = labels.squeeze()
            ones = torch.zeros(num)
            ones[preds_b == labels] = 1
            acc = torch.sum(ones) / num
                
            mi_f1, ma_f1, weighted_f1 = utils.cal_f1(preds_b.cpu().detach().numpy(), labels.cpu().detach().numpy())
            self.metrics['micro_f1'].append(mi_f1)
            self.metrics['macro_f1'].append(ma_f1)
            self.metrics['weighted_f1'].append(weighted_f1)
            self.metrics['acc'].append(acc.numpy())
            self.metrics['loss'].append(loss)
        
        self.preds.append(preds_b)
        self.labels.append(labels)
        
        
    def statistic(self, epoch):
        for k, v in self.metrics.items():
            self.mean_metrics[k].append(np.mean(v))
            
        self.metrics = defaultdict(list)
            
    def eval_on_test(self):
        if self.is_regression:
            preds = torch.cat(self.preds, dim=0)
            labels = torch.cat(self.labels, dim=0)
            mask = (labels != 0.0)
            mask = mask.float()
            mask /= torch.mean(mask)
            # handle all zeros.
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            mse = (preds - labels) ** 2
            mae = torch.abs(preds - labels)
            mape = mae / labels
            mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
            rmse = torch.sqrt(mse)
            
            self.metrics['mae'].append(mae)
            self.metrics['mape'].append(mape)
            self.metrics['rmse'].append(rmse)
            
        else:
            preds = torch.cat(self.preds, dim=0)
            labels = torch.cat(self.labels, dim=0)
            mi_f1, ma_f1, weighted_f1 = utils.cal_f1(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
            self.total_metrics['micro_f1'] = mi_f1
            self.total_metrics['macro_f1'] = ma_f1
            self.total_metrics['weighted_f1'] = weighted_f1
    
    def print_info(self):
        micro_f1 = self.metrics[-1]['micro_f1']
        macro_f1 = self.metrics[-1]['macro_f1']
        weighted_f1 = self.metrics[-1]['weighted_f1']
        acc = self.metrics[-1]['acc']
        loss =  self.metrics[-1]['loss']
        print('------------- metrics -------------------')
        print(f'micro f1: {"%.3f" % micro_f1}, macro f1:  {"%.3f" % macro_f1},weighted f1:  {"%.3f" % weighted_f1},\n \
            acc:  {"%.3f" % acc}, loss: {"%.3f" % loss}')
        
    
        
    def plot_metrics(self):
        
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)
        
        if not self.is_regression:
            plot_confuse_matrix(preds, labels)
        
        loss_list = self.mean_metrics['loss']
        
        plot_loss(loss_list)
        
        

def mirror_adj(a: torch.Tensor):
    upper_tri = torch.triu(a)
    a1 = (upper_tri+upper_tri.T).fill_diagonal_(1.0)
    lower_tri = torch.tril(a)
    a2 = (lower_tri+lower_tri.T).fill_diagonal_(1.0)
    return a1, a2
    

class DirectModelAdapter(nn.Module):
    def __init__(self, model, pooling, out_dim, node_fea=None, device='cuda:0'):
        super(DirectModelAdapter, self).__init__()
        self.device = device
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
            dense_a = models.BaseGraphUtils.edge_index_to_coo(graphs.pyg_graph.edge_index, graphs.pyg_graph.num_nodes, device=self.device).to(self.device).to_dense()
            dense_a1, dense_a2 = mirror_adj(dense_a)
            # TODO: dense to sparse.
            coo1 = models.BaseGraphUtils.dense_to_coo(dense_a1, self.device)
            coo2 = models.BaseGraphUtils.dense_to_coo(dense_a2, self.device)
            edge_index1 = coo1.coalesce().indices().to(self.device)
            edge_index2 = coo2.coalesce().indices().to(self.device)
            
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
        out = self.pooling(out, graphs.pyg_graph.batch)
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
            
        if graphs.graph_type == 'pyg' or graphs.graph_type == 'coo':
            edge_indices = graphs.get_edge_index()
            mid_feature = self.model(node_x, edge_indices, graphs=graphs)
        else:
            mid_feature = self.model(node_x, adj, graphs)
        
        print('mid device:', mid_feature.device)
        print('pyg_graph device:', graphs.pyg_graph.batch.device)
        out = self.pooling(mid_feature, graphs.pyg_graph.batch) if self.pooling is not None else mid_feature
        out = self.ln(out)
        if self.mid_fea:
            return out, mid_feature
        else:
            return out

    
    

def choose_model(args, name, node_fea_dim=1, graph_fea_dim=1, class_num=6, node_num=20,
                 out_mid_fea=False, node_wise=False):
    import importlib
    
    # import baseline_models.gnn_lspe.nets.OGBMOL_graph_classification.gatedgcn_net as lspe_net
    # import baseline_models.gnn_lspe.layers.gatedgcn_layer as lspe_layers

    import baseline_models.gnn_baselines
    from baseline_models import identity_GNN


    from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
    
    importlib.reload(models)
    importlib.reload(identity_GNN)
    
    
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
    elif name == 'sparse_gin':
        if node_wise:
            pool = None
        else:
            pool = global_mean_pool
            # pool = global_mean_pool
        layer_num = 3
        my_model = GNNModelAdapter(models.LSDGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6, bi=False), pool, class_num, mid_fea=out_mid_fea)
    elif name == 'gin_direc':
        pool =  models.GateGraphPooling(None, N = N)
        layer_num = 3
        di_model = models.DiGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        my_model = DirectModelAdapter(di_model, pool, class_num)
    elif name == 'lsd_gin':
        pool = global_mean_pool
        pool = global_add_pool
        layer_num = 3
        di_model = models.LSDGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        my_model = DirectModelAdapter(di_model, pool, class_num)
    
    elif name == 'idgnn':
        pool = global_add_pool
        layer_num = 3
        idgnn_model = identity_GNN.IDGNN(args, node_fea_dim, 64, 32, layer_num)
        my_model = GNNModelAdapter(idgnn_model, pool, class_num, mid_fea=out_mid_fea)
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
    else:
        raise NotImplementedError
    
        
    my_model.cuda()
    return my_model


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MAECal(nn.Module):
    def __init__(self):
        super(MAECal, self).__init__()
        pass
    
    def forward(self, pred_y, labels, null_val=0.0):
        
        labels = labels.squeeze()
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        mae = torch.abs(pred_y - labels)
        mae = mask_and_fillna(mae, mask)
        return mae

class CELossCal(nn.Module):
    def __init__(self, weights=None):
        super(CELossCal, self).__init__()
        self.crite = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, pred_y, y):
        return self.crite(pred_y, y)
 
 
 

def train_gnn(args, train_loader, test_loader, gnn_name='gnn', epoch=200, plot=False, node_fea_dim=1,
              class_num=6, node_num=20, **xargs):
        
    def get_value(key, default=None):
        return xargs[key] if key in xargs else default
    
    # use GNN to trian:
    args.debug = False
    utils.DLog.init(args)
    
    is_node_wise = get_value('is_node_wise', False)
    is_regression = get_value('is_regression', False)
    scaler = get_value('scaler')
    opt = get_value('opt', 'sgd')
    lr = xargs['lr'] if 'lr' in xargs else 0.0002
    
    
    
    gnn_model = choose_model(args, gnn_name, node_fea_dim=node_fea_dim, class_num=class_num,
                             node_num=node_num,
                             node_wise=is_node_wise)
    if opt == 'adam':
        opt = optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(gnn_model.parameters(), lr=lr)
        
    
    loss_cal = get_value('loss_cal', None)
    if loss_cal is None:
        loss_cal =  MAECal() if is_regression else CELossCal()
    elif loss_cal == 'mse':
        loss_cal = nn.MSELoss()
    elif loss_cal == 'mae':
        loss_cal = MAECal()
        

    trainer = utils.Trainer(gnn_model, optimizer=opt, loss_cal=loss_cal, scaler=scaler)
    train_sim_evl= SimpleEvaluator(args, is_regression=is_regression)
    test_sim_evl= SimpleEvaluator(args,is_regression=is_regression)

    training(epoch, trainer, train_sim_evl, test_sim_evl, train_loader, test_loader)

    if plot:
        train_sim_evl.plot_metrics()
        test_sim_evl.plot_metrics()
        
    return train_sim_evl, test_sim_evl, gnn_model