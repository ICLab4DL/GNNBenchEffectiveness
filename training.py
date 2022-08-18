import utils
import models
from models import *

import baseline_models.gnn_lspe.nets.OGBMOL_graph_classification.gatedgcn_net as lspe_net
import baseline_models.gnn_lspe.layers.gatedgcn_layer as lspe_layers

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
    def __init__(self, args):
        super(SimpleEvaluator, self).__init__()
        self.args = args
        self.metrics = defaultdict(list)
        self.preds = []
        self.labels = []
        self.epoch_loss = []
        self.mean_metrics = defaultdict(list)
        self.total_metrics = {}
        
    def evaluate(self, preds, labels, loss):
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
        
    
        
    def plot_metrics(self, fig_save_path='./figs/'):
    
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)
        
        plot_confuse_matrix(preds, labels, utils.append_tag(fig_save_path, 'confusion.png'))
        
        loss_list = self.mean_metrics['loss']
        correct = int((preds == labels).sum())  # Check against ground-truth labels.
        acc = correct / labels.shape[0]  # Derive ratio of correct predictions.
        print('Acc: ', acc)
        
        plot_loss(loss_list, save_file_path=utils.append_tag(fig_save_path, 'loss.png'))
        

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
    def __init__(self, model, pooling, out_dim, node_fea=None):
        super(GNNModelAdapter, self).__init__()
        self.model = model
        self.pooling = pooling
        self.node_fea = node_fea
        self.ln = nn.Linear(model.out_dim, out_dim)
        
    def forward(self, graphs:models.BaseGraph):
        node_x, adj = graphs.get_node_features(), graphs.A
        if self.node_fea is not None:
            node_x = self.node_fea
            
        if graphs.graph_type == 'pyg':
            edge_index = graphs.pyg_graph.edge_index
        else:
            edge_index = adj
            
        out = self.model(node_x, edge_index, graphs)
        out = self.pooling(out, graphs.pyg_graph.batch)
        out = self.ln(out)
        return out
    
    

def choose_model(args, name, node_fea_dim=1, pooling='gate', graph_fea_dim=1, class_num=6, N=-1):
    
    if name =='mlp':
        my_model = models.ClassPredictor(64, class_num, 3, dropout=0.6)
    elif name == 'gcn':
        # pool = global_mean_pool
        my_model = models.GCNO(node_fea_dim, 64, class_num)
        # my_model = GNNModelAdapter(models.GCNO(node_fea_dim, 64), pool, class_num)
        
    elif name == 'cnn':
        my_model = models.SimpleCNN(graph_fea_dim, 64, class_num, dropout=0.6)
    elif name == 'cnn_big':
        my_model = models.SimpleCNN(graph_fea_dim, 64, class_num, dropout=0.6, kernelsize=(11, 11))
    elif name == 'gnn':
        if pooling == 'gate':
            pool =  models.GateGraphPooling(None, N = N)
        elif pooling == 'mean':
            pool = global_mean_pool
        layer_num = 3
        my_model = GNNModelAdapter(models.MultilayerGNN(layer_num, node_fea_dim, 64, 32, dropout=0.6), pool, class_num)
    elif name == 'gin':
        pool =  models.GateGraphPooling(None, N = N)
        layer_num = 3
        my_model = GNNModelAdapter(models.GINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6), pool, class_num)
    elif name == 'gin_direc':
        pool =  models.GateGraphPooling(None, N = N)
        layer_num = 3
        di_model = models.DiGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        my_model = DirectModelAdapter(di_model, pool, class_num)
    elif name == 'lsd_gin':
        layer_num = 3
        di_model = models.LSDGINNet(args, node_fea_dim, 64, 32, 3, dropout=0.6)
        # pool =  models.LinearGraphPooling(32, 32)
        pool = global_mean_pool
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

def train_svm(train_loader, test_loader):
    train_x = train_loader.dataset.x
    
    datasets = dict()
    print('len:',len(xs))
    for i in range(len(xs)):
        datasets[f'{cates[i]}_x'] = xs[i].reshape(xs[i].shape[0],-1)
        datasets[f'{cates[i]}_y'] = ys[i]
        print(f'{cates[i]}_y: shape',datasets[f'{cates[i]}_y'].shape)
        print(f'{cates[i]}_x: shape',datasets[f'{cates[i]}_x'].shape)

    c = Counter(datasets['train_y'])

    if os.path.exists('svm_clf.pkl'):
        clf = joblib.load('svm_clf.pkl')
        print('loaded svm_clf!!!!!!!')
    else:
        clf = svm.SVC()
        clf.fit(datasets['train_x'], datasets['train_y'])
        joblib.dump(clf, 'svm_clf.pkl')
    # mi_f1 = f1_score(datasets['test_y'], preds, average='micro')
    # ma_f1 = f1_score(datasets['test_y'], preds, average='macro')
    # weighted_f1 = f1_score(datasets['test_y'], preds, average='weighted')
    preds = clf.decision_function(test_x)
    return preds


def train_gnn(args, train_loader, test_loader, gnn_name='gnn', pooling='gate', epoch=10, plot=False, node_fea_dim=1):
    # use GNN to trian:
    gnn_model = choose_model(args, gnn_name, node_fea_dim=node_fea_dim, pooling=pooling, class_num=args.class_num)
    if args.cuda:
        gnn_model.cuda()
    # opt = optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # opt = optim.SGD(gnn_model.parameters(), lr=0.001)
    opt = torch.optim.Adam(gnn_model.parameters(), lr=0.005)
    ce_loss_cal = CELossCal()

    trainer = utils.Trainer(gnn_model, optimizer=opt, loss_cal=ce_loss_cal)
    
    train_sim_evl= SimpleEvaluator(args)
    test_sim_evl= SimpleEvaluator(args)

    training(epoch, trainer, train_sim_evl, test_sim_evl, train_loader, test_loader)

    if plot:
        basedir, file_tag = os.path.split(args.fig_filename)
        date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
        fig_save_dir = os.path.join(basedir, date_dir)
        if not os.path.exists(fig_save_dir):
            os.mkdirs(fig_save_dir)
        fig_save_path = os.path.join(fig_save_dir, f'_{file_tag}_')
        
        train_sim_evl.plot_metrics(fig_save_path = fig_save_path)
        test_sim_evl.plot_metrics(fig_save_path = fig_save_path)
        
    return train_sim_evl, test_sim_evl
