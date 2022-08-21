import utils
from models import *
import models
import dataset_loader
import training
from torch.utils.data import DataLoader

from functools import reduce
import os
import time
import random

# plot f1 curves:

def get_f1s(evls):
    mi_f1 = []
    ma_f1 = []
    w_f1 = []

    for evl in evls:
        mi_f1.append(evl.total_metrics['micro_f1'])
        ma_f1.append(evl.total_metrics['macro_f1'])
        w_f1.append(evl.total_metrics['weighted_f1'])
    return mi_f1, ma_f1, w_f1

    
    
def plot_f1_curves(mi_f1, ma_f1, w_f1):
    plt.figure(figsize=(4, 3), dpi=150)


    x = np.linspace(0, 1, 24)
    plt.plot(x, mi_f1,  marker="8")
    plt.plot(x, ma_f1,  marker=11)
    ax = plt.axes()
  
# Setting the background color of the plot 
# using set_facecolor() method
    ax.set_facecolor("snow")
    
    plt.grid()
    plt.show()
    plt.savefig('test_lspe')




def construct_dataset(graph_data, node_features, norm=True, lap_encode=False, \
                      lap_en_dim=8, y_torch_type=torch.LongTensor, sparse=False):
    
    
    train_adjs, train_y, test_adjs, test_y = graph_data
    # construct node features:
    train_node_fea, val_node_fea = node_features
    if norm:
        if sparse:
            mean_y = np.mean(np.concatenate(train_node_fea, axis=0))
            std_y = np.std(np.concatenate(train_node_fea, axis=0))
            scaler = utils.StandardScaler(mean=mean_y, std=std_y)
            train_node_fea = scaler.transform(train_node_fea)
            val_node_fea =  utils.normalize(val_node_fea)
        else:
            mean_y = train_node_fea.mean()
            std_y = train_node_fea.std()
            scaler = utils.StandardScaler(mean=mean_y, std=std_y)
            train_node_fea =  scaler.transform(train_node_fea)
            val_node_fea =  utils.normalize(val_node_fea)
        
    else:
        scaler = None

    train_base_graphs = []
    if sparse:
        for i, adj in enumerate(train_adjs):
            g = models.BaseGraphUtils.from_scipy_coo(adj.tocoo())
            g.set_node_feat(train_node_fea[i])
            train_base_graphs.append(g)

            test_base_graphs = []
            for i, adj in enumerate(test_adjs):
                g = models.BaseGraphUtils.from_scipy_coo(adj.tocoo())
                g.set_node_feat(val_node_fea[i])
                test_base_graphs.append(g)
    else:
        for i, adj in enumerate(train_adjs):
            g = models.BaseGraphUtils.from_numpy(adj)
            g.set_node_feat(train_node_fea[i])
            train_base_graphs.append(g)

        test_base_graphs = []
        for i, adj in enumerate(test_adjs):
            g = models.BaseGraphUtils.from_numpy(adj)
            g.set_node_feat(val_node_fea[i])
            test_base_graphs.append(g)
    
    
    train_dataset = GraphDataset(x=train_base_graphs, y=y_torch_type(train_y))
    test_dataset = GraphDataset(x=test_base_graphs, y=y_torch_type(test_y))

    if lap_encode:
        train_dataset._add_lap_positional_encodings(lap_en_dim)
        test_dataset._add_lap_positional_encodings(lap_en_dim)
    
    return train_dataset, test_dataset, scaler


def assemble_dataloader(train_dataset: GraphDataset, test_dataset: GraphDataset, scaler=None, cuda=False, batch_size=20):
    if cuda:
        train_dataset.cuda()
        test_dataset.cuda()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate)

    def print_tuple(x):
        if isinstance(x, list) or isinstance(x, tuple):
            adj = x[1]
            x = x[0]
        
    # for x, y in train_dataloader:
    #     print(type(x))
    #     print_tuple(x)
    #     print(y.is_cuda)
    #     break
    
    return (train_dataloader, test_dataloader)

    
def generate_node_feature(all_data, node_cons_func, sparse=False):
    train_node_feas = [] # train, val
    test_node_feas = [] # train, val
    for train_adj ,_, test_adj, _ in all_data:
        if sparse:
            train_node_feas.append( [node_cons_func(adj) for adj in train_adj])
            test_node_feas.append( [node_cons_func(adj) for adj in test_adj])
        else:
            train_node_feas.append(np.stack([node_cons_func(adj) for adj in train_adj], axis=0))
            test_node_feas.append(np.stack([node_cons_func(adj) for adj in test_adj], axis=0))
        
    return train_node_feas, test_node_feas    


def generate_cc_no_degree_corr_samples(cc_range_num=20):
        
    def random_connect_graph(graph_list:list):
        # NOTE: relabeling the nodes.
        new_graphs = []
        np.random.shuffle(graph_list)
        node_idx = 0
        for g in graph_list:
            len_nodes = len(list(g.nodes))
            mapping = {}
            for i in range(len_nodes):
                mapping[i] = i+node_idx
            new_g = nx.relabel_nodes(g, mapping)
            new_graphs.append(new_g)
            node_idx += len_nodes
            
        def connect_graphs(g1, g2):
            n1 = list(g1.nodes)
            n2 = list(g2.nodes)
            e1 = random.choices(n1, k=1)[0]
            e2 = random.choices(n2, k=1)[0]
            g_cur = nx.compose(g1, g2)
            g_cur.add_edge(e1, e2)
            return g_cur
        
        g_all = reduce(connect_graphs, new_graphs)
        
        return g_all
        
        
    def random_add_edges(graph, E=3):
        nodes = list(graph.nodes)
        for i in range(E):
            e = random.sample(nodes, k=2)
            graph.add_edge(*e)
        return graph
        
    cc_range_num = 20
    graphs_cc = []
    for k in range(1, cc_range_num):
        m = cc_range_num - k
        G_tri = [nx.complete_graph(3) for _ in range(k)]
        G_sqr = [nx.cycle_graph(4) for _ in range(m)]
        cur_graphs = [random_connect_graph(utils.flatten_list([G_tri, G_sqr])) for _ in range(5)]
        # repeat for 5 times:
        for _ in range(5):
            [graphs_cc.append(random_add_edges(g, E=3)) for g in cur_graphs]        

    np.random.shuffle(graphs_cc)
    
    
    test_sample_size = int(len(graphs_cc)/3)
    train_adjs, train_y, test_adjs, test_y = [],[],[],[]
    
    for ws in graphs_cc[:-test_sample_size]:
        # TODO: generate some deviation:
        train_adjs.append(ws)
        train_y.append(nx.average_clustering(ws))
        
    for ws in graphs_cc[-test_sample_size:]:
        test_adjs.append(ws)
        test_y.append(nx.average_clustering(ws))
            
    train_adjs = [nx.to_scipy_sparse_matrix(g) for g in train_adjs]
    test_adjs = [nx.to_scipy_sparse_matrix(g) for g in test_adjs]

    train_y = np.stack(train_y, axis=0)
    test_y = np.stack(test_y, axis=0)
    
    return (train_adjs, train_y, test_adjs, test_y)
    
    
    
    
def load_cc_degree_free_dataloaders():
        # Shuffle, and then split.
    # cc_train_adjs, cc_train_y, cc_test_adjs, cc_test_y

    import networkx as nx


    cc_train_adjs, cc_train_y, cc_test_adjs, cc_test_y = generate_cc_no_degree_corr_samples(cc_range_num=20)
    # random add edges:
    # add E edges, repeat for 5 times.
    print(type(cc_train_adjs[0].todense()))

    data_graphs = [(cc_train_y[i],np.mean(np.sum(cc_train_adjs[i].todense(), axis=1))) for i in range(len(cc_train_adjs))]

    data_graphs_s = sorted(data_graphs, key=lambda x: x[0])

    ccs = [d[0] for d in data_graphs_s]
    degrees = [d[1]/10 for d in data_graphs_s]

    plt.figure()
    plt.plot(ccs, label='cc')
    plt.plot(degrees, label='degree')
    plt.legend()
    plt.show()

    data_graphs = [(cc_test_y[i],np.mean(np.sum(cc_test_adjs[i].todense(), axis=1))) for i in range(len(cc_test_adjs))]

    data_graphs_s = sorted(data_graphs, key=lambda x: x[0])

    ccs_test = [d[0] for d in data_graphs_s]
    degrees_test = [d[1]/10 for d in data_graphs_s]

    plt.figure()
    plt.plot(ccs_test, label='cc')
    plt.plot(degrees_test, label='degree')
    plt.legend()
    plt.show()
    


    cc_degree_avg_labels_train,  cc_degree_avg_labels_test =  generate_node_feature([(cc_train_adjs, None, cc_test_adjs, None)] ,node_degree_feature,
                                                                                    sparse=True)


    cc_degree_mean_train = np.stack([np.mean(cc_deg) for cc_deg in cc_degree_avg_labels_train[0]]).reshape(-1, 1)
    cc_degree_mean_test = np.stack([np.mean(cc_deg) for cc_deg in cc_degree_avg_labels_test[0]]).reshape(-1, 1)

    cc_dataset = (cc_train_adjs, cc_train_y, cc_test_adjs, cc_test_y)

    allone_train,  allone_test=  generate_node_feature([(cc_train_adjs, None, cc_test_adjs, None)], node_allone_feature,
                                                    sparse=True)

    # # # add Gaussion noise:

    allone_train = [0.2 * allone_train[0][i] + np.random.random(size=allone_train[0][i].shape) for i in range(len(allone_train[0]))]
    allone_test = [0.2 * allone_test[0][i] + np.random.random(size=allone_test[0][i].shape) for i in range(len(allone_test[0]))]


    cc_dataloaders = []
    cc_dataloaders.append(assemble_dataloader(
        *construct_dataset(cc_dataset,
                            (allone_train, allone_test), y_torch_type=torch.FloatTensor, sparse=True), cuda=False))
    
    return cc_dataloaders[0]



def node_degree_feature(adj):
    """ node (weighted, if its weighted adjacency matrix) degree as the node feature.
    """
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    return degrees.astype(np.float32)

def node_allone_feature(adj):
    """return (N, 1) all one node feature
    """
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    return np.ones(N).reshape(N, 1).astype(np.float32)

def node_index_feature(adj):
    """return (N, 1) node feature, feature equals to the index+1
    """
    N = adj.shape[0]
    return np.arange(1, N+1).reshape(N, 1).astype(np.float32)

def node_deviated_feature(adj):
    N = adj.shape[0]
    block_N = int(N/2)
    fea1 = np.arange(1, block_N+1).reshape(block_N, 1).astype(np.float32)
    fea2 = 3 * np.arange(block_N+1, N+1).reshape(block_N, 1).astype(np.float32)
    return np.concatenate([fea1, fea2], axis=0)
    

# node clustering coefficient


def node_cc_avg_feature(adj):
    N = adj.shape[0]
    g_cur = nx.from_numpy_array(adj)
    feats = nx.average_clustering(g_cur)
    return feats

def node_cc_feature(adj):
    N = adj.shape[0]
    g_cur = nx.from_numpy_array(adj)
    feas_dict = nx.clustering(g_cur)
    feats = []
    for i in range(N):
        feats.append(feas_dict[i])
    feats = np.array(feats).reshape(N, 1).astype(np.float32)
    return feats


    
if __name__ == '__main__':

    args = utils.get_common_args()

    args = args.parse_args()
    args.cuda = True
    args.batch_size = 32
    args.tag = 'unit_test'
    args.fig_filename= f'/li_zhengdao/github/GenerativeGNN/results/{args.tag}'
    
    device = 'cuda:0' if args.cuda else 'cpu'
    
    # args.pos_en = 'lap_pe'
    # args.pos_en_dim = 8


    train_dataloader, test_dataloader = load_cc_degree_free_dataloaders()
    
    # train_dataloader, test_dataloader = dataset_loader.load_data('AIDS', args)
    args.class_num = 2
    
    # get feature dimension.
    x, _ = train_dataloader.dataset.__getitem__(0)
    print('node feature shape:', x.get_node_features().shape)
    node_feature_dim = x.get_node_features().shape[-1]
    
    
    gnn_model_task12 = []
    cc_gnn_evls_train = []
    cc_gnn_evls_test = []
    train_evl, test_evl, cur_model = training.train_gnn(args, train_dataloader, test_dataloader, gnn_name='idgnn',
                                    epoch=1,
                                    node_fea_dim=1,
                                    class_num=1,
                                    node_num=40, lr=0.0001, is_regression=True, is_node_wise=False)
    
    gnn_model_task12.append(cur_model)
    cc_gnn_evls_train.append(train_evl)
    cc_gnn_evls_test.append(test_evl)
    
        
    mi_f1, ma_f1, w_f1 = get_f1s(cc_gnn_evls_test)

    print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    fig_save_path = os.path.join(fig_save_dir, f'{file_tag}')
    test_evl.plot_metrics(utils.append_tag(fig_save_path, 'test_idgnn'))
    train_evl.plot_metrics(utils.append_tag(fig_save_path, 'train_idgnn'))
    
