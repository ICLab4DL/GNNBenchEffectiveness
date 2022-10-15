import numpy as np
import networkx as nx
from collections import defaultdict

# node or edge feature generation:

from gnn_comparison.utils import utils


def xargs(f):
    def wrap(**xargs):
        return f(**xargs)
    return wrap


@xargs
def node_cycle_feature(adj, k=4):
    # TODO: make it only calculate once for the same dataset?
    
    nx_g = nx.from_numpy_array(adj)
    cycles = nx.cycle_basis(nx_g)

    # collect all len 4 sets.
    # 
    node_fea = np.zeros((adj.shape[0], 1))
    for c in cycles:
        if len(c) == k:
            for id in c:
                node_fea[id] += 1
        
    return node_fea.astype(np.float32)

@xargs
def node_tri_cycles_feature(adj, k=2):
    """ A^k as node features. so the dim of feature equals to the number of nodes.
    """

    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    adj = np.multiply(adj, np.matmul(adj, adj))
    adj = np.sum(adj, axis=1).reshape(-1, 1)
    return adj.astype(np.float32)

@xargs
def node_k_adj_feature(adj, k=2):
        
    """ A^k as node features. so the dim of feature equals to the number of nodes.
    """
    if not isinstance(k, int):
        k = int(k)
        
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    ori_adj = adj
    for _ in range(k-1):
        adj = np.matmul(adj, ori_adj)
    return adj.astype(np.float32)

@xargs
def node_degree_feature(adj):
    """ node (weighted, if its weighted adjacency matrix) degree as the node feature.
    """
    
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    return degrees.astype(np.float32)


@xargs
def node_random_id_feature(adj, ratio=1.0):
        
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()

    N = adj.shape[0]
    id_features = np.random.randint(1, int(N*ratio), size=N).reshape(N, 1).astype(np.float32)
    return id_features

@xargs
def node_allone_feature(adj):
    """return (N, 1) all one node feature
    """
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    return np.ones(N).reshape(N, 1).astype(np.float32)


@xargs
def node_gaussian_feature(adj, mean_v=0.1, std_v=1.0, dim=1):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    
    return np.random.normal(loc=mean_v, scale=std_v, size=(N, dim)).astype(np.float32)



@xargs
def node_index_feature(adj):
    """return (N, 1) node feature, feature equals to the index+1
    """
    N = adj.shape[0]
    return np.arange(1, N+1).reshape(N, 1).astype(np.float32)

@xargs
def node_deviated_feature(adj):
    N = adj.shape[0]
    block_N = int(N/2)
    fea1 = np.arange(1, block_N+1).reshape(block_N, 1).astype(np.float32)
    fea2 = 3 * np.arange(block_N+1, N+1).reshape(block_N, 1).astype(np.float32)
    return np.concatenate([fea1, fea2], axis=0)
    

# node clustering coefficient

@xargs
def node_cc_avg_feature(adj):
    N = adj.shape[0]
    g_cur = nx.from_numpy_array(adj)
    feats = nx.average_clustering(g_cur)
    return feats

@xargs
def node_cc_feature(adj):
    N = adj.shape[0]
    g_cur = nx.from_numpy_array(adj)
    feas_dict = nx.clustering(g_cur)
    feats = []
    for i in range(N):
        feats.append(feas_dict[i])
    feats = np.array(feats).reshape(N, 1).astype(np.float32)
    return feats


# TODO, d), graph feature pipeline.

def add_graph_features(graph_features, cons_fea_func, c_dim=0):
    """input:
            by default, the graph feature at dim=0 of graph_features (numpy) is the original adj matrix.
            graph_features shape: (B, N, N, C), where C is the graph feature number.
            cons_fea_func is the function to construct new graph features with NxN and append to the C-th dimension.
            
            by default, c_dim is 0, use the first adjacency matrix to construct new features. 
       return:
            graph_features, shape will be (B, N, N, C+1). 
    """
    if graph_features.ndim == 3:
        graph_features = np.expand_dims(graph_features, axis=-1)
        
    new_graph_features = []
    for ori_feature in graph_features[..., c_dim]:
        new_graph_features.append(cons_fea_func(ori_feature))
    
    new_graph_features = np.expand_dims(np.stack(new_graph_features, axis=0), axis=-1)
    
    graph_features = np.concatenate([graph_features, new_graph_features], axis=-1)
    
    return graph_features



def composite_node_feature_list(node_features:list, padding=False, padding_len=128, pad_value=0):
    """node_features: list of list, e.g., [fea1, fea2, ...], fea1:[node1, node2,...]
    """
    feas = []
    for i in range(len(node_features[0])):
        each_node = []
        for fea in node_features:
            each_node.append(fea[i])
        each_fea = np.concatenate(each_node, axis=-1)
        if padding:
            each_fea = np.pad(each_fea, ((0,0),(0, padding_len-each_fea.shape[-1])), mode='constant', constant_values=pad_value)
        feas.append(each_fea)
        
    return feas



def composite_node_features(*node_features, padding=False, padding_len=128, pad_value=0):
    """ just concatenate the new_node_features with the cur_node_features (N, C1)
        output new node features: (N, C1+C2)
    """
    if padding is None:
        padding=False
        
    if isinstance(node_features[0], list):
        res = []
        for i in range(len(node_features[0])):
            fea = np.concatenate((node_features[0][i],node_features[1][i]), axis=-1)
            if padding:
                fea = np.pad(fea, ((0,0),(0, padding_len-fea.shape[-1])), mode='constant', constant_values=pad_value)
            res.append(fea)
        return res
    
    fea = np.concatenate(node_features, axis=-1)
    if padding:
        fea = np.pad(fea, ((0,padding_len-fea.shape[-1])), mode='constant', constant_values=pad_value)
        
    return fea

def get_features_by_ids(*indices, cur_features, pad=None):
    if len(indices) < 2:
        return (cur_features[indices[0]][0], cur_features[indices[0]][1])
    
    train_fea = composite_node_features(*tuple([cur_features[i][0] for i in indices]), padding=pad)
    test_fea = composite_node_features(*tuple([cur_features[i][1] for i in indices]), padding=pad)
    return (train_fea, test_fea)


def gen_node_features(adjs, sparse, node_cons_func, **xargs):
    if sparse:
        # NOTE: the numbers of Node are different, so need sparse.
        node_features = [node_cons_func(adj=adj, **xargs) for adj in adjs]
        for i in range(len(node_features)):
            node_features[i] = utils.fill_nan_inf(node_features[i])
    else:
        node_features = np.stack([node_cons_func(adj=adj, **xargs) for adj in adjs], axis=0)
        node_features = utils.fill_nan_inf(node_features)
    
    return node_features

    
def generate_node_feature(all_data, sparse, node_cons_func, **xargs) -> tuple:
    train_adj, _, test_adj, _ = all_data
    if sparse:
        train_node_feas = [node_cons_func(adj=adj, **xargs) for adj in train_adj]
        test_node_feas = [node_cons_func(adj=adj, **xargs) for adj in test_adj]
    else:
        train_node_feas = np.stack([node_cons_func(adj=adj, **xargs) for adj in train_adj], axis=0)
        test_node_feas = np.stack([node_cons_func(adj=adj, **xargs) for adj in test_adj], axis=0)
    
    return (train_node_feas, test_node_feas) 

 # Shuffle, and then split.
# cc_train_adjs, cc_train_y, cc_test_adjs, cc_test_y

def to_dict(var_str:str):
    d = {}
    for i in var_str.split(';'):
        kv = i.split(':')
        d[kv[0]] = kv[1]
    return d


class NodeFeaRegister(object):
    def __init__(self, file_path=None):
        self.id = id(self)
        self.file_path = file_path
        if file_path is not None:
            self.funcs = {} # TODO: load from file.
            pass
        else:
            self.funcs = {
                "degree":node_degree_feature,
                "allone":node_allone_feature,
                "index_id":node_index_feature,
                "guassian":node_gaussian_feature,
                "tri_cycle":node_tri_cycles_feature,
                "cycle":node_cycle_feature,
                "kadj": node_k_adj_feature,
                "rand_id":node_random_id_feature
                }
        self.registered = []

    def register_by_str(self, arg_str:str=None):
        # arg_str format: name@key:value;key:value....
        print('argstr:', arg_str)
        args = arg_str.split("@")
        print('args:', args)
        
        if len(args)>1:
            self.register(args[0], **to_dict(args[1]))
        else:
            self.register(args[0])
        
    def contains(self, name:str) -> bool:
        for i in self.registered:
            if i[0] == name:
                return True
        return False
    
    def remove(self, re_name):
        del_id = None
        for i, (cur, _, _) in self.registered:
            if re_name == cur:
                del_id = i
                break
        if del_id is not None:
            self.registered.pop(del_id)
            print('remove func:', re_name)
        else:
            print('func name not found', re_name)
                
        
    def register(self, func_name, **xargs):
        if func_name not in self.funcs:
            print('func_name:', func_name)
            raise NotImplementedError
        
        self.registered.append((func_name, self.funcs[func_name], xargs))
    
    def get_registered(self):
        return self.registered
    
    def list_registered(self):
        for i, (name, _, arg) in enumerate(self.registered):
            print('index:', i, name, ' args: ',arg)


def register_node_features(adjs, fea_register:NodeFeaRegister):
    node_feature_list = []
    for fea_reg in fea_register.get_registered():
        node_fea = gen_node_features(adjs, sparse=True, node_cons_func=fea_reg[1], **fea_reg[2])
        node_feature_list.append(node_fea)
    return node_feature_list
    
def construct_node_features(alldata, fea_register:NodeFeaRegister):
    node_feature_list = []
    for fea_reg in fea_register.get_registered():
        node_feature_list.append(generate_node_feature(alldata, sparse=True, node_cons_func=fea_reg[1], **fea_reg[2]))
    return node_feature_list
    