import numpy as np
import networkx as nx

from functools import reduce
import random
import utils
# from scipy.sparse import coo_matrix


def connect_graphs(g1, g2):
    n1 = list(g1.nodes)
    n2 = list(g2.nodes)
    e1 = random.choices(n1, k=1)[0]
    e2 = random.choices(n2, k=1)[0]
    g_cur = nx.compose(g1, g2)
    g_cur.add_edge(e1, e2)
    return g_cur

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
        
    g_all = reduce(connect_graphs, new_graphs)
    
    return g_all


def generate_CSL(each_class_num:int, N:int, S:list):
    samples = []
    for s in S:
        g = nx.circulant_graph(N, [1, s])
        # pers = set()
        # if each_class_num < math.factorial(N):
        #     uniq_per = set()
        #     while len(uniq_per) < each_class_num:
        #         per = np.random.permutation(list(np.arange(0, 8)))
        #         pers.add()
        # else:
        pers = [np.random.permutation(list(np.arange(0, N))) for _ in range(each_class_num)]

        A_g = nx.to_scipy_sparse_matrix(g).todense()
        # Permutate:
        for per in pers:
            A_per = A_g[per, :]
            A_per = A_per[:, per]
            # NOTE: set cycle length or skip length as label.
            samples.append((nx.from_numpy_array(A_per), s))
    
    return samples


def generate_training_graphs(graphs_cc):

    np.random.shuffle(graphs_cc)
    
    test_sample_size = int(len(graphs_cc)/3)
    train_adjs, train_y, test_adjs, test_y = [],[],[],[]
    
    for g in graphs_cc[:-test_sample_size]:
        # TODO: generate some deviation:
        if isinstance(g, tuple):
            adj, y = g
            train_adjs.append(adj)
            train_y.append(y)
        else:
            train_adjs.append(g)
            train_y.append(nx.average_clustering(g))
        
    for g in graphs_cc[-test_sample_size:]:
        if isinstance(g, tuple):
            adj, y = g
            test_adjs.append(adj)
            test_y.append(y)
        else:
            test_adjs.append(g)
            test_y.append(nx.average_clustering(g))
            
    train_adjs = [nx.to_scipy_sparse_matrix(g) for g in train_adjs]
    test_adjs = [nx.to_scipy_sparse_matrix(g) for g in test_adjs]


    train_y = np.stack(train_y, axis=0)
    test_y = np.stack(test_y, axis=0)
    
    return (train_adjs, train_y, test_adjs, test_y)
    

def generate_cc_no_degree_corr_samples(cc_range_num=20, rand_con=True):
    
    def random_add_edges(graph, E=3):
        nodes = list(graph.nodes)
        for i in range(E):
            e = random.sample(nodes, k=2)
            graph.add_edge(*e)
        return graph
        
    cc_range_num = cc_range_num
    graphs_cc = []
    for k in range(1, cc_range_num):
        m = cc_range_num - k
        G_tri = [nx.complete_graph(3) for _ in range(k)]
        G_sqr = [nx.cycle_graph(4) for _ in range(m)]
        cur_graphs = [random_connect_graph(utils.flatten_list([G_tri, G_sqr])) for _ in range(5)]
        # repeat for 5 times:
        for _ in range(5):
            [graphs_cc.append(random_add_edges(g, E=3)) for g in cur_graphs]        

    return generate_training_graphs(graphs_cc)
    
def get_value(xargs, key, default=None):
    """return (N, 1) all one node feature
    """
    return xargs[key] if key in xargs else default


def numerical_to_categorical(num_list):
    cates = {}
    idx = -1 # from 0 to maximal unique label index.
    labels = []
    for n in num_list:
        if isinstance(n, np.ndarray):
            n = n.item()
        if n not in cates:
            idx += 1
            cates[n] = idx
        labels.append((n, cates[n]))
    labels = [l[-1] for l in labels]
    
    return labels


