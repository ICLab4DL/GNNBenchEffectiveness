import my_utils
from dataset_utils import node_feature_utils
from datasets.tu_utils import parse_tu_data, create_graph_from_tu_data, get_dataset_node_num, create_graph_from_nx
from datasets.sampler import RandomSampler
from datasets.dataset import GraphDataset, GraphDatasetSubset
from datasets.dataloader import DataLoader
from datasets.data import Data
from datasets.synthetic_dataset_generator import *
from utils.encode_utils import NumpyEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.nn import functional as F
import torch
import numpy as np
from collections import defaultdict
import pickle as pk
from numpy import linalg as LA
import torch.nn as nn
from networkx import normalized_laplacian_matrix
import networkx as nx
from pathlib import Path
import zipfile
import requests
import json
import os
import io
import sys
import os
sys.path.append(os.getcwd())


# import k_gnn

class EmptyNodeFeatureException(Exception):
    def __init__(self) -> None:
        super().__init__("There is no node feature as input!!!")

class GraphDatasetManager:
    def __init__(self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None, seed=42, holdout_test_size=0.1,
                 use_node_degree=False, use_node_attrs=False, use_one=False, use_shared=False, use_1hot=False,
                 use_random_normal=False, use_pagerank=False, use_eigen=False, use_eigen_norm=False,
                 use_deepwalk=False, precompute_kron_indices=False, additional_features: str = None, additional_graph_features: str = None,
                 max_reductions=10, DATA_DIR='./DATA', config={}):

        self.root_dir = Path(DATA_DIR) / self.name
        self.kfold_class = kfold_class
        self.holdout_test_size = holdout_test_size
        self.config = config
        if additional_features is not None:
            add_features = additional_features.strip().split(',')
            self.use_1hot = True if 'use_onehot' in add_features else use_1hot
            self.use_random_normal = True if 'use_random' in add_features else use_random_normal
            self.use_pagerank = True if 'use_pagerank' in add_features else use_pagerank
            self.use_eigen = True if 'use_eigen' in add_features else use_eigen
            self.use_deepwalk = True if 'use_deepwalk' in add_features else use_deepwalk
        else:
            self.use_1hot = use_1hot
            self.use_random_normal = use_random_normal
            self.use_pagerank = use_pagerank
            self.use_eigen = use_eigen
            self.use_deepwalk = use_deepwalk
                        
        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one
        self.use_shared = use_shared
        self.use_eigen_norm = use_eigen_norm
        self.precompute_kron_indices = precompute_kron_indices
        # will compute indices for 10 pooling layers --> approximately 1000 nodes
        self.KRON_REDUCTIONS = max_reductions
        # 2022.10.02
        self.additional_features = additional_features
        # 2022.10.20
        self.additional_graph_features = additional_graph_features

        self.Graph_whole = None
        self.Graph_whole_pagerank = None
        self.Graph_whole_eigen = None

        self.outer_k = outer_k
        assert (outer_k is not None and outer_k > 0) or outer_k is None

        self.inner_k = inner_k
        assert (inner_k is not None and inner_k > 0) or inner_k is None

        self.seed = seed

        self.raw_dir = self.root_dir / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._download()

        self.processed_dir = self.root_dir / "processed"
        print('processed_dir: ', self.processed_dir)
        if not (self.processed_dir / f"{self.name}.pt").exists():
            if not self.processed_dir.exists():
                os.makedirs(self.processed_dir)
            self._process()

        print('load dataset !')
        self.dataset = GraphDataset(torch.load(
            self.processed_dir / f"{self.name}.pt"))
        print('dataset len: ', len(self.dataset))
        splits_filename = self.processed_dir / f"{self.name}_splits.json"
        if not splits_filename.exists():
            self.splits = []
            self._make_splits()
        else:
            self.splits = json.load(open(splits_filename, "r"))
            print('load splits:', splits_filename)
        print('split counts:', len(self.splits))
        # TODO: if add more node features:
        if self.additional_features is not None:
            # TODO: pass node function?
            # node register
            self._add_features()

        if self.additional_graph_features is not None:
            self._add_graph_features()

    @property
    def init_method(self):
        if self.use_random_normal:
            return "random_nomral"

    @property
    def num_graphs(self):
        return len(self.dataset)

    @property
    def dim_target(self):
        if not hasattr(self, "_dim_target") or self._dim_target is None:
            # not very efficient, but it works
            # todo not general enough, we may just remove it
            self._dim_target = np.unique(self.dataset.get_targets()).size
        return self._dim_target

    @property
    def dim_features(self):
        # TODO: check the graph level features:
        if self.additional_graph_features is not None:
            self._dim_features = self.dataset.data[0].g_x.shape[-1]
        else:
            self._dim_features = self.dataset.data[0].x.size(1)
        print('input feature dimension: ', self._dim_features)

        # best for feature initialization based on the current implementation
        # if not hasattr(self, "_dim_features") or self._dim_features is None:
        # not very elegant, but it works
        # todo not general enough, we may just remove it
        # self._dim_features = self.dataset.data[0].x.size(1)
        # feature initialization
        return self._dim_features

    def _add_graph_features(self):
        # TODO: add graph-wise features:
        self.additional_graph_features = self.additional_graph_features.strip().split(',')
        graph_fea_reg = node_feature_utils.GraphFeaRegister()
        for feature_arg in self.additional_graph_features:
            graph_fea_reg.register_by_str(feature_arg)
        self.graph_fea_reg = graph_fea_reg

        adjs = [d.to_numpy_array() for d in self.dataset.data]

        # TODO: load from file if exist, if not exist, then save if it's the first fold test.
        # TODO: save each feature type as separately, e.g., cycle4.pkl, degree.pkl, etc.
        feature_names = self.graph_fea_reg.get_registered()
        graph_features = []
        for ts in feature_names:
            # NOTE: check existence.
            name = ts[0]
            add_features_path = os.path.join(
                self.processed_dir, f'graphwise_{self.name}_add_{name}.pkl')
            if os.path.exists(add_features_path):
                with open(add_features_path, 'rb') as f:
                    graph_feature = pk.load(f)
                    print('laod graph_features len: ', len(graph_feature))
                    print('load graph_feature: ', name)
                    graph_features.append(graph_feature)
                # remove from register_node_features.
                self.graph_fea_reg.remove(name)

        # NOTE: generate rest features:
        if len(self.graph_fea_reg.get_registered()) > 0:
            print('Generate rest features!',
                  self.graph_fea_reg.get_registered())
            rest_graph_features = node_feature_utils.register_features(
                adjs, self.graph_fea_reg)
            # save each
            for i, ts in enumerate(self.graph_fea_reg.get_registered()):
                add_features_path = os.path.join(
                    self.processed_dir, f'graphwise_{self.name}_add_{ts[0]}.pkl')
                graph_features.append(rest_graph_features[i])
                print('rest graph features: ', rest_graph_features[i][0].shape)
                with open(add_features_path, 'wb') as f:
                    pk.dump(rest_graph_features[i], f)
                    print('dump graph features: ', ts[0])

        print('aft:', len(graph_features), ' shape: ',
              graph_features[0][0].shape, graph_features[0][3].shape)

        graph_features = node_feature_utils.composite_graph_feature_list(
            graph_features)
        # 2022.10.20, NOTE: normalize:

        if 'norm_feature' in self.config:
            if self.config['norm_feature']:
                print('Need to normalize graph features !!!!!!!!!!!')
                graph_features = my_utils.normalize(
                    graph_features, along_axis=-1)

        # store in graph as graph not x, but g_x.
        for i, d in enumerate(self.dataset.data):
            # concatenate with pre features.
            d.set_additional_attr('g_x', torch.FloatTensor(graph_features[i]))

        print('add graph feature done!')


    def _add_additional_features(self, additional_features_list:list):
        if len(additional_features_list) < 1:
            return None
        
        node_fea_reg = node_feature_utils.NodeFeaRegister()
        for feature_arg in additional_features_list:
            node_fea_reg.register_by_str(feature_arg)
        self.node_fea_reg = node_fea_reg

        # TODO: check padding:
        need_pad = False
        if self.node_fea_reg.contains('kadj'):
            need_pad = True

        # get maximum node num:
        adjs = []
        max_N = 0
        for d in self.dataset.data:
            adjs.append(d.to_numpy_array())
            if max_N < d.N:
                max_N = d.N

        feature_names = self.node_fea_reg.get_registered()
        node_features = []
        for ts in feature_names:
            name = ts[0]
            add_features_path = os.path.join(
                self.processed_dir, f'{self.name}_add_{name}.pkl')
            print('add_features_path: ', add_features_path)
            if os.path.exists(add_features_path):
                with open(add_features_path, 'rb') as f:
                    node_feature = pk.load(f)
                    print('laod node_features len: ', len(node_feature))
                    print('load node_feature: ', name)
                    node_features.append(node_feature)
                # remove from register_node_features.
                self.node_fea_reg.remove(name)

        # NOTE: generate rest node features:
        if len(self.node_fea_reg.get_registered()) > 0:
            print('has rest features:')
            rest_node_features = node_feature_utils.register_features(
                adjs, self.node_fea_reg)
            # TODO: save each
            for i, ts in enumerate(self.node_fea_reg.get_registered()):
                add_features_path = os.path.join(
                    self.processed_dir, f'{self.name}_add_{ts[0]}.pkl')
                node_features.append(rest_node_features[i])

                with open(add_features_path, 'wb') as f:
                    pk.dump(rest_node_features[i], f)
                    print('dump node_feature: ', ts[0])

        print('aft:', len(node_features),
            ' shape: ', node_features[0][0].shape)


        # NOTE: padding
        if need_pad:
            node_features = node_feature_utils.composite_node_feature_list(
                node_features, padding=True, padding_len=max_N+10)
        else:
            node_features = node_feature_utils.composite_node_feature_list(
                node_features, padding=False)

        # 2022.10.20, NOTE: normalize:
        # TODO: normalize through each graph ????
        node_features = my_utils.normalize(
            node_features, along_axis=-1, same_data_shape=False)
        
        return node_features
    
    def _add_features(self):
        # TODO: load from files, if no files, create ???

        print('adding additional features --')
        all_features = self.additional_features.strip().split(',')
        additional_features_list ,use_features_list = [], []
        for s in all_features:
            if s.startswith('use_'): 
                use_features_list.append(s)
            else: 
                additional_features_list.append(s)
        
        addi_node_features = self._add_additional_features(additional_features_list)

        node_attribute = False
        if 'node_attribute' in self.config:
            node_attribute = self.config['node_attribute']
        print('original node_attribute: ', node_attribute)
        
        used_features = None
        if len(use_features_list) > 0:
            used_features = self._save_load_use_features()
            print('used_features len:', len(used_features))
            
        # NOTE: composite [attr, additional, used] those 3 features:
        
        for i, d in enumerate(self.dataset.data):
            # concatenate with pre features.
            new_x = []
            if node_attribute:
                new_x.append(d.x)
                
            if addi_node_features is not None:
                new_x.append(torch.FloatTensor(addi_node_features[i]))
                
            if used_features is not None:
                all_feas = []
                for each_fea in used_features:
                    if len(each_fea) > 0:
                        all_feas.append(torch.FloatTensor(each_fea[i]))
                all_feas = torch.cat(all_feas, dim=-1)
                new_x.append(all_feas)
                
            if len(new_x) == 0:
                raise EmptyNodeFeatureException
                
            d.x = torch.cat(new_x, axis=-1)
        # TODO: shuffle x among all node samples.
        if 'shuffle_feature' in self.config:
            if self.config['shuffle_feature']:
                node_num_total = 0
                node_index = {}
                start_id = 0
                for i, d in enumerate(self.dataset.data):
                    node_num = d.x.shape[0]
                    node_num_total += node_num
                    for j in range(node_num):
                        node_index[start_id] = (i, j)
                        start_id += 1
                        
                shuf_idx = list(np.arange(node_num_total))
                np.random.shuffle(shuf_idx)
                # construct pairs
                pairs = []
                for i in range(0, len(shuf_idx), 2):
                    if i + 1 < len(shuf_idx):
                        pairs.append((shuf_idx[i], shuf_idx[i+1]))
                
                print(f'shuffle feature!, total len: {node_num_total}, pair len: {len(pairs)}')
                # reconstruct:
                for (p1, p2) in pairs:
                    # swich p1 p2 in place
                    p1_node, p1_x_id = node_index[p1]
                    p2_node, p2_x_id = node_index[p2]
                    tmp = self.dataset.data[p1_node].x[p1_x_id]
                    self.dataset.data[p1_node].x[p1_x_id] = self.dataset.data[p2_node].x[p2_x_id]
                    self.dataset.data[p2_node].x[p2_x_id] = tmp
                # TODO: how to check correctness???????
                    
                
        print('added feature done!')

    def _save_load_use_features(self, graphs=None):
        raise NotImplementedError
    
    def _process(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        targets = self.dataset.get_targets()
        all_idxs = np.arange(len(targets))

        if self.outer_k is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            else:
                outer_split = train_test_split(all_idxs,
                                               stratify=targets,
                                               test_size=self.holdout_test_size)
                train_o_split, test_split = outer_split
            split = {"test": all_idxs[test_split], 'model_selection': []}

            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                if self.holdout_test_size == 0:
                    train_i_split, val_i_split = train_o_split, []
                else:
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  stratify=train_o_targets,
                                                                  test_size=self.holdout_test_size)
                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                inner_kfold = self.kfold_class(
                    n_splits=self.inner_k, shuffle=True)
                for train_ik_split, val_ik_split in inner_kfold.split(train_o_split, train_o_targets):
                    split['model_selection'].append(
                        {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            self.splits.append(split)

        else:  # cross validation assessment strategy

            outer_kfold = self.kfold_class(
                n_splits=self.outer_k, shuffle=True)

            for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
                split = {
                    "test": all_idxs[test_ok_split], 'model_selection': []}

                train_ok_targets = targets[train_ok_split]

                if self.inner_k is None:  # holdout model selection strategy
                    assert self.holdout_test_size is not None
                    train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                  stratify=train_ok_targets,
                                                                  test_size=self.holdout_test_size)
                    split['model_selection'].append(
                        {"train": train_i_split, "validation": val_i_split})

                else:  # cross validation model selection strategy
                    inner_kfold = self.kfold_class(
                        n_splits=self.inner_k, shuffle=True)
                    for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                        split['model_selection'].append(
                            {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                self.splits.append(split)

        filename = self.processed_dir / f"{self.name}_splits.json"
        with open(filename, "w") as f:
            json.dump(self.splits[:], f, cls=NumpyEncoder)

    def _get_loader(self, dataset, batch_size=1, shuffle=True):
        # dataset = GraphDataset(data)
        sampler = RandomSampler(dataset) if shuffle is True else None

        # 'shuffle' needs to be set to False when instantiating the DataLoader,
        # because pytorch  does not allow to use a custom sampler with shuffle=True.
        # Since our shuffler is a random shuffler, either one wants to do shuffling
        # (in which case he should instantiate the sampler and set shuffle=False in the
        # DataLoader) or he does not (in which case he should set sampler=None
        # and shuffle=False when instantiating the DataLoader)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
                          pin_memory=True)

    def get_test_fold(self, outer_idx, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0

        idxs = self.splits[outer_idx]["test"]
        test_data = GraphDatasetSubset(self.dataset.get_data(), idxs)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = self._get_loader(test_data, batch_size, shuffle)

        return test_loader

    def get_model_selection_fold(self, outer_idx, inner_idx=None, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0
        inner_idx = inner_idx or 0

        idxs = self.splits[outer_idx]["model_selection"][inner_idx]
        train_data = GraphDatasetSubset(self.dataset.get_data(), idxs["train"])
        val_data = GraphDatasetSubset(
            self.dataset.get_data(), idxs["validation"])

        train_loader = self._get_loader(train_data, batch_size, shuffle)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = self._get_loader(val_data, batch_size, shuffle)

        return train_loader, val_loader


class TUDatasetManager(GraphDatasetManager):
    URL = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"
    classification = True

    def _download(self):
        url = self.URL.format(name=self.name)
        response = requests.get(url)
        stream = io.BytesIO(response.content)
        with zipfile.ZipFile(stream) as z:
            for fname in z.namelist():
                z.extract(fname, self.raw_dir)

    def _process(self):
        graphs_data, num_node_labels, num_edge_labels, Graph_whole = parse_tu_data(
            self.name, self.raw_dir)  # Graph_whole contains all nodes and edges in the dataset
        targets = graphs_data.pop("graph_labels")

        self.Graph_whole = Graph_whole
        print("in _process")
        # TODO, NOTE: whole graph level !!!
        if self.use_pagerank:
                self.Graph_whole_pagerank = nx.pagerank(self.Graph_whole)
        if self.use_eigen or self.use_eigen_norm:
            try:
                print("{name}".format(name=self.name))
                if self.use_eigen:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector.npy".format(name=self.name))
                else:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector_degree_normalized.npy".format(name=self.name))
                print('eigen shape:', self.Graph_whole_eigen.shape)
            except:
                num_node = get_dataset_node_num(self.name)
                adj_matrix = nx.to_numpy_array(self.Graph_whole)
                if self.use_eigen_norm:
                    # normalize adjacency matrix with degree
                    sum_of_rows = adj_matrix.sum(axis=1)
                    normalized_adj_matrix = np.zeros((num_node, num_node))
                    # deal with edge case of disconnected node:
                    for i in range(num_node):
                        if sum_of_rows[i] != 0:
                            normalized_adj_matrix[i, :] = adj_matrix[i,
                                                                     :] / sum_of_rows[i, None]
                    adj_matrix = normalized_adj_matrix
                print("start computing eigen vectors")
                w, v = LA.eig(adj_matrix)
                indices = np.argsort(w)[::-1]
                v = v.transpose()[indices]
                # only save top 200 eigenvectors
                if self.use_eigen:
                    np.save(
                        "DATA/{name}_eigenvector".format(name=self.name), v[:200])
                else:
                    np.save(
                        "DATA/{name}_eigenvector_degree_normalized".format(name=self.name), v[:200])
                self.Graph_whole_eigen = v
                print('eigen shape:', self.Graph_whole_eigen.shape)

            print('Graph_whole_eigen: ', self.Graph_whole_eigen)
            print('nonzero: ', np.count_nonzero(self.Graph_whole_eigen == 0))
            
            node_num = get_dataset_node_num(self.name)
            # why top 50????
            
            
            embedding = np.zeros((node_num, 50))
            for i in range(node_num):
                for j in range(50):
                    embedding[i, j] = self.Graph_whole_eigen[j, i]
            self.Graph_whole_eigen = embedding
            print(self.Graph_whole_eigen)
        if self.use_1hot:
            self.onehot = nn.Embedding(self.Graph_whole.number_of_nodes(), 64)

        if self.use_deepwalk:
            self.deepwalk = self.extract_deepwalk_embeddings(
                    "DATA/proteins.embeddings")

        if self.use_random_normal:
            num_of_nodes = self.Graph_whole.number_of_nodes()
            self.rn = np.random.normal(0, 1, (num_of_nodes, 50))

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v)
                            for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        graphs = []
        for i, target in enumerate(targets, 1):
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}
            G = create_graph_from_tu_data(
                graph_data, target, num_node_labels, num_edge_labels, Graph_whole)

            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                # TODO: convert to numpy : npy
                
                data = self._to_data(G)
                # TODO save here:

                dataset.append(data)
                G.__class__()
                graphs.append(G)
        
        # Save
        self._save_load_use_features(graphs=graphs)
        
        torch.save(dataset, self.processed_dir / f"{self.name}.pt")
        print(f"saved: {self.processed_dir} / saved : {self.name}.pt")
        
    def _save_load_use_features(self, graphs=None) -> list:

        res = []
        pagerank_fea, onehot_fea, random_fea, eigen_fea, deepwalk_fea = [], [], [], [], []

        def print_fea_info(str1, fea_path, fea:np.ndarray):
            if isinstance(fea, np.ndarray):
                print(f'{str1}: {fea_path}, shape: {fea.shape}')
            else:
                print(f'{str1}: {fea_path}, len: {len(fea)}, shape: {fea[0].shape}')
            
            
        if self.use_pagerank:
            save_dir = f'DATA/{self.name}_tensor_pagerank.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append([self.Graph_whole_pagerank[node]] * 50)
                    pagerank_fea.append(np.array(feas))
                    
                pk.dump(pagerank_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, pagerank_fea)
            else:
                pagerank_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, pagerank_fea)
                
            res.append(pagerank_fea)
         
        if self.use_1hot:
            save_dir = f'DATA/{self.name}_tensor_onehot.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        arr = self.onehot(torch.LongTensor([node-1]))
                        feas.append(list(arr.view(-1).detach().numpy())) 
                    onehot_fea.append(np.array(feas))
                    
                pk.dump(onehot_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor', save_dir, onehot_fea)
            else:
                onehot_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor', save_dir, onehot_fea)
                
            res.append(onehot_fea)
            
        if self.use_random_normal:
            save_dir = f'DATA/{self.name}_tensor_random.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        arr = self.rn[node-1, :]
                        feas.append(list(arr)) # [1,...,50]
                    feas = np.array(feas)
                    random_fea.append(feas) # (N, 50)
                    
                pk.dump(random_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, random_fea)
            else:
                random_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, random_fea)
                
                
            res.append(random_fea)
            
        if self.use_eigen:
            save_dir = f'DATA/{self.name}_tensor_eigen.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append(list(self.Graph_whole_eigen[node-1]))
                    eigen_fea.append(np.array(feas))
                    
                pk.dump(eigen_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, eigen_fea)
            else:
                eigen_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, eigen_fea)
                
            res.append(eigen_fea)
            
        if self.use_deepwalk:
            save_dir = f'DATA/{self.name}_tensor_deepwalk.pkl'
            if not os.path.exists(save_dir):
                for gg in graphs:
                    feas = []
                    for node in gg.nodes:
                        feas.append(list(self.deepwalk[node-1]))
                    deepwalk_fea.append(np.array(feas))
                    
                pk.dump(deepwalk_fea, open(save_dir, 'wb'))
                print_fea_info('save tensor:', save_dir, deepwalk_fea)
            else:
                deepwalk_fea = pk.load(open(save_dir, 'rb'))
                print_fea_info('load tensor:', save_dir, deepwalk_fea)
        
            res.append(deepwalk_fea)
        
        return res

    def _to_data(self, G):
        datadict = {}
        # embedding = None
        # if self.use_1hot:
        #     embedding = self.Graph_whole_embedding
        # elif self.use_random_normal:
        #     embedding = self.Graph_whole_embedding
        # elif self.use_pagerank:
        #     # embedding is essentially pagerank dictionary
        #     embedding = self.Graph_whole_pagerank
        # elif self.use_eigen:
        #     embedding = self.Graph_whole_eigen
        # elif self.use_deepwalk:
        #     embedding = self.Graph_whole_deepwalk
        # TODO: only save attributes

        node_features = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one,
                                self.use_shared, self.use_1hot, self.use_random_normal, self.use_pagerank,
                                self.use_eigen, self.use_deepwalk)
        datadict.update(x=node_features)

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)

        target = G.get_target(classification=self.classification)
        datadict.update(y=target)

        data = Data(**datadict)

        return data

    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(
            G).todense())  # I - D^{-1/2}AD^{-1/2}
        # print(X.shape, lap.shape)

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
                # print(lap.shape)
            else:
                v_plus, lap = self._vertex_decimation(lap)
                # print(lap.shape)
                # print(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(
        ), (max_eigenvec < 0).squeeze()

        # print(v_plus, v_minus)

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # I assumed v_minus should have ones, but this is not necessarily the case. So I added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - \
            torch.mm(torch.mm(L_plus_minus, torch.inverse(
                L_minus_minus)), L_minus_plus)

        return v_plus, L_new

    def _precompute_assignments(self):
        pass

    def extract_deepwalk_embeddings(self, filename):
        print("start to load embeddings")
        node_num = get_dataset_node_num(self.name)
        with open(filename) as f:
            feat_data = []
            for i, line in enumerate(f):
                info = line.strip().split()
                if i == 0:
                    feat_data = np.zeros((node_num, int(info[1])))
                else:
                    idx = int(info[0]) - 1
                    feat_data[idx, :] = list(map(float, info[1::]))

        print("finished loading deepwalk embeddings")
        return feat_data


class SyntheticManager(TUDatasetManager):

    def _download(self):
        if self.name == 'CSL':
            graphs = generate_CSL(each_class_num=150, N=41, S=[2, 3, 4, 7])
        elif self.name == 'MDG':
            graphs = generate_mix_degree_graphs()
        else:
            raise NotImplementedError

        labels = []
        G = []
        for (g, y) in graphs:
            G.append(g)
            labels.append(y)
        print('dataset name:', self.name, 'len of samples:', len(G))
        # TODO: save G only, test the pickle file size:
        pk.dump(G, open(f'{self.raw_dir}/{self.name}_graph.pkl', 'wb'))
        pk.dump(labels, open(f'{self.raw_dir}/{self.name}_label.pkl', 'wb'))
        print('saved pkl data')

    def _process(self):
        graph_nodes = defaultdict(list)
        graph_edges = defaultdict(list)
        node_labels = defaultdict(list)
        node_attrs = defaultdict(list)
        edge_labels = defaultdict(list)
        edge_attrs = defaultdict(list)

        graphs = pk.load(open(f'{self.raw_dir}/{self.name}_graph.pkl', 'rb'))
        graph_labels = pk.load(
            open(f'{self.raw_dir}/{self.name}_label.pkl', 'rb'))

        for i, g in enumerate(graphs):
            graph_nodes[i] = g.nodes
            graph_edges[i] = g.edges

        graphs_data = {
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_labels": graph_labels,
            "node_labels": node_labels,
            "node_attrs": node_attrs,
            "edge_labels": edge_labels,
            "edge_attrs": edge_attrs
        }

        print("in _process")

        if self.use_pagerank:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole
            self.Graph_whole_pagerank = nx.pagerank(self.Graph_whole)
        elif self.use_eigen or self.use_eigen_norm:
            try:
                print("{name}".format(name=self.name))
                if self.use_eigen:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector.npy".format(name=self.name))
                else:
                    self.Graph_whole_eigen = np.load(
                        "DATA/{name}_eigenvector_degree_normalized.npy".format(name=self.name))
                print(self.Graph_whole_eigen.shape)
            except:
                num_node = get_dataset_node_num(self.name)
                adj_matrix = nx.to_numpy_array(self.Graph_whole)
                if self.use_eigen_norm:
                    # normalize adjacency matrix with degree
                    sum_of_rows = adj_matrix.sum(axis=1)
                    normalized_adj_matrix = np.zeros((num_node, num_node))
                    # deal with edge case of disconnected node:
                    for i in range(num_node):
                        if sum_of_rows[i] != 0:
                            normalized_adj_matrix[i, :] = adj_matrix[i,
                                                                     :] / sum_of_rows[i, None]
                    adj_matrix = normalized_adj_matrix
                print("start computing eigen vectors")
                w, v = LA.eig(adj_matrix)
                indices = np.argsort(w)[::-1]
                v = v.transpose()[indices]
                # only save top 200 eigenvectors
                if self.use_eigen:
                    np.save(
                        "DATA/{name}_eigenvector".format(name=self.name), v[:200])
                else:
                    np.save(
                        "DATA/{name}_eigenvector_degree_normalized".format(name=self.name), v[:200])
                self.Graph_whole_eigen = v

            print(self.Graph_whole_eigen)
            print(np.count_nonzero(self.Graph_whole_eigen == 0))
            node_num = get_dataset_node_num(self.name)
            embedding = np.zeros((node_num, 50))
            for i in range(node_num):
                for j in range(50):
                    embedding[i, j] = self.Graph_whole_eigen[j, i]
            self.Graph_whole_eigen = embedding
            print(self.Graph_whole_eigen)
        elif self.use_1hot:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole

            self.Graph_whole_embedding = nn.Embedding(
                self.Graph_whole.number_of_nodes(), 64)
        elif self.use_deepwalk:
            self.Graph_whole_deepwalk = self.extract_deepwalk_embeddings(
                "DATA/proteins.embeddings")
        elif self.use_random_normal:
            # TODO: create whole graphs:
            Graph_whole = nx.disjoint_union_all(graphs)
            self.Graph_whole = Graph_whole
            num_of_nodes = self.Graph_whole.number_of_nodes()
            self.Graph_whole_embedding = np.random.normal(
                0, 1, (num_of_nodes, 50))
        else:
            print('use other base node features! e.g., degree')

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v)
                            for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        for i, g in enumerate(graphs):
            G = create_graph_from_nx(g, graph_labels[i])
            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                data = self._to_data(G)
                dataset.append(data)
                G.__class__()
        torch.save(dataset, self.processed_dir / f"{self.name}.pt")


class NCI1(TUDatasetManager):
    name = "NCI1"
    _dim_features = 37
    _dim_target = 2
    max_num_nodes = 111


class RedditBinary(TUDatasetManager):
    name = "REDDIT-BINARY"
    _dim_features = 1
    _dim_target = 2
    max_num_nodes = 3782


class Reddit5K(TUDatasetManager):
    name = "REDDIT-MULTI-5K"
    _dim_features = 1
    _dim_target = 5
    max_num_nodes = 3648


class Proteins(TUDatasetManager):
    name = "PROTEINS_full"
    _dim_features = 3
    _dim_target = 2
    max_num_nodes = 620


class DD(TUDatasetManager):
    name = "DD"
    _dim_features = 89
    _dim_target = 2
    max_num_nodes = 5748


class Enzymes(TUDatasetManager):
    name = "ENZYMES"
    _dim_features = 20
    _dim_target = 6
    max_num_nodes = 126


class IMDBBinary(TUDatasetManager):
    name = "IMDB-BINARY"
    _dim_features = 50
    _dim_target = 2
    max_num_nodes = 136


class IMDBMulti(TUDatasetManager):
    name = "IMDB-MULTI"
    _dim_features = 1
    _dim_target = 3
    max_num_nodes = 89


class Collab(TUDatasetManager):
    name = "COLLAB"
    _dim_features = 1
    _dim_target = 3
    max_num_nodes = 492


class Mutag(TUDatasetManager):
    name = "MUTAG"
    _dim_features = 71
    _dim_target = 2
    max_num_nodes = 100


class CSL(SyntheticManager):
    name = "CSL"
    _dim_features = 1
    _dim_target = 4
    max_num_nodes = 41


class MDG(SyntheticManager):
    name = "MDG"
    _dim_features = 1
    _dim_target = 3
    max_num_nodes = 200
