# %%
# load datasets:
import numpy as np

import matplotlib.pyplot as plt

from dataset_utils.node_feature_utils import *
import sys,os
sys.path.append(os.getcwd())


from PrepareDatasets import DATASETS

print(DATASETS.keys())
"""
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'NCI1': NCI1,
    'DD': DD,
    "MUTAG": Mutag,
    'CSL': CSL
"""

data_names = ['PROTEINS']
data_names = ['DD']
data_names = ['ENZYMES']
data_names = ['NCI1']
data_names = ['IMDB-MULTI']
data_names = ['REDDIT-BINARY']
data_names = ['CIFAR10']
data_names = ['ogbg_molhiv']

# NOTE:new kernel:
data_names = ['DD', 'PROTEINS', 'ENZYMES']

data_names = ['ogbg_moltox21','ogbg-molbace']

data_names = ['MUTAG']
data_names = []
datasets_obj = {}
for k, v in DATASETS.items():
    if k not in data_names:
        continue
    print('loaded dataset, name:', k)
    dat = v(use_node_attrs=True)
    datasets_obj[k] = dat


import dataset_utils.node_feature_utils as nfu
from scipy.stats import pearsonr

import json

"""
{"best_config": {"config": {"model": "GIN", "device": "cuda:1", "batch_size": 64, "learning_rate": 0.0001, "classifier_epochs": 200, "hidden_units": [64, 64, 64, 64], "layer_num": 5, "optimizer": "Adam", "scheduler": {"class": "StepLR", "args": {"step_size": 50, "gamma": 0.5}}, "loss": "MulticlassClassificationLoss", "train_eps": false, "l2": 0.0, "aggregation": "sum", "gradient_clipping": null, "dropout": 0.5, "early_stopper": {"class": "Patience", "args": {"patience": 50, "use_loss": false}}, "shuffle": true, "resume": false, "additional_features": "degree", "node_attribute": false, "shuffle_feature": false, "roc_auc": false, "mol_split": false, "dataset": "syn_cc", "config_file": "gnn_comparison/config_GIN_lzd_degree.yml", "experiment": "endtoend", "result_folder": "results/result_0422_GIN_lzd_degree_syn_cc_0.1", "dataset_name": "syn_cc", "dataset_para": "0.1", "outer_folds": 10, "outer_processes": 2, "inner_folds": 5, "inner_processes": 1, "debug": true, "ogb_evl": false}, "TR_score": 16.183574925298277, "VL_score": 21.505376272304083, "TR_roc_auc": -1, "VL_roc_auc": -1}, "OUTER_TR": 14.774557204254199, "OUTER_TS": 11.003236511378612, "OUTER_TR_ROCAUC": -1, "OUTER_TE_ROCAUC": -1}
"""

_OUTER_RESULTS_FILENAME = 'outer_results.json'

def get_test_acc(data_root_path, fold=10, as_whole=False):
    if data_root_path is None:
        return None if as_whole else [None for _ in range(fold)]
    
    if as_whole:
        assess_path = os.path.join(data_root_path, 'assessment_results.json')
        # load file to json, and return avg_TS_score and std_TS_score from json
        with open(assess_path, 'r') as fp:
            assess_results = json.load(fp)
            return float(assess_results['avg_TS_score'])
    
    outer_TR_scores,outer_TS_scores,outer_TR_ROCAUC,outer_TE_ROCAUC = [],[],[],[]
    for i in range(1, fold+1):
        config_filename = os.path.join(data_root_path, f'OUTER_FOLD_{i}', _OUTER_RESULTS_FILENAME)

        with open(config_filename, 'r') as fp:
            outer_fold_scores = json.load(fp)

            outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
            outer_TS_scores.append(outer_fold_scores['OUTER_TS'])
            
            if 'OUTER_TR_ROCAUC' in outer_fold_scores:
                outer_TR_ROCAUC.append(outer_fold_scores['OUTER_TR_ROCAUC'])
                outer_TE_ROCAUC.append(outer_fold_scores['OUTER_TE_ROCAUC'])

    return outer_TS_scores


def extract_features(adjs, labels):
    
    def get_mean_std_corr(features, labels):
        
        mean = np.array(np.mean(features))
        std = np.array(np.std(features))
        # print(np.isnan(mean).any(), np.isnan(std).any())
        x = np.array(features).reshape(-1)
        # NOTE: if multilabel, use the average of all labels:
        if len(labels[0].shape) > 1:
            corrs = []
            y = np.concatenate(labels, axis=0)
            for i in range(y.shape[1]):
                # ignore nan in y[:, i]
                not_nan = ~np.isnan(y[:, i])
                x_i = x[not_nan]
                y_i = y[not_nan]
                corr, _ = pearsonr(x_i, y_i[:, i].squeeze())
                if np.isnan(corr):
                    corr = np.array([0])
                corrs.append(corr)
            corr = np.mean(corrs)
        else:
            y = np.array(labels)
            # NOTE:
            
            not_nan = ~np.isnan(y)
            x_nn = x[not_nan]
            y_nn = y[not_nan]
            
            corr, _ = pearsonr(x_nn, y_nn)
            if np.isnan(corr):
                corr = np.array([0])
            
            if not isinstance(corr, np.ndarray):
                corr = np.array([corr])
                
        return np.array([mean.item(), std.item(), corr.item()])

    
    # F1: avgD:
    avg_d = [nfu.graph_avg_degree(adj=adj) for adj in adjs]
    f_avgD = get_mean_std_corr(avg_d, labels)
    # F2: avgCC:
    avg_cc = [nfu.node_cc_avg_feature(adj=adj) for adj in adjs]
    f_avgCC = get_mean_std_corr(avg_cc, labels)
    
    # F3: avgD/N:
    avg_DN = [nfu.graph_avgDN_feature(adj=adj) for adj in adjs]
    f_avgDN = get_mean_std_corr(avg_DN, labels)
    
    # F4: node num N:
    avg_N = [adj.shape[0] for adj in adjs]
    f_avgN = get_mean_std_corr(avg_N, labels)
    
    # F5: labels
    # calculate each dimension of labels:
    
    if len(labels[0].shape)> 1 and labels[0].shape[1] > 1:
        f_Ys = []
        Y = np.concatenate(labels, axis=0)
        for i in range(labels[0].shape[0]):
            y_i = Y[:, i]
            print('y_i: ', y_i.shape)
            y_i = y_i[~np.isnan(y_i)]
            print('y_i not nan: ', y_i.shape)
            
            f_Ys.append(get_mean_std_corr(y_i, y_i)[:2])
        f_Y = np.concatenate(f_Ys)
        f_Y = np.mean(f_Ys, axis=0)
    else:
        
        f_Y = get_mean_std_corr(labels, labels)[:2]
    
    
    # F6: cycles:
    avg_cyc = [nfu.graph_cycle_feature(adj=adj,k='4-5-6-7') for adj in adjs]
    f_cyc4 = get_mean_std_corr([c[0] for c in avg_cyc], labels)
    f_cyc5 = get_mean_std_corr([c[1] for c in avg_cyc], labels)
    f_cyc6 = get_mean_std_corr([c[2] for c in avg_cyc], labels)
    f_cyc7 = get_mean_std_corr([c[3] for c in avg_cyc], labels)
    
        
    feas = np.concatenate([f_avgD, f_avgCC, f_avgDN, f_avgN, f_Y, f_cyc4, f_cyc5, f_cyc6, f_cyc7], axis=0)
    return feas

# construct E of each fold, and plot

# Effectiveness 



# NOTE: get E for each dataset:

def plot_E(es, ax=None, title='E=(E_struct+E_attr)/2'):
    
    # e_res = sorted(es, key=lambda x:x[0])
    e_res = es
    labels = [e[1] for e in e_res]

    if ax is None:
        fig, ax = plt.subplots(dpi=100)
        
    for e in e_res:
        bars = ax.bar(e[1], e[0], label=e[1], hatch='\\', edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='center')
    ax.set_axisbelow(True)
    ax.grid(linestyle='dashed',zorder=0)
    ax.set_title(title)


"""
the ixj th sample : (D_i_split_j, E_i_j)
"""

# Check mutag
# construct each E with each fold and
def is_pyg_dataset(d_name:str):
    return d_name.startswith('ogb') or d_name.startswith('syn')

def get_dense_adjs(dataset, dataset_name):
    adjs = []

    if is_pyg_dataset(dataset_name):
        for d in dataset:
            if d.edge_index.numel() < 1:
                N = d.x.shape[0]
                adj = np.ones(shape=(N, N))
            else:
                adj = torch_utils.to_dense_adj(d.edge_index).numpy()[0]
            adjs.append(adj)
    else:
        # NOTE: not correct, need to be fixed
        if hasattr(dataset, 'dataset'):
            adjs = [d.to_numpy_array() for d in dataset.dataset]
        else:
            adjs = [d.to_numpy_array() for d in dataset]
        
    return adjs

def get_E(Acc_MLP_avg_degree, Acc_GNN_degree, Acc_MLP_attr, Acc_GNN_attr, is_abs=True):
    factor = 0.5
    if Acc_MLP_avg_degree is None:
        E_struct = 0
        factor = 1
    else:
        if is_abs:
            E_struct = (abs(Acc_GNN_degree - Acc_MLP_avg_degree) / min(Acc_MLP_avg_degree, Acc_GNN_degree)) * (100 - min(Acc_GNN_degree, Acc_MLP_avg_degree))
        else:
            E_struct = ((Acc_GNN_degree - Acc_MLP_avg_degree) / min(Acc_MLP_avg_degree, Acc_GNN_degree)) * (100 - min(Acc_GNN_degree, Acc_MLP_avg_degree))
    
    if Acc_MLP_attr is None:
        E_attribute = 0
        factor = 1
    else:
        if is_abs:
            E_attribute = (abs(Acc_GNN_attr - Acc_MLP_attr) / min(Acc_MLP_attr, Acc_GNN_attr)) * (100 - min(Acc_GNN_attr, Acc_MLP_attr))
        else:
            E_attribute = ((Acc_GNN_attr - Acc_MLP_attr) / min(Acc_MLP_attr, Acc_GNN_attr)) * (100 - min(Acc_GNN_attr, Acc_MLP_attr))
    return (E_struct+E_attribute) * factor



def get_new_E(Acc_MLP_attr, Acc_GNN_attr, Acc_MLP_avg_degree, Acc_GNN_degree, Y_num, is_abs=True):
    factor = 0.5
    if Acc_MLP_avg_degree is None:
        E_struct = 0
        factor = 1
    else:
        if is_abs:
            E_struct = abs(Acc_GNN_degree - Acc_MLP_avg_degree) * (100 -  min(Acc_MLP_avg_degree, Acc_GNN_degree)) * Y_num / (100 - 100/Y_num)
        else:
             E_struct = (Acc_GNN_degree - Acc_MLP_avg_degree) * (100 -  min(Acc_MLP_avg_degree, Acc_GNN_degree)) * Y_num / (100 - 100/Y_num)
    
    if Acc_MLP_attr is None:
        E_attribute = 0
        factor = 1
    else:
        if is_abs:
            E_attribute = abs(Acc_GNN_attr - Acc_MLP_attr) * (100 - min(Acc_MLP_attr, Acc_GNN_attr)) * Y_num / (100 -100/Y_num)
        else:
            E_attribute = (Acc_GNN_attr - Acc_MLP_attr) * (100 - min(Acc_MLP_attr, Acc_GNN_attr)) * Y_num / (100 -100/Y_num)
    return (E_struct+E_attribute) * factor

# %%

def E_datasets(dataset, 
               MLP_log_path_attr=None, GNN_log_path_attr=None, 
               MLP_log_path_struct=None, GNN_log_path_struct=None, fold=10, as_whole=False):
    
    
    MLP_test_acc_attr = get_test_acc(MLP_log_path_attr, fold=fold, as_whole=as_whole)
    GNN_test_acc_attr = get_test_acc(GNN_log_path_attr, fold=fold, as_whole=as_whole)
    
    MLP_test_acc_struct = get_test_acc(MLP_log_path_struct, fold=fold, as_whole=as_whole)
    GNN_test_acc_struct = get_test_acc(GNN_log_path_struct, fold=fold, as_whole=as_whole)

    print('_dim_targets: ', dataset._dim_target)
    mutag_splits = []

    if as_whole:
        adjs = get_dense_adjs(dataset, dataset.name)
        labels = dataset.get_labels()
        feas = extract_features(adjs=adjs, labels=labels)
        e = get_new_E(MLP_test_acc_attr, GNN_test_acc_attr, MLP_test_acc_struct, GNN_test_acc_struct, dataset._dim_target)
        mutag_splits.append((feas, e))
        return mutag_splits
        # labels = [d.y for d in train_loader.dataset] + [d.y for d in val_loader.dataset]
    else:
        for i in range(fold):
            train_loader, val_loader = dataset.get_model_selection_fold(outer_idx=i, inner_idx=0, batch_size=1, shuffle=False)
            adjs = get_dense_adjs(train_loader.dataset, dataset.name) + get_dense_adjs(val_loader.dataset, dataset.name)
            
            labels = [d.y for d in train_loader.dataset] + [d.y for d in val_loader.dataset]
            feas = extract_features(adjs=adjs, labels=labels)
            # e = get_E(MLP_test_acc_struct[i], GNN_test_acc_struct[i],  MLP_test_acc_attr[i], GNN_test_acc_attr[i])
            e = get_new_E( MLP_test_acc_attr[i], GNN_test_acc_attr[i], MLP_test_acc_struct[i], GNN_test_acc_struct[i], dataset._dim_target)
                
            mutag_splits.append((feas, e))
            
    return mutag_splits


# save datasets
import pickle as pk

def save_datasets(datasets, file_name):
    with open(file_name, 'wb') as f:
        pk.dump(datasets, f)

def load_datasets(file_name):
    with open(file_name, 'rb') as f:
        datasets = pk.load(f)
    return datasets



# pref = 'whole_'

pref = 'new_'
def generate_save_regression_dataset(dataset_name:str,
                                     MLP_log_path_attr=None, GNN_log_path_attr=None,
                                     MLP_log_path_degree=None, GNN_log_path_degree=None,
                                     as_whole=False):
    
    data_names = [dataset_name]
    datasets_obj = {}
    for k, v in DATASETS.items():
        if k not in data_names:
            continue
        print('loaded dataset, name:', k)
        dat = v(use_node_attrs=True)
        datasets_obj[k] = dat
        
    dataset = datasets_obj[dataset_name]
    
    cur_datasets = E_datasets(dataset, MLP_log_path_attr, GNN_log_path_attr,
                                MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
    # mutag_datasets = E_datasets(dataset, MLP_log_path_degree, GNN_log_path_degree, MLP_log_path_degree, GNN_log_path_degree)

    save_datasets(cur_datasets, f'{pref}{dataset_name.lower()}_datasets.pkl')
    
    
def generate_mutag(as_whole=False):
    MLP_log_path_attr = f'./results/result_GIN_0327_finger_mlp_attr_multicrossen_MUTAG/MolecularFingerprint_MUTAG_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_MUTAG/GIN_MUTAG_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0423_Baseline_lzd_mlp_MUTAG/MolecularGraphMLP_MUTAG_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0403_GIN_degree_MUTAG/GIN_MUTAG_assessment/10_NESTED_CV'
    generate_save_regression_dataset('MUTAG', MLP_log_path_attr, GNN_log_path_attr,
                                     MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)


def generate_DD(as_whole=False):
        
    MLP_log_path_attr = f'./results/result_0516_Baseline_lzd_fingerprint_attr_DD/MolecularFingerprint_DD_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0516_GIN_lzd_attr_DD/GIN_DD_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0516_Baseline_lzd_mlp_DD/MolecularGraphMLP_DD_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0516_GIN_lzd_degree_DD/GIN_DD_assessment/10_NESTED_CV'
    generate_save_regression_dataset('DD', MLP_log_path_attr, GNN_log_path_attr, 
                                     MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)


def generate_PROTEINS(as_whole=False):
    MLP_log_path_attr = f'./results/result_GIN_0327_finger_mlp_attr_crossen_PROTEINS/MolecularFingerprint_PROTEINS_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_PROTEINS/GIN_PROTEINS_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0423_Baseline_lzd_mlp_PROTEINS/MolecularGraphMLP_PROTEINS_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0403_GIN_degree_PROTEINS/GIN_PROTEINS_assessment/10_NESTED_CV'
    generate_save_regression_dataset('PROTEINS', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)

def generate_ENZYMES(as_whole=False):
    MLP_log_path_attr = f'./results/result_0516_Baseline_lzd_fingerprint_attr_ENZYMES/MolecularFingerprint_ENZYMES_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_ENZYMES/GIN_ENZYMES_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0516_Baseline_lzd_mlp_ENZYMES/MolecularGraphMLP_ENZYMES_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0403_GIN_degree_ENZYMES/GIN_ENZYMES_assessment/10_NESTED_CV'
    generate_save_regression_dataset('ENZYMES', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)


def generate_CIFAR10(as_whole=False):
    MLP_log_path_attr = f'./results/result_0510_Baseline_lzd_fingerprint_attr_CIFAR10/MolecularFingerprint_CIFAR10_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0510_GIN_lzd_attr_CIFAR10/GIN_CIFAR10_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0510_Baseline_lzd_mlp_CIFAR10/MolecularGraphMLP_CIFAR10_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0510_GIN_lzd_degree_CIFAR10/GIN_CIFAR10_assessment/10_NESTED_CV'
    generate_save_regression_dataset('CIFAR10', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)

def generate_MNIST(as_whole=False):
        
    MLP_log_path_degree = f'./results/result_0510_Baseline_lzd_mlp_MNIST/MolecularGraphMLP_MNIST_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0510_GIN_lzd_degree_MNIST/GIN_MNIST_assessment/10_NESTED_CV'
    MLP_log_path_attr = f'./results/result_0510_Baseline_lzd_fingerprint_attr_MNIST/MolecularFingerprint_MNIST_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0510_GIN_lzd_attr_MNIST/GIN_MNIST_assessment/10_NESTED_CV'
    generate_save_regression_dataset('MNIST', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
    
def generate_AIDS(as_whole=False):
    MLP_log_path_degree = f'./results/result_GIN_0405_graph_mlp_avgDegree_AIDS/MolecularGraphMLP_AIDS_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0405_GIN_degree_AIDS/GIN_AIDS_assessment/10_NESTED_CV'
    MLP_log_path_attr = f'./results/result_0517_Baseline_lzd_fingerprint_attr_AIDS/MolecularFingerprint_AIDS_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0405_GIN_attr_AIDS/GIN_AIDS_assessment/10_NESTED_CV'
    generate_save_regression_dataset('AIDS', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
    

def generate_NCI1(as_whole=False):
    MLP_log_path_degree = f'./results/result_0423_Baseline_lzd_mlp_NCI1/MolecularGraphMLP_NCI1_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0403_GIN_degree_NCI1/GIN_NCI1_assessment/10_NESTED_CV'
    MLP_log_path_attr = f'./results/result_GIN_0327_finger_mlp_attr_multicrossen_NCI1/MolecularFingerprint_NCI1_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_NCI1/GIN_NCI1_assessment/10_NESTED_CV'
    generate_save_regression_dataset('NCI1', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)


def generate_IMDB_B(as_whole=False):

    MLP_log_path_degree = f'./results/result_0424_Baseline_lzd_mlp_IMDB-BINARY/MolecularGraphMLP_IMDB-BINARY_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0226_load_degree_norm/GIN_IMDB-BINARY_assessment/10_NESTED_CV'
    # MLP_log_path_attr = f'./results/result_GIN_0327_finger_mlp_attr_multicrossen_NCI1/MolecularFingerprint_NCI1_assessment/10_NESTED_CV'
    # GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_NCI1/GIN_NCI1_assessment/10_NESTED_CV'
    generate_save_regression_dataset('IMDB-BINARY', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                        MLP_log_path_degree=MLP_log_path_degree,
                                        GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole)



def generate_IMDB_M(as_whole=False):


    MLP_log_path_degree = f'./results/result_0424_Baseline_lzd_mlp_IMDB-MULTI/MolecularGraphMLP_IMDB-MULTI_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0313_only_degree_IMDB-MULTI/GIN_IMDB-MULTI_assessment/10_NESTED_CV'
    # MLP_log_path_attr = f'./results/result_GIN_0327_finger_mlp_attr_multicrossen_NCI1/MolecularFingerprint_NCI1_assessment/10_NESTED_CV'
    # GNN_log_path_attr = f'./results/result_GIN_0404_GIN_attr_NCI1/GIN_NCI1_assessment/10_NESTED_CV'
    generate_save_regression_dataset('IMDB-MULTI', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                        MLP_log_path_degree=MLP_log_path_degree,
                                        GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole)
    

def generate_HIV(as_whole=False):
    MLP_log_path_attr = f'./results/result_0511_Baseline_lzd_mlp_ogbg_molhiv/MolecularGraphMLP_ogbg_molhiv_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0510_GIN_lzd_attr_ogbg_molhiv/GIN_ogbg_molhiv_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0510_Baseline_lzd_fingerprint_attr_ogbg_molhiv/MolecularFingerprint_ogbg_molhiv_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0510_GIN_lzd_degree_ogbg_molhiv/GIN_ogbg_molhiv_assessment/10_NESTED_CV'

    generate_save_regression_dataset('ogbg_molhiv', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
    
def generate_tox21(as_whole=False):
    MLP_log_path_attr = f'./results/result_GIN_0411_atomencoder_attr_ogbg_moltox21/AtomMLP_ogbg_moltox21_assessment/1_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0409_EGNN_lzd_attr_ogbg_moltox21/EGNN_ogbg_moltox21_assessment/1_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0424_Baseline_lzd_mlp_mol_ogbg_moltox21/MolecularGraphMLP_ogbg_moltox21_assessment/1_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0410_GIN_lzd_degree_ogbg_moltox21/GIN_ogbg_moltox21_assessment/1_NESTED_CV'

    generate_save_regression_dataset('ogbg_moltox21', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
    


def generate_bace(as_whole=False):
    MLP_log_path_attr = f'./results/result_0510_Baseline_lzd_fingerprint_attr_ogbg-molbace/MolecularFingerprint_ogbg-molbace_assessment/10_NESTED_CV'
    GNN_log_path_attr = f'./results/result_GIN_0510_GIN_lzd_attr_ogbg-molbace/GIN_ogbg-molbace_assessment/10_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0510_Baseline_lzd_mlp_ogbg-molbace/MolecularGraphMLP_ogbg-molbace_assessment/10_NESTED_CV'
    GNN_log_path_degree = f'./results/result_GIN_0510_GIN_lzd_degree_ogbg-molbace/GIN_ogbg-molbace_assessment/10_NESTED_CV'

    generate_save_regression_dataset('ogbg-molbace', MLP_log_path_attr, GNN_log_path_attr,
                                        MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
def generate_ppa(as_whole=False):

    MLP_log_path_attr = f'./results/result_0508_Baseline_lzd_mlp_edge_attr_ogbg_ppa/MolecularFingerprint_ogbg_ppa_assessment/1_NESTED_CV'
    GNN_log_path_attr = f'./results/result_0508_GIN_lzd_degree_ogbg_ppa_/GIN_ogbg_ppa_assessment/1_NESTED_CV'
    MLP_log_path_degree = f'./results/result_0507_Baseline_lzd_mlp_ogbg_ppa/MolecularGraphMLP_ogbg_ppa_assessment/1_NESTED_CV'
    GNN_log_path_degree = f'./results/result_0508_GIN_lzd_attr_edge_ogbg_ppa/OGBGNN_ogbg_ppa_assessment/1_NESTED_CV'
    
    generate_save_regression_dataset('ogbg_ppa', MLP_log_path_attr, GNN_log_path_attr,
                                     MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole)
        

# %%
# generate_mutag()
# generate_NCI1()
# generate_AIDS()


# generate_bace()
# generate_DD()
# generate_ENZYMES()
# generate_PROTEINS()
# generate_HIV()
# generate_tox21()
# generate_ppa()

# generate_IMDB_M()
# generate_IMDB_B()
# TODO: reddit_Binary
# TODO: COLLAB

generate_CIFAR10()


# generate_MNIST()
