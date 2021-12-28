import numpy as np
import networkx as nx

from torch.utils.tensorboard import SummaryWriter

# torch
import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from torch_sparse import SparseTensor, matmul

from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing

# OGB:
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
 
# sys
from typing import Union, Tuple
import pickle
import os
import random
import argparse
import configparser

# from local models:
from model import *


def train(model, predictor, edge_attr, node_emb, emb_ea, adj_t, split_edge, optimizer, batch_size):
    edge_index = adj_t

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(node_emb.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(node_emb, adj_t, edge_attr, emb_ea)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=node_emb.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(node_emb, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, edge_attr, x, emb_ea, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t, edge_attr, emb_ea)

    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def main(jupyter=False):

    parser = argparse.ArgumentParser(description='Link_Pred_DDI')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--node_emb', type=int, default=256)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--server_tag', type=str, default='torch17')

    arg_list = None
    if jupyter:
        # load from config.ini
        config = configparser.ConfigParser()
        config.read('config.ini')
        arg_list = []
        for k, v in config['train'].items():
            arg_list.append("--"+k)
            arg_list.append(v)

    args = parser.parse_args(arg_list)

    print('args: ', args)

    import time
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./tboard/"+args.server_tag+"/" + dt
    print('tensorboard path:', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]
    edge_index = data.edge_index.to(device)
    split_edge = dataset.get_edge_split()

    # init model:
    # model = GraphSAGE(args.node_emb, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout).to(device)
    
    model = MultiGCN(args.node_emb, args.hidden_channels, args.hidden_channels, args.num_layers).to(device)
    
    emb = torch.nn.Embedding(data.num_nodes, args.node_emb).to(device)
    emb_ea = torch.nn.Embedding(args.num_samples, args.node_emb).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers+1, args.dropout).to(device)

    print('Number of parameters:',
          sum(p.numel() for p in list(model.parameters()) +
          list(predictor.parameters()) + list(emb.parameters()) + list(emb_ea.parameters())))

    # ------ start encode distance information
    np.random.seed(0)
    
    if os.path.exists('ppi_graph.pkl'):
        nx_graph = pickle.load(open('ppi_graph.pkl', 'rb'))
        print('loaded networkx!')
    else:
        nx_graph = to_networkx(data, to_undirected=True)
        pickle.dump(nx_graph, open('ppi_graph.pkl', 'wb'))
        print('dumped networkx!')

    node_mask = []
    for _ in range(args.num_samples):
        node_mask.append(np.random.choice(500, size=200, replace=False))
    node_mask = np.array(node_mask)
    node_subset = np.random.choice(nx_graph.number_of_nodes(), size=500, replace=False)
    print('num of node: ', nx_graph.number_of_nodes())
    spd = get_spd_matrix(G=nx_graph, S=node_subset, max_spd=5)
    spd = torch.Tensor(spd).to(device)
    edge_attr = spd[edge_index, :]
    edge_attr = edge_attr.mean(0)[:, node_mask].mean(2)
    # normalize:
    a_max = torch.max(edge_attr, dim=0, keepdim=True)[0]
    a_min = torch.min(edge_attr, dim=0, keepdim=True)[0]
    edge_attr = (edge_attr - a_min) / (a_max - a_min + 1e-6)
    print('edge_attr: ', edge_attr.shape)
    # ------- get edge_attr end -----

    
    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        random.seed(run)
        torch.manual_seed(run)
        torch.nn.init.xavier_uniform_(emb.weight)
        torch.nn.init.xavier_uniform_(emb_ea.weight)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(emb_ea.parameters()) + list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        writer.add_scalars(f'run_{run}/epoch/loss', {'Train_loss': loss, 'Val': valid_hits, 'Test':test_hits}, epoch)
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":

    main()
