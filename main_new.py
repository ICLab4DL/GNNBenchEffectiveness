import utils
from models import *
import dataset_loader
import training

import os
import time

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

    tu_train_dataloader, tu_test_dataloader = dataset_loader.load_data('AIDS', args)
    args.class_num = 2
    
    # get feature dimension.
    x, _ = tu_train_dataloader.dataset.__getitem__(0)
    print('node feature shape:', x.get_node_features().shape)
    node_feature_dim = x.get_node_features().shape[-1]
    
    gin_evls = []
    train_gin_evls = []
    train_evl, test_evl = training.train_gnn(args, tu_train_dataloader, tu_test_dataloader, 
                                        node_fea_dim=node_feature_dim, gnn_name=args.model_name, 
                                        pooling=args.gnn_pooling, epoch=args.epochs, plot=False)
    gin_evls.append(test_evl)
    train_gin_evls.append(train_evl)
        
    mi_f1, ma_f1, w_f1 = get_f1s(gin_evls)

    print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    fig_save_path = os.path.join(fig_save_dir, f'{file_tag}')
    test_evl.plot_metrics(utils.append_tag(fig_save_path, 'test'))
    train_evl.plot_metrics(utils.append_tag(fig_save_path, 'train'))