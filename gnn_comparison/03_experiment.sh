
#    CHEMICAL:
#         NCI1
#         DD
#         ENZYMES
#         PROTEINS
#    SOCIAL[_1 | _DEGREE]:
#         IMDB-BINARY
#         IMDB-MULTI
#         REDDIT-BINARY
#         REDDIT-MULTI-5K
#         COLLAB


dat='all'
dat='COLLAB'
dat="CSL"
dat='IMDB-BINARY'
dat="PROTEINS"
dat='NCI1'
dat='ENZYMES'
dat='DD'
dat='MUTAG'


# EEG03, 2022.10.24,
# nohup python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/01_config_Baseline_lzd_mlp.yml \
# --dataset-name ${dat} --result-folder result_eeg03_1024 --debug > eeg03_nohup_ppi.log 2>&1 &


# # GNN: EEG03, 2022.10.30,
# nohup python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml \
# --dataset-name ${dat} --result-folder result_eeg03_GNN_1030 --debug > eeg03_nohup_ppi.log 2>&1 &


nohup python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_Baseline_lzd_mlp.yml \
--dataset-name ${dat} --result-folder result_eeg03_MLP_0105 --debug > eeg03_nohup_ppi.log 2>&1 &


