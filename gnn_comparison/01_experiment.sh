
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
dat='NCI1'
dat="CSL"
dat="PROTEINS"
dat='IMDB-BINARY'
dat='MUTAG'
dat='DD'
dat='IMDB-MULTI'
dat='ENZYMES'
dat='COLLAB'


# EEG01, 2023.01.03,
nohup python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_Baseline_lzd_mlp.yml \
--dataset-name ${dat} --result-folder result_eeg01_MLP_0103 --debug > eeg01_nohup_ppi.log 2>&1 &