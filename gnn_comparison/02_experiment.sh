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

gpu=02

dat='all'
dat='NCI1'
dat='ENZYMES'
dat='DD'
dat="CSL"
dat='COLLAB'
dat='REDDIT-BINARY'
dat="PROTEINS"
dat='IMDB-BINARY'
dat='MUTAG'

# 2023.01.28

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/config_GIN_lzd.yml \
--dataset-name ${dat} --result-folder result_GIN_0128 --debug > ${gpu}_nohup.log 2>&1 &