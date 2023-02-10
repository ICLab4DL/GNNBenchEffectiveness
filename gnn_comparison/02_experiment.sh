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
dt=0209
tag=random_id

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

# TODO: Random. 2023.02.09
nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/config_GIN_lzd.yml \
--dataset-name ${dat} --result-folder results/result_GIN_${dt}_${tag} --debug > ${gpu}_${dt}_${tag}_nohup.log 2>&1 &