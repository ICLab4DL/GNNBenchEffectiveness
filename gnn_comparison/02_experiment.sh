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
dat="CSL"
dat='COLLAB'
dat='REDDIT-BINARY'
dat='IMDB-BINARY' # no attribute
dat="PROTEINS"
dat='MUTAG'
dat='DD'
dat='NCI1'
dat='ENZYMES'


# conf_file='config_Adapter.yml'

# degree + attributes:

# dats='NCI1 ENZYMES'

dats='PATTERN'
dats='MUTAG NCI1 PROTEINS DD'
dats='ogbg_molhiv'
dats='CIFAR10'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'
dats='AIDS'


model_set='GIN_lzd_attr GIN_lzd_mix GIN_lzd_degree Baseline_lzd_mlp'

dt=0521
gpu=01
dats='ogbg-molbbbp'
dats='ogbg_moltox21'

dats='ogbg_moltox21 ogbg-molbace'

model_set='GIN_lzd_attr'

model_set='EGNN_lzd_attr'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

echo 'running '${conf_file}

tag=${ms}_${dat}

# --outer-folds 1 \
# --inner-folds 1 \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--mol_split True \
--ogb_evl True \
--dataset-name ${dat} --result-folder results/result_GIN_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done

done



# running config_EGNN_lzd_attr.yml
#     check log:
# tail -f logs/01_0406_EGNN_lzd_attr_ogbg_moltox21_nohup.log
# running config_EGNN_lzd_attr.yml
#     check log:
# tail -f logs/01_0406_EGNN_lzd_attr_ogbg-molbbbp_nohup.log
# running config_EGNN_lzd_attr.yml
#     check log:
# tail -f logs/01_0406_EGNN_lzd_attr_ogbg-molbace_nohup.log