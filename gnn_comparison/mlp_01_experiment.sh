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

dt=0411
gpu=01
conf_file='config_fingerprint_attr.yml'
dats='IMDB-MULTI COLLAB'
dats='PATTERN'

dats='hiv'
dats='ogbg_molhiv'

dats='AIDS'

for dat in ${dats};do

echo 'running finger sumpool only attr: '${dat}
tag=finger_multiclassloss_sumpool_only_attr_${dat}

# --outer-folds 1 \
# --inner-folds 1 \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} --result-folder results/result_GIN_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done