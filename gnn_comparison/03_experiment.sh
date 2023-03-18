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



gpu=01
dt=0313
# conf_file='config_Adapter.yml'

# degree + attributes:
conf_file='config_GIN_lzd_pagerank.yml'

# dats='NCI1 ENZYMES'

dats='COLLAB NCI1'
for dat in ${dats};do

echo 'running only degree: '${dat}
tag=pagerank_${dat}

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} --result-folder results/result_GIN_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done