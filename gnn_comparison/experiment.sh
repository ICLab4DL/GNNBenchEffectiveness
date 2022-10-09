
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
dat="PROTEINS"
dat="CSL"
dat='all'
dat='NCI1'
dat='COLLAB'

# python3 PrepareDatasets.py DATA/SYNTHETIC --dataset-name ${dat} --outer-k 10 --use-degree
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-degree
# cp -r DATA/SYNTHETIC/${dat}/ DATA/


python3 gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1009 --debug 

# python3 Launch_Experiments.py --config-file config_GraphSAGE.yml --dataset-name ${dat} --result-folder lzd --debug