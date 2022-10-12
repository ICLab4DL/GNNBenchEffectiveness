
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
dat='MUTAG'
dat='COLLAB'
dat='IMDB-BINARY'
dat='ENZYMES'
dat='DD'

# python3 PrepareDatasets.py DATA/SYNTHETIC --dataset-name ${dat} --outer-k 10 --use-degree
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-degree
# cp -r DATA/SYNTHETIC/${dat}/ DATA/


# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1009 --debug 

# python3 Launch_Experiments.py --config-file config_GraphSAGE.yml --dataset-name ${dat} --result-folder lzd --debug

python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name ${dat} --result-folder result_1012 --debug 

# TODO: test all datasets using all models.

# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 
# python3 -u gnn_comparison/Launch_Experiments.py --config-file gnn_comparison/config_GraphSAGE_lzd.yml --dataset-name all --result-folder result_1009 --debug 




# 2022.10.09. dataset: IMDB-BINARY, GraphSAGE. (GIN better in paper)
#result_1009/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_1/HOLDOUT_MS/winner_o  {"config": {"model": "GraphSAGE", "device": "cuda:0", "batch_size": 32, "learning_rate": 0.001, "l2": 0.0, "classifier_epochs": 200, "optimizer": "Adam", "scheduler": null, "loss": "MulticlassClassificationLoss", "gradient_clipping": null, "early_stopper": {"class": "Patience", "args": {"patience": 100, "use_loss": false}}, "shuffle": true, "dim_embedding": 32, "num_layers": 3, "aggregation": "sum", "additional_features": "degree,tri_cycle", "dataset": "IMDB-BINARY"}, "TR_score": 66.04938271604938, "VL_score": 82.22222290039062}# 






# cat result_1009/pre_results/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_*/outer_results.json 
# cat result_1009/GraphSAGE_IMDB-BINARY_assessment/10_NESTED_CV/OUTER_FOLD_*/outer_results.json 