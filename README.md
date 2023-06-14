# Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs


## Prepare Datasets:

- before you run experiments, you could download all datasets at first using run_experiment.sh
- specify the config file in run_experiment.sh
- NOTE: you could specify all config files separately, or use the all_config.yml, in which you could specify all models and parameters in all_config.yml
- NOTE: we suggest you run models sequentially.


1. `mkdir DATA`
1. set the dataset name you want to run in run_experiment.sh, then `bash run_experiment.sh` will automatically download the dataset in ./DATA

## Configuration of benchmark, all config files are in gnn_comparison/*.yml, two types configs are: Baseline_*.yml, and GNN_*.yml.

- specify parameters including model name, batch size, lr, feature types, etc in  gnn_comparison/*.yml files
- then set the config name suffix Baseline_[xxxx].yml in run_experiment.sh file

## Generate and plot performance gap and effectiveness of benchmark results:

- `bash gnn_comparison/run_experiment.sh`
- all plots in the paper were generated using plot_performance_gaps.ipynb

## Regression:

- run generate in generate_regreesion_datasets.py
- run regression in regressor.ipynb

## Grpah kernel baselines:
- all kernels are in kernel_baseline.ipynb
