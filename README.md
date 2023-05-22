# Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs

## Prepare Datasets:

- `mkdir DATA`
- set the dataset name you want to run in experiment.sh, then `bash experiment.sh` will automatically download the dataset in ./DATA

## Configuration of benchmark:

- check gnn_comparison/*.yml files associated in experiment.sh

## Generate Perfomance gap and effectiveness of benchmark results:

- run: `bash experiment.sh` for all datasets.
- run plot in plot_performance_gaps.ipynb

## Regression:

- run generate in generate_regreesion_datasets.py
- run regression in regressor.ipynb


