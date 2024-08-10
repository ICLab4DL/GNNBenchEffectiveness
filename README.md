# Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs

This repository supports our paper titled "Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs," which has been accepted by IJCAI 2024. You can access the paper and get bibtex [here](https://www.ijcai.org/proceedings/2024/237).


## Prepare Datasets:

1. Create a directory for data:
   ```bash
   mkdir DATA
   ```
2. Navigate to the `gnn_comparison` directory:
   ```bash
   cd gnn_comparison/
   ```
3. Set the dataset name in `gnn_comparison/run_real_experiment.sh`, then execute:
   ```bash
   bash run_real_experiment.sh
   ```
   This script will automatically download the dataset into the `./DATA` directory and run the benchmark. If you only need to download the dataset, run:
   ```bash
   bash prepare_experiment.sh
   ```
4. **NOTE:** Some bash parameters need to be set in `run_real_experiment.sh`, such as `dats` (dataset names), `model_set` (models to run, corresponding to each running config file *.yml), etc. You can run experiments in parallel by setting `dats` and `model_set` to multiple values, but we suggest running them one by one to avoid memory issues.

## Configuration of Benchmark

All configuration files are located in the `gnn_comparison/*.yml` directory.

- Specify parameters including model name, batch size, learning rate, feature types, etc., in the `gnn_comparison/*.yml` files.
- For simplicity, we have separated each main configuration into different config files, such as `config_GIN_attr.yml`, `config_GCN_degree.yml`, etc.

## Run Benchmark

1. Specify the config file name in `run_real_experiment.sh` for real-world datasets or `run_syn_experiment.sh` for synthetic datasets.
2. Set config file parameters in `config_Baseline_[xxxx].yml`, `config_GIN_[xxxx].yml`, `config_GCN_[xxxx].yml`, etc. Check the details in `gnn_comparison/*.yml`.
3. Run the benchmark:
   ```bash
   bash gnn_comparison/run_real_experiment.sh
   ```
   or
   ```bash
   bash gnn_comparison/run_syn_experiment.sh
   ```
4. **NOTE:** All log and result locations are specified in `run_real_experiment.sh` and `run_syn_experiment.sh`. The results are saved in the `./results/` folder for further performance analysis. The folder name will be used for extracting statistics in `plot_performance_gaps.ipynb` and `plot_statistics.ipynb`.

## Generate Performance Gap and Effectiveness of Benchmark Results

- The results presented in the paper were generated in `plot_performance_gaps.ipynb`.
- Some statistics of datasets can be found in `plot_statistics.ipynb`.

## Graph Kernel Baselines

- All kernels are implemented in `kernel_baseline.ipynb`.
- For parallel processing, run:
  ```bash
  bash run_kernel_baseline.sh
  ```

## Regression

- Generate datasets using `generate_regression_datasets.py`.
- Run regression analysis in `regressor.ipynb`.

For further details, please refer to our paper [here](https://www.ijcai.org/proceedings/2024/237).
