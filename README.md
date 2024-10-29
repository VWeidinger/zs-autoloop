# LDB: Model Repository
This repository contains different cost model architectures and baselines.

## Train a model
To train a model remotely, please configure the script and run it with the following command:
```
python3 exp_train_model.py--task setup deliver start monitor
```


To train a model locally, please configure the script and run it with the following command:
```
python3 main.py 
--mode train --model_type dace --target_dir ../data/evaluation/dace  --model_dir ../data/models/dace 
--statistics_file ../data/statistics/parsed_plans_feature_statistics.json 
--workload_runs 
../data/runs/parsed_plans/credit_scaled5/workload_100k_s1.json 
../data/runs/parsed_plans/tpc_h_scaled1/workload_100k_s1.json --num_workers 1 
--column_statistics /Users/rheinrich/Projekte/ldb_models/data/colum_statistics/column_statistics_scaled5.json 
--test_workload_runs /Users/rheinrich/Projekte/ldb_models/data/runs/parsed_plans/airline_scaled1/workload_100k_s1.json
```

To generate feature statistics (both for JSON and Parsed Plans), please run the following command:
```
python3 gather_feature_stats.py 
--database airline_scaled1 credit_scaled5 geneea_scaled23 hepatitis_scaled2000 movielens_scaled8 tpc_h_scaled1 
--workload workload_100k_s1.json 
--target_folder ../data/statistics/ 
--file_name feature_statistics
```

## How to profiling

Run the train command with the `--pt_profiler`. The profiling results will then be stored next to the model files (same filename, file ending: `.prof`)

To view the profiling results, do the following:
1. Download to your local machine (either manually or using the octopus scrip at `scripts/exp_runner/exp_pickup_profiler.py`)
2. Open `chrome://tracing` in your Edge/Chrome browser and load the file