from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens, SetupVenv

from classes.classes import ModelConfig, ModelType, \
    DACEModelConfig, DACEModelNoCostsConfig, QPPModelNoCostsConfig, FlatModelActCardModelConfig, QPPModelActCardsConfig, \
    DACEModelActCardConfig, QPPNetModelConfig, ZeroShotModelActCardConfig, QueryFormerModelConfig, TrainingServers, \
    ScaledPostgresModelConfig, ZeroShotModelConfig, FlatModelConfig, E2EModelConfig, MSCNModelConfig

from scripts.exp_runner.exp_runner import ExpRunner


def generate_training_commands(datasets: List[Dataset], seeds: list[int], python_version: str,
                               model_list: list[ModelConfig] = ModelConfig) -> List[str]:

    commands = []
    for target_dataset in datasets:
        for target_model in model_list:
            training_runs = " ".join(target_model.get_training_workloads(training_datasets=datasets, target_dataset=target_dataset))
            test_runs = target_model.get_test_workloads(target_dataset)
            statistics_path = target_model.get_statistics(source_path=ClusterPaths(), dataset=target_dataset)

            for seed in seeds:
                command = (f"python{python_version} main.py "
                           f"--mode train "
                           f"--wandb_project cost-eval "
                           f"--model_type {target_model.name.NAME} "
                           f"--device {target_model.device} "
                           f"--model_dir {target_model.get_model_dir(ClusterPaths(), target_dataset)} "
                           f"--target_dir {target_model.get_eval_dir(ClusterPaths(), target_dataset)} "
                           f"--workload_runs {training_runs} "
                           f"--statistics_file {statistics_path} "
                           f"--seed {seed} "
                           f"--wandb_name {target_model.name.NAME}/{target_dataset.db_name}/{seed}")

                if target_model.type == ModelType.WL_AGNOSTIC:
                    command += f" --test_workload_runs {test_runs} "

                if target_model.column_statistics:
                    command += f" --column_statistics {target_model.get_column_stats(target_dataset)} "

                if target_model.word_embeddings:
                    command += f" --word_embeddings {target_model.get_word_embeddings(source_path=ClusterPaths(), database=target_dataset)} "

                if target_model.hyperparameter:
                    command += f" --hyperparameter_path {target_model.hyperparameter} "

                commands.append(command)
    return commands


if __name__ == '__main__':
    install_venv = False

    setup_steps = [Rsync(src=[str(LocalPaths().Models.src)], dest=[ClusterPaths().Models.root], update=True, put=True)]

    if install_venv:
        setup_steps.append(SetupVenv(use_requirements_txt=True,
                                     requirements_txt_filename=f'{ClusterPaths().Models.src}/requirements.txt',
                                     force=False,
                                     python_cmd=f'python3.9',
                                     python_version='3.9'))

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().Data.runs], dest=[LocalPaths().data], put=False, update=True)]

    datasets = [
        Dataset('airline_scaled1'),
        Dataset('credit_scaled5'),
        Dataset('geneea_scaled23'),
        Dataset('hepatitis_scaled2000'),
        Dataset('movielens_scaled8'),
        Dataset('tpc_h_scaled1')
    ]

    seeds = [0, 1, 2]
    node = TrainingServers.NODE03

    # Standard Models
    # model = ScaledPostgresModelConfig()
    # model = FlatModelConfig()
    # model = E2EModelConfig()
    # model = QPPNetModelConfig()
    # model = MSCNModelConfig()
    # model = DACEModelConfig()
    # model = QueryFormerModelConfig()
    model_config = ZeroShotModelConfig()

    # Ablation Models (without postgres Estimates)
    # model = DACEModelNoCostsConfig()
    # model = QPPModelNoCostsConfig()

    # Ablation Models (actual cardinalities)
    # model = FlatModelActCardModelConfig()
    # model = QPPModelActCardsConfig()
    # model = DACEModelActCardConfig()
    # model = ZeroShotModelActCardConfig()

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       root_path=ClusterPaths().root,
                       python_version=node["python"])

    eval_commands = []
    for dataset in datasets:
        eval_commands += generate_training_commands(datasets=datasets,
                                                    model_list=[model_config],
                                                    python_version=node["python"],
                                                    seeds=seeds)

    runner.run_exp(node_names=[node["hostname"]],
                   commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=len(datasets),
                   offset=0)
