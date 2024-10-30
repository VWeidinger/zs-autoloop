from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens, SetupVenv, RsyncCurrentRepo

from classes.classes import ModelConfig, ModelType, \
    ZSLearnFromTimeoutConfig, ZeroShotModelConfig
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers


def generate_training_commands(datasets: List[Dataset], seeds: list[int], python_version: str, wandb_project: str,
                               wandb_entity: str,
                               model_list: list[ModelConfig] = ModelConfig, quick_run: bool = False,
                               pt_profiler: bool = False, leave_one_out_style: bool = True) -> List[str]:
    commands = []
    for target_dataset in datasets:
        for target_model in model_list:
            training_runs = " ".join(
                target_model.get_training_workloads(training_datasets=datasets, target_dataset=target_dataset))
            test_runs = target_model.get_test_workloads(target_dataset)
            statistics_path = target_model.get_statistics(source_path=ClusterPaths(),
                                                          dataset=target_dataset,
                                                          format=target_model.input_format)

            for seed in seeds:
                command = (f"python{python_version} main.py "
                           f"--mode train "
                           f"--wandb_project {wandb_project} "
                           f"--wandb_entity {wandb_entity} "
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

                if quick_run:
                    command += " --quick_run"

                if pt_profiler:
                    command += " --pt_profiler"

                commands.append(command)

        if not leave_one_out_style:
            break

    return commands


if __name__ == '__main__':
    install_venv = False

    setup_steps = [RsyncCurrentRepo(gitignore_path='.gitignore')]

    if install_venv:
        setup_steps.append(SetupVenv(use_requirements_txt=True,
                                     requirements_txt_filename=f'{ClusterPaths().Models.src}/requirements.txt',
                                     force=False,
                                     ))

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().Data.runs], dest=[LocalPaths().data], put=False, update=True)]

    datasets = [
        Dataset('airline', max_no_joins=5),
        # Dataset('imdb'),
        # Dataset('ssb', max_no_joins=3),
        Dataset('tpc_h', max_no_joins=5),
        # Dataset('walmart', max_no_joins=2),
        # Dataset('financial', scale=4),
        # Dataset('basketball', scale=200),
        # Dataset('accidents', scale=1, contain_unicode=True),
        Dataset('movielens', scale=8),
        # Dataset('baseball', scale=10),
        Dataset('hepatitis', scale=2000),
        # Dataset('tournament', scale=50),
        Dataset('credit', scale=5),
        # Dataset('employee', scale=3),
        # Dataset('consumer', scale=6),
        Dataset('geneea', scale=23, contain_unicode=True),
        # Dataset('genome', scale=6),
        # Dataset('carcinogenesis', scale=674),
        # Dataset('seznam', scale=2),
        # Dataset('fhnk', scale=2)
    ]

    seeds = [42]
    node = TrainingServers.NODE03

    quick_run = True
    pt_profiler = False
    leave_one_out_train_style = False

    # Standard Models
    # model_config = ScaledPostgresModelConfig()
    # model_config = FlatModelConfig()                  # --> Node 05 OK
    # model_config = E2EModelConfig()
    # model_config = QPPNetModelConfig()                # --> Node 00 OK
    # model_config = MSCNModelConfig()
    # model_config = DACEModelConfig()                  # --> Node 03 OK
    # model_config = QueryFormerModelConfig()           # --> Node
    # model_config = ZeroShotModelConfig()              # --> Node 04
    model_config = ZSLearnFromTimeoutConfig()

    # Ablation Models (without postgres Estimates)
    # model_config = DACEModelNoCostsConfig()
    # model_config = QPPModelNoCostsConfig()

    # Ablation Models (actual cardinalities)
    # model_config = FlatModelActCardModelConfig()
    # model_config = QPPModelActCardsConfig()
    # model_config = DACEModelActCardConfig()
    # model_config = ZeroShotModelActCardConfig()

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       python_version=node["python"])

    # for dataset in datasets:
    eval_commands = generate_training_commands(datasets=datasets,
                                               model_list=[model_config],
                                               python_version=node["python"],
                                               seeds=seeds,
                                               wandb_project=runner.wandb_project,
                                               wandb_entity=runner.wandb_entity,
                                               quick_run=quick_run,
                                               pt_profiler=pt_profiler,
                                               leave_one_out_style=leave_one_out_train_style
                                               )

    runner.run_exp(node_names=[node["hostname"]],
                   commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=len(datasets),
                   offset=0)
