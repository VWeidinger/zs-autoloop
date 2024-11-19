from pathlib import Path
from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens, SetupVenv, RsyncCurrentRepo

from classes.classes import ModelConfig, ModelType, \
    ZeroShotModelConfig
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers


def generate_training_commands(datasets: List[Dataset], seeds: list[int], python_version: str, wandb_project: str,
                               wandb_entity: str,
                               model_list: list[ModelConfig] = ModelConfig, quick_run: bool = False,
                               pt_profiler: bool = False, leave_one_out_style: bool = True,
                               ft_with_ranking: bool = False, bound_predictions: bool = False,
                               eval_join_order: bool = False) -> List[str]:
    commands = []
    for target_dataset in datasets:
        for target_model in model_list:
            training_runs = " ".join(
                target_model.get_training_workloads(training_datasets=datasets, target_dataset=target_dataset))
            test_runs = target_model.get_test_workloads(target_dataset)
            if eval_join_order:
                test_runs += " " + str(Path(ClusterPaths().Data.parsed_plans / "imdb_scaled1" / "join_order.json"))
            statistics_path = target_model.get_statistics(source_path=ClusterPaths(),
                                                          dataset=target_dataset,
                                                          format=target_model.input_format)

            for seed in seeds:
                model_suffix = ''

                if ft_with_ranking:
                    model_suffix += '_ft_rank'
                if bound_predictions:
                    model_suffix += '_bound'

                wandb_name = f"{target_model.name.NAME}{model_suffix}/{target_dataset.db_name}/{seed}"

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
                           f"--wandb_name {wandb_name}")

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

                if ft_with_ranking:
                    command += " --ft_with_ranking"

                if bound_predictions:
                    command += " --bound_predictions"

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
        # Dataset('airline', max_no_joins=5),
        # Dataset('imdb'),
        Dataset('ssb', max_no_joins=3),
        Dataset('tpc_h', max_no_joins=5),
        Dataset('walmart', max_no_joins=2),
        # Dataset('financial', scale=4),
        # Dataset('basketball', scale=200),
        # Dataset('accidents', scale=1, contain_unicode=True),
        # Dataset('movielens', scale=8),
        # Dataset('baseball', scale=10),
        Dataset('hepatitis', scale=2000),
        # Dataset('tournament', scale=50),
        Dataset('credit', scale=5),
        # Dataset('employee', scale=3),
        # Dataset('consumer', scale=6),
        Dataset('geneea', scale=23, contain_unicode=True),
        Dataset('genome', scale=6),
        # Dataset('carcinogenesis', scale=674),
        Dataset('seznam', scale=2),
        # Dataset('fhnk', scale=2)
    ]

    seeds = [1]
    node = TrainingServers.NODE04

    # ===================
    # CONFIGS
    # ===================
    quick_run = False # if True, it will run only a few epochs on a few workloads
    pt_profiler = False # Profiler for PyTorch
    leave_one_out_train_style = False  # if True, it will train all models for all datasets
    ft_with_ranking = False  # fine-tune with ranking (pretrains a model and then fine-tunes it with ranking)
    bound_predictions = False  # bound predictions to the range of the training data and output them as separate metrics - at the moment not competitive
    eval_join_order = False  # evaluate join order using JOB-Light runs with exhaustive join orders
    # ===================

    # Standard Models
    # model_config = ScaledPostgresModelConfig()        # --> Working
    # model_config = FlatModelConfig()                  # --> Working
    # model_config = E2EModelConfig()                   # --> ToDo: Adapt to new codebase and generate sentences
    # model_config = QPPNetModelConfig()                  # --> Working?
    # model_config = MSCNModelConfig()                  # --> ToDo: Adapt to new codebase and generate sentences
    # model_config = DACEModelConfig()                  # --> Working
    # model_config = QueryFormerModelConfig()           # --> ToDo: Generate Sentences
    model_config = ZeroShotModelConfig()  # --> Working
    # model_config = ZSLearnFromTimeoutConfig()         # --> Working
    # model_config = ZSRankLossConfig()                 # --> Working

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
                                               leave_one_out_style=leave_one_out_train_style,
                                               ft_with_ranking=ft_with_ranking,
                                               bound_predictions=bound_predictions,
                                               eval_join_order=eval_join_order)

    runner.run_exp(node_names=[node["hostname"]],
                   commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   screens_per_node=len(datasets),
                   offset=0)
