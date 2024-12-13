from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens, SetupVenv, RsyncCurrentRepo, SetupRYEVenv

from classes.classes import ModelConfig, ModelType, \
    ZeroShotModelConfig
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers


def generate_training_commands(datasets: List[Dataset], seeds: list[int], python_version: str, wandb_project: str,
                               wandb_entity: str,
                               model_list: list[ModelConfig] = ModelConfig, quick_run: bool = False,
                               pt_profiler: bool = False, test_against: str = None,
                               ft_with_ranking: bool = False, bound_predictions: bool = False,
                               eval_join_order: bool = False, eval_access_path: bool = False,
                               eval_physical_operator: bool = False,
                               qo_eval_after_each_epoch: bool = False) -> List[str]:
    if test_against:
        print(f'Testing the models against {test_against}')

    commands = []
    for target_dataset in datasets:
        if test_against is not None and target_dataset.db_name != test_against:
            continue

        for target_model in model_list:
            training_runs = " ".join(
                target_model.get_training_workloads(training_datasets=datasets,
                                                    target_dataset=target_dataset,
                                                    stratified_workload=target_model.stratified_workloads))
            test_runs = target_model.get_test_workloads(target_dataset)
            if eval_join_order:  # and target_dataset.db_name in ['imdb']:
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), 'join_order',
                                                                 stratified_workloads=False)
            if eval_access_path:  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "access_path_selection",
                                                                 stratified_workloads=False)
            if eval_physical_operator:  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "physical_operator",
                                                                 stratified_workloads=False)
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "physical_operator_w_idx",
                                                                 stratified_workloads=False)

            statistics_path = target_model.get_statistics(source_path=ClusterPaths(),
                                                          dataset=target_dataset,
                                                          format=target_model.input_format)
            statistics_path = '/mnt/labstore/SIGs/ML/learned_db/runs_roman/parsed_plans/statistics_complex_workload_combined.json'

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

                if qo_eval_after_each_epoch:
                    command += " --qo_eval_after_each_epoch"

                commands.append(command)

    return commands


if __name__ == '__main__':
    install_venv = False
    install_rye = True

    setup_steps = [
        RsyncCurrentRepo(gitignore_path='.gitignore'),
    ]

    if install_venv:
        setup_steps.append(SetupVenv(use_requirements_txt=True,
                                     requirements_txt_filename=f'{ClusterPaths().Models.src}/requirements.txt',
                                     force=False,
                                     ))
    if install_rye:
        setup_steps.append(SetupRYEVenv())

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().Data.runs], dest=[LocalPaths().data], put=False, update=True)]

    datasets = [
        Dataset('airline', max_no_joins=5),
        Dataset('imdb'),
        Dataset('ssb', max_no_joins=3),
        Dataset('tpc_h', max_no_joins=5),
        Dataset('walmart', max_no_joins=2),
        Dataset('financial', scale=4),
        Dataset('basketball', scale=200),
        Dataset('accidents', scale=1, contain_unicode=True),
        Dataset('movielens', scale=8),
        Dataset('baseball', scale=10),
        Dataset('hepatitis', scale=2000),
        Dataset('tournament', scale=50),
        Dataset('credit', scale=5),
        Dataset('employee', scale=3),
        Dataset('consumer', scale=6),
        Dataset('geneea', scale=23, contain_unicode=True),
        Dataset('genome', scale=6),
        Dataset('carcinogenesis', scale=674),
        Dataset('seznam', scale=2),
        Dataset('fhnk', scale=2)
    ]

    seeds = [43]
    nodes = [
        # TrainingServers.NODE00,
        # TrainingServers.NODE02,
        TrainingServers.NODE03,
        # TrainingServers.NODE04,
        # TrainingServers.NODE05,
    ]

    # ===================
    # CONFIGS
    # ===================
    quick_run = False  # if True, it will run only a few epochs on a few workloads
    pt_profiler = False  # Profiler for PyTorch
    test_against = 'imdb'  # if not set, will test against all in an leave one out style
    ft_with_ranking = False  # fine-tune with ranking (pretrains a model and then fine-tunes it with ranking)
    bound_predictions = False  # bound predictions to the range of the training data and output them as separate metrics - at the moment not competitive
    eval_join_order = True  # evaluate join order using JOB-Light runs with exhaustive join orders
    eval_access_path = True  # evaluate access path selection (Idx Scan vs. SeqScan)
    eval_physical_operator = True  # evaluate physical operator selection
    qo_eval_after_each_epoch = True  # if True, will run a full QO evaluation after each epoch
    # ===================

    # Standard Models
    model_configs = [
        # ScaledPostgresModelConfig(),  # --> Working
        # FlatModelConfig(),  # --> Working
        # E2EModelConfig(),                   # --> ToDo: Adapt to new codebase and generate sentences
        # QPPNetModelConfig(),                # --> Working?
        # MSCNModelConfig(),                  # --> ToDo: Adapt to new codebase and generate sentences
        # DACEModelConfig(),                  # --> Working
        # QueryFormerModelConfig(),           # --> ToDo: Generate Sentences
        ZeroShotModelConfig(),  # --> Working
        # ZSLearnFromTimeoutConfig(),         # --> Working
        # ZSRankLossConfig(),                 # --> Working

        # Stratified Models
        # FlatStratifiedModelConfig(),  # --> Working
        # ZSStratifiedModelConfig(),  # --> Working
        # DACEStratifiedModelConfig(),        # --> Not yet working
        # ScaledPostgresStratifiedModelConfig(),  # --> Working

        # Ablation Models (without postgres Estimates)
        # DACEModelNoCostsConfig(),
        # QPPModelNoCostsConfig(),

        # Ablation Models (actual cardinalities)
        # FlatModelActCardModelConfig(),
        # QPPModelActCardsConfig(),
        # DACEModelActCardConfig(),
        # ZeroShotModelActCardConfig(),
    ]

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"] for node in nodes],
                       python_version='rye',
                       available_sockets=[0])

    # for dataset in datasets:
    eval_commands = generate_training_commands(datasets=datasets,
                                               model_list=model_configs,
                                               python_version='',
                                               seeds=seeds,
                                               wandb_project=runner.wandb_project,
                                               wandb_entity=runner.wandb_entity,
                                               quick_run=quick_run,
                                               pt_profiler=pt_profiler,
                                               test_against=test_against,
                                               ft_with_ranking=ft_with_ranking,
                                               bound_predictions=bound_predictions,
                                               eval_join_order=eval_join_order,
                                               eval_access_path=eval_access_path,
                                               eval_physical_operator=eval_physical_operator,
                                               qo_eval_after_each_epoch=qo_eval_after_each_epoch)

    runner.run_exp(commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   offset=0)
