import os
from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens, SetupVenv, RsyncCurrentRepo, SetupRYEVenv

from classes.classes import ModelConfig, ModelType
from classes.model_config_helper import parameterized_config
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers
from training.featurizations import PostgresEstWithIsIndexCond


def generate_training_commands(datasets: List[Dataset], seeds: list[int], python_version: str, wandb_project: str,
                               wandb_entity: str,
                               model_list: list[ModelConfig] = ModelConfig, quick_run: bool = False,
                               pt_profiler: bool = False, test_against: str = None,
                               ft_with_ranking: bool = False, bound_predictions: bool = False,
                               eval_join_order: bool = False, eval_access_path: bool = False,
                               eval_physical_operator: bool = False,
                               qo_eval_after_each_epoch: bool = False,
                               use_romans_runs: bool = False,
                               is_pretrained: bool = False) -> List[str]:
    if test_against:
        print(f'Testing the models against {test_against}')

    commands = []
    for target_dataset in datasets:
        if test_against is not None and target_dataset.db_name != test_against:
            continue

        for target_model in model_list:
            if is_pretrained:
                training_runs = None
            else:
                training_runs = " ".join(
                    target_model.get_training_workloads(training_datasets=datasets,
                                                        target_dataset=target_dataset,
                                                        stratified_workload=target_model.stratified_workloads,
                                                        romans_runs=use_romans_runs))

            test_runs = target_model.get_test_workloads(target_dataset, romans_runs=use_romans_runs)
            if eval_join_order:  # and target_dataset.db_name in ['imdb']:
                # JOB-light
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), 'join_order',
                                                                 stratified_workloads=False,
                                                                 romans_runs=use_romans_runs)
                # JOB
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), 'join_order_job_w_idxs_combined',
                                                                 stratified_workloads=False,
                                                                 romans_runs=use_romans_runs)
            if eval_access_path:  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "access_path_selection",
                                                                 stratified_workloads=False,
                                                                 romans_runs=use_romans_runs)
            if eval_physical_operator:  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "physical_operator",
                                                                 stratified_workloads=False,
                                                                 romans_runs=use_romans_runs)
                test_runs += " " + target_model.assemble_wl_path(Dataset('imdb'), "physical_operator_w_idx",
                                                                 stratified_workloads=False,
                                                                 romans_runs=use_romans_runs)

            if use_romans_runs:
                statistics_path = '/mnt/labstore/SIGs/ML/learned_db/runs_roman/parsed_plans/statistics_complex_workload_combined.json'
            else:
                statistics_path = target_model.get_statistics(source_path=ClusterPaths(),
                                                              dataset=target_dataset,
                                                              format=target_model.input_format).as_posix()

            # model name
            name = target_model.name.NAME

            # extract model_type information
            if name.startswith('pm-'):
                model_type = name[3:].split('-')[0]
            else:
                model_type = name.split('-')[0]
            if model_type.startswith('postgres'):
                model_type = 'postgres'
            elif model_type.startswith('zeroshot'):
                model_type = 'zeroshot'
            elif model_type.startswith('dace'):
                model_type = 'dace'
            elif model_type.startswith('flat'):
                model_type = 'flat'
            elif model_type.startswith('t3'):
                model_type = 't3'
            else:
                raise ValueError(f"Model type {model_type} not recognized")

            for seed in seeds:
                model_suffix = ''

                if ft_with_ranking:
                    model_suffix += '_ft_rank'
                if bound_predictions:
                    model_suffix += '_bound'
                if use_romans_runs:
                    model_suffix += '_romans'

                if 'zeroshot' in name:
                    name_short = name.replace('zeroshot', 'zs')
                else:
                    name_short = name

                wandb_name = f"{name_short}{model_suffix}/{target_dataset.db_name}/{seed}"

                if is_pretrained:
                    wandb_name += '_pretrained'

                # shorten wandb name
                wandb_name = wandb_name.replace('include_complex_workloads', 'complex')
                wandb_name = wandb_name.replace('include_pg_opt_workloads', 'pgopt')
                wandb_name = wandb_name.replace('include_hint_workloads', 'hint')
                wandb_name = wandb_name.replace('include_hint_idx_workloads', 'hintidx')
                wandb_name = wandb_name.replace('include_scan_only_hint_workloads', 'scanhint')
                wandb_name = wandb_name.replace('train_on_subplans', 'subplans')
                wandb_name = wandb_name.replace('PostgresEstWithIsIndexCond', 'FeatPGEstIsIdxC')
                wandb_name = wandb_name.replace('limit_plans_per_file', 'limitpf')
                wandb_name = wandb_name.replace('PostgresEstWithIsJoinCond', 'isjoincond')
                wandb_name = wandb_name.replace('filter_inline_col_featsFalse', 'nofiltercolinline')
                wandb_name = wandb_name.replace('AdamWScheduleFree', 'AdamWSF')
                wandb_name = wandb_name.replace('annotate_base_table_card_to_scan_op', 'tablecardatscan')

                command = (f"python{python_version} main.py "
                           f"--mode train "
                           f"--wandb_project {wandb_project} "
                           f"--wandb_entity {wandb_entity} "
                           f"--wandb_group {target_model.wandb_group} "
                           f"--model_type {name} "
                           f"--device {target_model.device} "
                           f"--model_dir {target_model.get_model_dir(ClusterPaths(), target_dataset).as_posix()} "
                           f"--target_dir {target_model.get_eval_dir(ClusterPaths(), target_dataset).as_posix()} "
                           f"--statistics_file {statistics_path} "
                           f"--seed {seed} "
                           f"--wandb_name {wandb_name} "
                           )

                if training_runs is not None and len(training_runs) > 0:
                    command += f" --workload_runs {training_runs} "

                if target_model.type == ModelType.WL_AGNOSTIC:
                    command += f" --test_workload_runs {test_runs} "

                if target_model.column_statistics:
                    command += f" --column_statistics {target_model.get_column_stats(target_dataset).as_posix()} "

                if target_model.word_embeddings:
                    command += f" --word_embeddings {target_model.get_word_embeddings(source_path=ClusterPaths(), database=target_dataset).as_posix()} "

                if target_model.hyperparameter:
                    command += f" --hyperparameter_path {target_model.hyperparameter.as_posix()} "

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

                if is_pretrained:
                    command += " --is_pretrained"

                commands.append(command)

    return commands


if __name__ == '__main__':
    install_venv = False
    install_rye = True

    if os.getenv('SSH_PATH') is not None:
        ssh_path = os.getenv('SSH_PATH')
    else:
        ssh_path = None

    setup_steps = [
        RsyncCurrentRepo(gitignore_path='.gitignore', local_folder_path=LocalPaths().Models.root.as_posix(),
                         ssh_path=ssh_path),
    ]

    if install_venv:
        setup_steps.append(SetupVenv(use_requirements_txt=True,
                                     requirements_txt_filename=f'{ClusterPaths().Models.src}/requirements.txt',
                                     force=False,
                                     ))
    if install_rye:
        setup_steps.append(SetupRYEVenv(no_lock=False))

    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().Data.runs.as_posix()], dest=[LocalPaths().data], put=False, update=True,
                          ssh_path=ssh_path)]

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

    seeds = [1]
    nodes = [
        # TrainingServers.NODE00,
        TrainingServers.NODE02,
        # TrainingServers.NODE03,
        # TrainingServers.NODE04,
        # TrainingServers.NODE05,
        # TrainingServers.NODE09,
    ]
    sockets = [0, 1]
    out_file_id_offset = 0
    num_parallel_screens = None
    num_parallel_jobs_dgx = 1

    # ===================
    # CONFIGS
    # ===================
    quick_run = False  # if True, it will run only a few epochs on a few workloads
    pt_profiler = False  # Profiler for PyTorch
    test_against = 'imdb'  # if not set, will test against all in an leave one out style
    ft_with_ranking = False  # fine-tune with ranking (pretrains a model and then fine-tunes it with ranking)
    bound_predictions = False  # bound predictions to the range of the training data and output them as separate metrics - at the moment not competitive
    eval_join_order = True and not quick_run  # evaluate join order using JOB-Light runs with exhaustive join orders
    eval_access_path = True and not quick_run  # evaluate access path selection (Idx Scan vs. SeqScan)
    eval_physical_operator = True and not quick_run  # evaluate physical operator selection
    qo_eval_after_each_epoch = True  # if True, will run a full QO evaluation after each epoch
    use_romans_runs = False
    is_pretrained = False  # if True, assumes that the model is pretrained and does not train it
    # ===================

    # generate model configs
    model_configs = []
    for model_type in [
        # 'postgres',
        'zeroshot',
        # 'dace',
        # 'flat',
        # 't3'
    ]:
        if model_type == 'dace':
            model_specific_args = {'pad_length': 60}
        else:
            model_specific_args = {}

        model_configs.extend([
            # parameterized_config(model_type, wandb_group='baselines', include_pg_opt_workloads=True,
            #                      **model_specific_args),  # std
            # parameterized_config(model_type, wandb_group='whintwl', include_hint_workloads=True,
            #                      include_pg_opt_workloads=True, **model_specific_args),  # hint
            # # parameterized_config(model_type, wandb_group='whint_hintidx', include_hint_workloads=True, include_hint_idx_workloads=True, **model_specific_args),  # hint + idx
            # parameterized_config(model_type, wandb_group='whintidx', include_hint_idx_workloads=True,
            #                      include_pg_opt_workloads=True,
            #                      **model_specific_args),  # hint-idx
            # # # parameterized_config(model_type, wandb_group='onlyhint', only_hint_workloads=True, **model_specific_args),  # hint_only
            # parameterized_config(model_type, wandb_group='wcomplex', include_complex_workloads=True,include_pg_opt_workloads=True,
            #                      **model_specific_args),  # complex
            # parameterized_config(model_type, wandb_group='whintidx_complex', include_complex_workloads=True,
            #                      include_pg_opt_workloads=True,
            #                      include_hint_idx_workloads=True,
            #                      **model_specific_args),  # complex + hint-idx
            # parameterized_config(model_type, wandb_group='wIsIdxFeat', featurization = PostgresEstWithIsIndexCond,include_pg_opt_workloads=True, **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wIsIdxFeat_complex_hintidxs_nopgopt', featurization=PostgresEstWithIsIndexCond, include_complex_workloads=True, include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wIsIdxFeat_complex',
            #                      featurization=PostgresEstWithIsIndexCond, include_complex_workloads=True,include_pg_opt_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wIsIdxFeat_complex_pgopt_scanhint',
            #                      featurization=PostgresEstWithIsIndexCond, include_complex_workloads=True,
            #                      include_hint_idx_workloads=True, include_scan_only_hint_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='notopaggr',include_pg_opt_workloads=True,skip_top_aggregation=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='notopaggr', include_pg_opt_workloads=True, include_scan_only_hint_workloads=True,
            #                      skip_top_aggregation=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wsubplans_pgopt_hintidx_complex',
            #                      include_complex_workloads=True,
            #                      include_pg_opt_workloads=True,
            #                      include_hint_idx_workloads=True,
            #                      train_on_subplans=True,
            #                      limit_plans_per_file=5000,
            #                      **model_specific_args),  # w subplans
            # parameterized_config(model_type, wandb_group='wIsJoinCond_complex_pgopt_hintidx',
            #                      featurization=PostgresEstWithIsJoinCond, include_complex_workloads=True,
            #                      include_pg_opt_workloads=True, include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wColFilterEdges_complex_pgopt_hintidx',
            #                      filter_inline_col_feats=False, include_complex_workloads=True,
            #                      include_pg_opt_workloads=True, include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='wColFilterEdges_wIsJoinCond_complex_pgopt_hintidx',
            #                      filter_inline_col_feats=False, featurization=PostgresEstWithIsJoinCond,
            #                      include_complex_workloads=True, include_pg_opt_workloads=True,
            #                      include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='adamwschedulefree_complex_pgopt_hintidx',
            #                      optimizer_class_name='AdamWScheduleFree', include_complex_workloads=True,
            #                      include_pg_opt_workloads=True, include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            parameterized_config(model_type, wandb_group='tablecardatscan_complex_pgopt_hintidx',
                                 annotate_base_table_card_to_scan_op=True,
                                 include_complex_workloads=True, include_pg_opt_workloads=True,
                                 include_hint_idx_workloads=True,
                                 **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='submonloss_complex_pgopt_hintidx',
            #                      subplan_monotonicity_loss=True,
            #                      include_complex_workloads=True, include_pg_opt_workloads=True,
            #                      include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
            parameterized_config(model_type, wandb_group='isidxcond_tablecardatscan_complex_pgopt_hintidx',
                                 annotate_base_table_card_to_scan_op=True,
                                 featurization=PostgresEstWithIsIndexCond,
                                 include_complex_workloads=True, include_pg_opt_workloads=True,
                                 include_hint_idx_workloads=True,
                                 **model_specific_args) if model_type == 'zeroshot' else None,
            # parameterized_config(model_type, wandb_group='numloops_complex_pgopt_hintidx',
            #                      featurization=PostgresEstWithNumLoops,
            #                      include_complex_workloads=True, include_pg_opt_workloads=True,
            #                      include_hint_idx_workloads=True,
            #                      **model_specific_args) if model_type == 'zeroshot' else None,
        ])

    model_configs = [model for model in model_configs if model is not None]

    # # Standard Models
    # model_configs = [
    # ScaledPostgresModelConfig(),  # --> Working
    # FlatModelConfig(),  # --> Working
    # E2EModelConfig(),                   # --> ToDo: Adapt to new codebase and generate sentences
    # QPPNetModelConfig(),                # --> Working?
    # MSCNModelConfig(),                  # --> ToDo: Adapt to new codebase and generate sentences
    # DACEModelConfig(),                  # --> Working
    # QueryFormerModelConfig(),           # --> ToDo: Generate Sentences
    # ZeroShotModelConfig(),  # --> Working
    # ZSLearnFromTimeoutConfig(),         # --> Working
    # ZSRankLossConfig(),                 # --> Working

    # Stratified Models
    # FlatStratifiedModelConfig(),  # --> Working
    # ZSStratifiedModelConfig(),  # --> Working
    # DACEStratifiedModelConfig(),        # --> Working
    # ScaledPostgresStratifiedModelConfig(),  # --> Working

    # ZeroShotWithPostgresConfig(),  # --> Working
    # ZeroShotModelWHintsConfig(),  # --> Working
    # ZeroShotModelOnlyHints(),
    # ZeroShotModel2Dim(),
    # ZSStratifiedPGCostWithHintPlan(),
    # ZSPGCostWithHintPlan(),
    # parameterized_config('zeroshot', include_hint_workloads=True, stratified_workloads=True),

    # ZeroShotXLConfig(),

    # Ablation Models (without postgres Estimates)
    # QPPModelNoCostsConfig(),

    # DACEModelNoCostsConfig(),
    # DACEModelWHintsConfig(),
    # DACEModelOnlyHints(),

    # Ablation Models (actual cardinalities)
    # FlatModelActCardModelConfig(),
    # QPPModelActCardsConfig(),
    # DACEModelActCardConfig(),
    # ZeroShotModelActCardConfig(),
    # T3ModelConfig(include_pg_opt_workloads=True),
    # T3TreeModelConfig(include_pg_opt_workloads=True),
    # T3FlatModelConfig(include_pg_opt_workloads=True),
    # ]

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"] for node in nodes],
                       python_version='rye',
                       available_sockets=sockets,
                       num_parallel_screens_or_jobs=num_parallel_screens,
                       dgx_num_parallel_screens_or_jobs=num_parallel_jobs_dgx,
                       ssh_path=ssh_path,
                       )

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
                                               qo_eval_after_each_epoch=qo_eval_after_each_epoch,
                                               use_romans_runs=use_romans_runs,
                                               is_pretrained=is_pretrained)

    runner.run_exp(commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=pickup_steps,
                   offset=out_file_id_offset)
