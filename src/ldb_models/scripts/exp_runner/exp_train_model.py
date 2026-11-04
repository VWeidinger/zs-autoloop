import os

# add to python path
from functools import lru_cache
from typing import List, Optional

import wandb
from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import (
    KillAllScreens,
    Rsync,
    RsyncCurrentRepo,
    SetupUVVenv,
    SetupVenv,
    Step,
)

from ldb_models.classes.configs.base import ModelConfig, shorten_model_name
from ldb_models.classes.model_config_helper import parameterized_config
from ldb_models.scripts.exp_runner.exp_runner import ExpRunner, TrainingServers
from ldb_models.training.featurizations import (
    MambaFullWIdxCor,
    PostgresEstSystemCardDetail,
    PostgresEstWithNumLoops,
)


@lru_cache(maxsize=None)
def get_wandb_api_runs(wandb_project: str):
    # retrieve wandb run id
    api = wandb.Api()
    runs = api.runs(f"{wandb_project}", per_page=1000)
    return runs


def generate_training_commands(
    datasets: List[Dataset],
    version: int,
    seeds: list[int],
    python_version: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    model_list: list[ModelConfig] = [],
    quick_run: bool = False,
    pt_profiler: bool = False,
    test_against: str | None = None,
    bound_predictions: bool = False,
    eval_join_order: bool = False,
    eval_access_path: bool = False,
    eval_physical_operator: bool = False,
    qo_eval_after_each_epoch: bool = False,
    use_romans_runs: bool = False,
    is_pretrained: bool = False,
    update_wandb_summary: bool = False,
    assemble_data_up_front: bool = False,
    enable_checkpoint_loading: bool = False,
) -> List[str]:
    if update_wandb_summary:
        assert is_pretrained, "update_wandb_summary only works with pretrained models"

    if test_against:
        print(f"Testing the models against {test_against}")

    commands = []
    for target_dataset in datasets:
        if test_against is not None and target_dataset.db_name != test_against:
            continue

        for target_model in model_list:
            if is_pretrained:
                training_runs = None
            else:
                training_runs = " ".join(
                    target_model.get_training_workloads(
                        training_datasets=datasets,
                        target_dataset=target_dataset,
                        stratified_workload=target_model.stratified_workloads,
                        romans_runs=use_romans_runs,
                    )
                )

            test_runs = target_model.get_test_workloads(
                target_dataset, romans_runs=use_romans_runs
            )
            if eval_join_order:  # and target_dataset.db_name in ['imdb']:
                # JOB-light
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "join_order",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )
                # JOB
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "job_w_optimal_w_idxs",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )

                # JOB-Complex
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "job_complex_w_optimal_w_idxs",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )
                # addition for experiment for Mihail [Johannes]
                # test_runs += " " + target_model.assemble_wl_path(
                #     Dataset("imdb"),
                #     "job_light_subplans",
                #     stratified_workloads=False,
                #     romans_runs=use_romans_runs,
                # )

            if (
                eval_access_path
            ):  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                # AP without aggregate
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "access_path_selection",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )
                # AP with aggregate
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "access_path_selection_with_aggregate",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )

            if (
                eval_physical_operator
            ):  # and target_dataset.db_name in ['imdb', 'tpc_h', 'baseball']:
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "physical_operator",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "physical_operator_w_idx",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )

            if use_romans_runs:
                statistics_path = "/mnt/labstore/SIGs/ML/learned_db/runs_roman/parsed_plans/statistics_complex_workload_combined.json"
            else:
                # Get feature stats suffix from the model config
                feature_stats_suffix = target_model.get_feature_stats_suffix()
                statistics_path = target_model.get_statistics(
                    source_path=ClusterPaths(),
                    dataset=target_dataset,
                    format=target_model.input_format,
                    suffix=feature_stats_suffix,
                ).as_posix()

            # model name
            name = target_model.name.NAME

            # extract model_type information
            if name.startswith("pm-"):
                model_type = name[3:].split("-")[0]
            else:
                model_type = name.split("-")[0]
            if model_type.startswith("postgres"):
                model_type = "postgres"
            elif model_type.startswith("zeroshot"):
                model_type = "zeroshot"
            elif model_type.startswith("zs"):
                model_type = "zeroshot"
            elif model_type.startswith("dace"):
                model_type = "dace"
            elif model_type.startswith("flat"):
                model_type = "flat"
            elif model_type.startswith("t3"):
                model_type = "t3"
            elif model_type.startswith("neo"):
                model_type = "neo"
            elif model_type.startswith("qppnet"):
                model_type = "qppnet"
            elif model_type.startswith("mscn"):
                model_type = "mscn"
            elif model_type.startswith("e2e"):
                model_type = "e2e"
            elif model_type.startswith("query_former"):
                model_type = "query_former"
            elif model_type.startswith("mamba"):
                model_type = "mamba"
            elif model_type.startswith("spn"):
                model_type = "spn"
            else:
                raise ValueError(f"Model type {model_type} not recognized")

            for seed in seeds:
                model_suffix = ""

                if bound_predictions:
                    model_suffix += "_bound"
                if use_romans_runs:
                    model_suffix += "_romans"

                if "zeroshot" in name:
                    name_short = name.replace("zeroshot", "zs")
                else:
                    name_short = name

                wandb_name = f"{name_short}{model_suffix}/{target_dataset.db_name}/{seed}v{version}"

                # shorten wandb name
                wandb_name = shorten_model_name(wandb_name)

                if update_wandb_summary:
                    # remove pm_ if existing
                    if wandb_name.startswith("pm-"):  # parameterized model
                        tmp_name = wandb_name[3:]
                    else:
                        tmp_name = wandb_name

                    runs = get_wandb_api_runs(wandb_project)

                    runs = [run for run in runs if run.name == tmp_name]
                    run_keys = [r.id for r in runs]

                    assert len(run_keys) == 1, (
                        f"Found {len(run_keys)} runs with name {tmp_name}. Expected 1. Please make sure only one run with this name exists."
                    )
                    wandb_resume_id = run_keys[0]
                else:
                    wandb_resume_id = None

                if is_pretrained:
                    wandb_name += "_pretrained"

                command = (
                    f"python{python_version} main.py "
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
                    f"--version {version} "
                    f"--wandb_name {wandb_name} "
                )

                if wandb_resume_id is not None:
                    command += f" --wandb_resume_id {wandb_resume_id} "

                if training_runs is not None and len(training_runs) > 0:
                    command += f" --workload_runs {training_runs} "

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

                if bound_predictions:
                    command += " --bound_predictions"

                if qo_eval_after_each_epoch:
                    command += " --qo_eval_after_each_epoch"

                if is_pretrained:
                    command += " --is_pretrained"

                if target_model.pretrained_model_dir is not None:
                    command += (
                        f" --pretrained_model_dir {target_model.pretrained_model_dir} "
                    )
                    command += f" --pretrained_model_name {target_model.pretrained_model_name} "

                if assemble_data_up_front:
                    command += " --assemble_data_up_front"

                if enable_checkpoint_loading:
                    command += " --enable_checkpoint_loading"

                commands.append(command)

    return commands


if __name__ == "__main__":
    install_venv = False
    install_rye = True

    if os.getenv("SSH_PATH") is not None:
        ssh_path = os.getenv("SSH_PATH")
        assert ssh_path is not None, "SSH_PATH environment variable is not set"
    else:
        ssh_path = "ssh"

    setup_steps = []
    setup_steps.append(
        RsyncCurrentRepo(
            gitignore_path=".gitignore",
            local_folder_path=LocalPaths().Models.root.as_posix(),
            ssh_path=ssh_path,
        )
    )

    if install_venv:
        setup_steps.append(
            SetupVenv(
                use_requirements_txt=True,
                requirements_txt_filename=f"{ClusterPaths().Models.src}/requirements.txt",
                force=False,
            )
        )
    if install_rye:
        setup_steps.append(SetupUVVenv())

    purge_steps: list[Step] = [KillAllScreens()]
    pickup_steps: list[Step] = [
        Rsync(
            src=[ClusterPaths().Data.runs.as_posix()],
            dest=[LocalPaths().data],
            put=False,
            update=True,
            ssh_path=ssh_path,
        )
    ]

    datasets = [
        Dataset("airline", max_no_joins=5),
        Dataset("imdb"),
        Dataset("ssb", max_no_joins=3),
        Dataset("tpc_h", max_no_joins=5),
        Dataset("walmart", max_no_joins=2),
        Dataset("financial", scale=4),
        Dataset("basketball", scale=200),
        Dataset("accidents", scale=1, contain_unicode=True),
        Dataset("movielens", scale=8),
        Dataset("baseball", scale=10),
        Dataset("hepatitis", scale=2000),
        Dataset("tournament", scale=50),
        Dataset("credit", scale=5),
        Dataset("employee", scale=3),
        Dataset("consumer", scale=6),
        Dataset("geneea", scale=23, contain_unicode=True),
        Dataset("genome", scale=6),
        Dataset("carcinogenesis", scale=674),
        Dataset("seznam", scale=2),
        Dataset("fhnk", scale=2),
    ]

    version = 5  # 3=parsed_plans2, 4=fixed card-annotation of optimal plans in JOB & -Complex and improved logging/plotting, 5=fixed parser error (new feature statistics) 10/2025)
    seeds = [9]
    nodes = [
        # TrainingServers.NODE01,
        # TrainingServers.NODE02,
        # TrainingServers.NODE03,
        TrainingServers.NODE03,
        # TrainingServers.NODE04,
        # TrainingServers.NODE05,
        # TrainingServers.NODE06,
        # TrainingServers.NODE00,
        # TrainingServers.NODE09,
    ]
    sockets = [0, 1]
    out_file_id_offset = 0
    num_parallel_screens = len(sockets)
    num_parallel_jobs_dgx = 1
    num_parallel_tasks_per_job_dgx = 1
    run_exp_pin = False

    # ===================
    # CONFIGS
    # ===================
    quick_run = False  # if True, it will run only a few epochs on a few workloads
    pt_profiler = False  # Profiler for PyTorch
    test_against = "imdb"  # if not set, will test against all in an leave one out style
    bound_predictions = False  # bound predictions to the range of the training data and output them as separate metrics - at the moment not competitive

    # training / test data: 3 = parsed_plans3, 50 = parsed_plans50 (parsed_plans3 + JIT information, no stratified copy
    # - the test workload is the full workload_200k_s1)
    parsed_plans_version = 50
    # encode the JIT information as plan node features (zeroshot based models, requires parsed_plans_version 50)
    use_jit_features = True

    # evaluate join order using JOB-Light runs with exhaustive join orders
    eval_join_order = True and not quick_run

    # evaluate access path selection (Idx Scan vs. SeqScan)
    eval_access_path = False  # TODO workaround always failes for parsed plans 3

    # evaluate physical operator selection
    # eval_physical_operator = True and not quick_run
    eval_physical_operator = False  # TODO workaround always failes for parsed plans 3

    # if True, will run a full QO evaluation after each epoch
    qo_eval_after_each_epoch = True

    use_romans_runs = False
    assemble_data_up_front = False  # run data / graph assemble up front i.e. during dataset creation instead of collate / training time

    enable_checkpoint_loading = (
        False  # if True, dont load checkpoint (excluding specific pretrain path set)
    )

    # =================== (pretrained / wandb reuse configs)
    is_pretrained = False  # if True, assumes that the model is pretrained and does not train it # noqa: E501
    update_wandb_summary = False  # (works only together with pretrained) if True, it will update the wandb summaries of the first run with this layout # noqa: E501
    # ===================

    # generate model configs

    model_configs = []

    for model_type in [
        "zeroshot",
        # "postgres",
        # "zeroshot",
        # "dace",
        # "flat",
        # "t3",
        # "zeroshot_vae",
        # "neo",
        # "qppnet",
        # "mscn",
        # "e2e",
        # "query_former",
        # "mamba",
        # "spn",
        # # "t3_act",
        # # "mamba_tanh",
        # # "mamba_relu",
        # # "zeroshot_act_card",
        # # 'zeroshot_low_min',
        # # 'zeroshot_xl',
        # # 'zeroshot_timeout',
    ]:
        zs_featurizations = [PostgresEstSystemCardDetail, PostgresEstWithNumLoops]

        if model_type == "dace":
            model_specific_args = {"pad_length": 60}
        elif model_type in ["postgres", "t3", "mscn", "flat", "spn"]:
            model_specific_args = {"epochs": 80}
        elif model_type == "mamba":
            # from ldb_models.training.featurizations import MambaFullWIdxCor
            model_specific_args = {
                "featurization": MambaFullWIdxCor,
                "scan_nte_relu": True,
                "filter_condense_operators": True,
                "layer_norm": True,
            }
        else:
            model_specific_args = {}

        # only part of the model name if not the default (keeps the names of existing parsed_plans3 runs)
        data_args = {}
        if parsed_plans_version != 3:
            data_args["parsed_plans_version"] = parsed_plans_version
        if use_jit_features:
            data_args["use_jit_features"] = True
        wandb_group = "jit" if use_jit_features else "baseline"

        """for featurization in [PostgresEstSystemCardDetail, PostgresEstWithNumLoops]:
            model_specific_args["featurization"] = featurization"""
        # model_configs.extend(
        #     [
        #         parameterized_config(
        #             model_type,
        #             include_pg_opt_workloads=True,
        #             subplan_monotonicity_loss=True,
        #             epochs=2,
        #             wandb_group="monotonicity",
        #             **model_specific_args,
        #         ),
        #         parameterized_config(
        #             model_type,
        #             include_hint_idx_workloads=True,
        #             subplan_monotonicity_loss=True,
        #             epochs=2,
        #             wandb_group="monotonicity",
        #             **model_specific_args,
        #         ),
        #     ]
        # )
        model_configs.extend(
            [
                parameterized_config(
                    model_type,
                    include_pg_opt_workloads=True,
                    reproducible_message_passing=True,
                    epochs=200,
                    wandb_group=wandb_group,
                    **data_args,
                    **model_specific_args,
                ),
                parameterized_config(
                    model_type,
                    include_pg_opt_workloads=True,
                    skip_top_aggregation=True,
                    reproducible_message_passing=True,
                    epochs=200,
                    wandb_group=wandb_group,
                    **data_args,
                    **model_specific_args,
                ),
            ]
        )

    # Try to load model with name pm-zs_sop_rl-stable-gs8-bgs32-regrW1-rankW1-subpW1-s10-ld5-sLNFalse-hiFalse-cr-mjFalse and seed 5 from /mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-stable-gs8-bgs32-regrW1-rankW1-subpW1-s10-ld5-sLNFalse-hiFalse-cr-mjFalse/imdb
    # No valid checkpoint found [Errno 2] No such file or directory: '/mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-stable-gs8-bgs32-regrW1-rankW1-subpW1-s10-ld5-sLNFalse-hiFalse-cr-mjFalse/imdb/pm-zs_sop_rl-stable-gs8-bgs32-regrW1-rankW1-subpW1-s10-ld5-sLNFalse-hiFalse-cr-mjFalse_5.pt'

    # pprint(asdict(model_configs[0]), sort_dicts=False)
    # model_configs = list(model_configs[:2])  # for testing only use the first 2 configs

    model_configs = [model for model in model_configs if model is not None]

    print(f"Total number of models: {len(model_configs)}")

    runner = ExpRunner(
        replicate=False,
        node_names=[node["hostname"] for node in nodes],
        python_version="uv",
        available_sockets=sockets,
        num_parallel_screens_or_jobs=min(num_parallel_screens, len(model_configs)),
        dgx_num_parallel_screens_or_jobs=num_parallel_jobs_dgx,
        ssh_path=ssh_path,
        stop_on_error=False,
        dgx_num_parallel_tasks_per_job=num_parallel_tasks_per_job_dgx,
    )

    # for dataset in datasets:
    eval_commands = generate_training_commands(
        datasets=datasets,
        model_list=model_configs,
        python_version="",
        version=version,
        seeds=seeds,
        wandb_project=runner.wandb_project,
        wandb_entity=runner.wandb_entity,
        quick_run=quick_run,
        pt_profiler=pt_profiler,
        test_against=test_against,
        bound_predictions=bound_predictions,
        eval_join_order=eval_join_order,
        eval_access_path=eval_access_path,
        eval_physical_operator=eval_physical_operator,
        qo_eval_after_each_epoch=qo_eval_after_each_epoch,
        use_romans_runs=use_romans_runs,
        is_pretrained=is_pretrained,
        update_wandb_summary=update_wandb_summary,
        assemble_data_up_front=assemble_data_up_front,
        enable_checkpoint_loading=enable_checkpoint_loading,
    )

    runner.run_exp(
        commands=eval_commands,
        set_up_steps=setup_steps,
        purge_steps=purge_steps,
        pickup_steps=pickup_steps,
        offset=out_file_id_offset,
        run_exp_pin=run_exp_pin,
    )
