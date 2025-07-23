import os

# add to python path
import sys
from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import (
    KillAllScreens,
    Rsync,
    RsyncCurrentRepo,
    SetupRYEVenv,
    SetupVenv,
)

import wandb

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from classes.classes import ModelConfig, shorten_model_name
from classes.model_config_helper import parameterized_config
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers
from training.featurizations import (
    MambaFull,
    MambaFullWIdxCor2,
)


def generate_training_commands(
    datasets: List[Dataset],
    seeds: list[int],
    python_version: str,
    wandb_project: str,
    wandb_entity: str,
    model_list: list[ModelConfig] = ModelConfig,
    quick_run: bool = False,
    pt_profiler: bool = False,
    test_against: str = None,
    ft_with_ranking: bool = False,
    bound_predictions: bool = False,
    eval_join_order: bool = False,
    eval_access_path: bool = False,
    eval_physical_operator: bool = False,
    qo_eval_after_each_epoch: bool = False,
    use_romans_runs: bool = False,
    is_pretrained: bool = False,
    update_wandb_summary: bool = False,
    assemble_data_up_front: bool = False,
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
                    "join_order_job_w_idxs_combined",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )

                # JOBExtended
                test_runs += " " + target_model.assemble_wl_path(
                    Dataset("imdb"),
                    "JOBExtended_selection1",
                    stratified_workloads=False,
                    romans_runs=use_romans_runs,
                )

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
                statistics_path = target_model.get_statistics(
                    source_path=ClusterPaths(),
                    dataset=target_dataset,
                    format=target_model.input_format,
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

                if ft_with_ranking:
                    model_suffix += "_ft_rank"
                if bound_predictions:
                    model_suffix += "_bound"
                if use_romans_runs:
                    model_suffix += "_romans"

                if "zeroshot" in name:
                    name_short = name.replace("zeroshot", "zs")
                else:
                    name_short = name

                wandb_name = (
                    f"{name_short}{model_suffix}/{target_dataset.db_name}/{seed}"
                )

                # shorten wandb name
                wandb_name = shorten_model_name(wandb_name)

                if update_wandb_summary:
                    # remove pm_ if existing
                    if wandb_name.startswith("pm-"):  # parameterized model
                        tmp_name = wandb_name[3:]
                    else:
                        tmp_name = wandb_name

                    # retrieve wandb run id
                    api = wandb.Api()
                    runs = api.runs(f"{wandb_project}", per_page=1000)
                    runs = [run for run in runs if run.name == tmp_name]
                    run_keys = [r.id for r in runs]

                    assert len(run_keys) == 1, (
                        f"Found {len(run_keys)} runs with name {wandb_name}. Expected 1. Please make sure only one run with this name exists."
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

                if ft_with_ranking:
                    command += " --ft_with_ranking"

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

                commands.append(command)

    return commands


if __name__ == "__main__":
    install_venv = False
    install_rye = True

    if os.getenv("SSH_PATH") is not None:
        ssh_path = os.getenv("SSH_PATH")
    else:
        ssh_path = "ssh"

    setup_steps = [
        RsyncCurrentRepo(
            gitignore_path=".gitignore",
            local_folder_path=LocalPaths().Models.root.as_posix(),
            ssh_path=ssh_path,
        ),
    ]

    if install_venv:
        setup_steps.append(
            SetupVenv(
                use_requirements_txt=True,
                requirements_txt_filename=f"{ClusterPaths().Models.src}/requirements.txt",
                force=False,
            )
        )
    if install_rye:
        setup_steps.append(SetupRYEVenv(no_lock=False))

    purge_steps = [KillAllScreens()]
    pickup_steps = [
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

    seeds = [3]
    nodes = [
        TrainingServers.NODE00,
        # TrainingServers.NODE01,
        # TrainingServers.NODE02,
        # TrainingServers.NODE03,
        # TrainingServers.NODE04,
        # TrainingServers.NODE05,
        # TrainingServers.NODE09,
    ]
    sockets = [0, 1]
    out_file_id_offset = 4
    num_parallel_screens = len(sockets)
    num_parallel_jobs_dgx = 8
    num_parallel_tasks_per_job_dgx = 1
    run_exp_pin = False

    # ===================
    # CONFIGS
    # ===================
    quick_run = False  # if True, it will run only a few epochs on a few workloads
    pt_profiler = False  # Profiler for PyTorch
    test_against = "imdb"  # if not set, will test against all in an leave one out style
    ft_with_ranking = False  # fine-tune with ranking (pretrains a model and then fine-tunes it with ranking)
    bound_predictions = False  # bound predictions to the range of the training data and output them as separate metrics - at the moment not competitive

    # evaluate join order using JOB-Light runs with exhaustive join orders
    eval_join_order = False and not quick_run

    # evaluate access path selection (Idx Scan vs. SeqScan)
    eval_access_path = True and not quick_run

    # evaluate physical operator selection
    eval_physical_operator = False and not quick_run

    # if True, will run a full QO evaluation after each epoch
    qo_eval_after_each_epoch = True

    use_romans_runs = False
    assemble_data_up_front = True  # run data / graph assemble up front i.e. during dataset creation instead of collate / training time

    # =================== (pretrained / wandb reuse configs)
    is_pretrained = False  # if True, assumes that the model is pretrained and does not train it # noqa: E501
    update_wandb_summary = False  # (works only together with pretrained) if True, it will update the wandb summaries of the first run with this layout # noqa: E501
    # ===================

    # generate model configs
    model_configs = []
    for model_type in [
        # 'postgres',
        # 'zeroshot',
        # 'zeroshot_low_min',
        # 'zeroshot_xl',
        # 'zeroshot_timeout',
        # 'dace',
        # 'flat',
        # 't3',
        # 'zeroshot_vae',
        # 'neo',
        # 'qppnet',
        # 'mscn',
        # 'e2e',
        # 'query_former'
        "mamba"
        # "mamba_tanh",
        # "mamba_relu",
        #'query_former'
        #'spn',
    ]:
        if model_type == "dace":
            model_specific_args = {"pad_length": 60}
        else:
            model_specific_args = {}

        for featurization in [
            MambaFull,
            # MambaFullWIdx,
            # MambaFullWIdxCor,
            MambaFullWIdxCor2,
        ]:
            for loss_fn in [
                # "QLogSqLoss",
                "MSELoss",
                # "LogQLoss",
                # "LogHuberQLoss",
                # "LogQLossPenalty",
            ]:
                model_configs.extend(
                    [
                        parameterized_config(
                            model_type,
                            wandb_group="ap_experiments",
                            layer_norm=True,
                            include_scan_only_hint_workloads_stratified=True,
                            featurization=featurization,
                            use_dcn=True,
                            loss_class_name=loss_fn,
                            epochs=75,
                            **model_specific_args,
                        )
                        if model_type.startswith("mamba")
                        else None,
                    ]
                )

    model_configs = [model for model in model_configs if model is not None]

    runner = ExpRunner(
        replicate=False,
        node_names=[node["hostname"] for node in nodes],
        python_version="rye",
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
        is_pretrained=is_pretrained,
        update_wandb_summary=update_wandb_summary,
        assemble_data_up_front=assemble_data_up_front,
    )

    runner.run_exp(
        commands=eval_commands,
        set_up_steps=setup_steps,
        purge_steps=purge_steps,
        pickup_steps=pickup_steps,
        offset=out_file_id_offset,
        run_exp_pin=run_exp_pin,
    )
