
from ldb_schema.datasets import dataset_list_dict
from ldb_schema.paths import ClusterPaths
from octopus.step import KillAllScreens, RsyncCurrentRepo, SetupUVVenv, SetupVenv

from ldb_models.scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == "__main__":
    install_venv = False
    install_rye = True

    setup_steps = []
    setup_steps.append(RsyncCurrentRepo(gitignore_path=".gitignore"))

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

    purge_steps = [KillAllScreens()]

    node = TrainingServers.NODE09

    runner = ExpRunner(
        replicate=False,
        node_names=[node["hostname"]],
        python_version="uv",
        available_sockets=[0, 1],
        num_parallel_screens_or_jobs=8,
    )

    # for dataset in datasets:

    datasets = dataset_list_dict["zs"]

    dataset_str = " ".join([d.data_folder for d in datasets])
    file_name = "feature_statistics"
    target_folder = f"{ClusterPaths().Data.root}/feature_statistics"

    parsed_plans_dirname = "parsed_plans3"

    # First generate generic feature statistics over all workloads
    suffix = "_v20260309"
    eval_commands = []
    eval_commands += [
        f"python3 gather_feature_stats.py "
        f"--database {' '.join([d.data_folder for d in datasets])} "
        f"--workload workload_200k_s1.json complex_workload_200k_s1.json complex_workload_many_joins_hints.json full_qo_complex_workload_200k_s1.json scan_only_s1_access_path_hints_stratified.json workload_200k_s1_full_hints_gt30.json workload_200k_s1_full_hints_idx.json workload_200k_s1_full_hints_lt30.json join_order.json access_path_selection.json access_path_selection_with_aggregate.json physical_operator.json physical_operator_w_idx.json job_w_optimal_w_idxs.json job_complex_w_optimal_w_idxs.json "
        f"--file_name {file_name} "
        f"--target_folder {target_folder} "
        f"--runs_dir {ClusterPaths().Data.runs} "
        f"--file_name feature_statistics{suffix} "
        f" --parsed_plans_dirname {parsed_plans_dirname}",
    ]

    # Next, generate workload-specific feature statistics
    for dataset in datasets:
        eval_commands += [
            f"python3 gather_feature_stats.py "
            f"--database {dataset.data_folder} "
            f"--workload workload_200k_s1.json complex_workload_200k_s1.json complex_workload_many_joins_hints.json full_qo_complex_workload_200k_s1.json scan_only_s1_access_path_hints_stratified.json workload_200k_s1_full_hints_gt30.json workload_200k_s1_full_hints_idx.json workload_200k_s1_full_hints_lt30.json join_order.json access_path_selection.json access_path_selection_with_aggregate.json physical_operator.json physical_operator_w_idx.json job_w_optimal_w_idxs.json job_complex_w_optimal_w_idxs.json "
            f"--file_name {file_name} "
            f"--target_folder {target_folder} "
            f"--runs_dir {ClusterPaths().Data.runs} "
            f"--file_name feature_statistics{suffix}_{dataset.db_name} "
            f"--parsed_plans_dirname {parsed_plans_dirname}",
        ]

    runner.run_exp(
        commands=eval_commands,
        set_up_steps=setup_steps,
        purge_steps=purge_steps,
        pickup_steps=[],
        offset=0,
    )
