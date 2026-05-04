
from ldb_schema.datasets import dataset_list_dict
from ldb_schema.paths import ClusterPaths
from octopus.step import (
    KillAllScreens,
    List,
    RsyncCurrentRepo,
    SetupUVVenv,
    SetupVenv,
    Step,
)

from ldb_models.scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == "__main__":
    install_venv = False
    install_rye = True

    setup_steps: List[Step] = [RsyncCurrentRepo(gitignore_path=".gitignore")]

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

    purge_steps: List[Step] = [KillAllScreens()]

    nodes = [
        # TrainingServers.NODE05,
        TrainingServers.NODE06,
        TrainingServers.NODE07,
        TrainingServers.NODE08,
        TrainingServers.NODE09,
    ]

    runner = ExpRunner(
        replicate=False,
        node_names=[node["hostname"] for node in nodes],
        python_version="uv",
        available_sockets=[0, 1],
        num_parallel_screens_or_jobs=1,
    )

    # for dataset in datasets:

    datasets = dataset_list_dict["zs"]
    # datasets = [d for d in datasets if d.db_name in {"financial"}]

    exp_commands = [
        f"python models/workload_driven/preprocessing/sample_vectors.py --dataset {dataset.db_name} --data_dir {ClusterPaths().Data.datasets}/{dataset.data_folder} --plan_path {ClusterPaths().Data.parsed_plans3}/{dataset.data_folder} --target_path {ClusterPaths().Data.augmented_plans3}/{dataset.data_folder} --workloads workload_200k_s1 complex_workload_200k_s1 complex_workload_many_joins_hints full_qo_complex_workload_200k_s1 scan_only_s1_access_path_hints_stratified workload_200k_s1_full_hints_gt30 workload_200k_s1_full_hints_idx workload_200k_s1_full_hints_lt30 join_order access_path_selection access_path_selection_with_aggregate physical_operator physical_operator_w_idx job_w_optimal_w_idxs job_complex_w_optimal_w_idxs"
        for dataset in datasets
    ]

    runner.run_exp(
        commands=exp_commands,
        set_up_steps=setup_steps,
        purge_steps=purge_steps,
        pickup_steps=[],
        offset=0,
    )
