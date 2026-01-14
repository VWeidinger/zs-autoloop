from classes.model_config_helper import parameterized_config
from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from scripts.exp_runner.exp_train_model import generate_training_commands


def replace_file_paths_in_train_command(train_command: str) -> str:
    # Replace the cluster path with the local path
    train_command = train_command.replace(
        ClusterPaths().Data.parsed_plans2.as_posix(),
        LocalPaths().Data.parsed_plans.as_posix(),
    )  # warning: pp2->pp1
    train_command = train_command.replace(
        ClusterPaths().Data.parsed_plans_stratified2.as_posix(),
        LocalPaths().Data.parsed_plans.as_posix(),
    )  # warning: pp2->pp1 & strat->non-strat

    train_command = train_command.replace(
        ClusterPaths().Data.parsed_plans.as_posix(),
        LocalPaths().Data.parsed_plans.as_posix(),
    )  # warning: pp2->pp1
    train_command = train_command.replace(
        ClusterPaths().Data.parsed_plans_stratified.as_posix(),
        LocalPaths().Data.parsed_plans.as_posix(),
    )  # warning: pp2->pp1 & strat->non-strat
    # replace statistics
    train_command = train_command.replace(
        ClusterPaths().Data.dataset_statistics.as_posix(),
        LocalPaths().Data.dataset_statistics.as_posix(),
    )
    train_command = train_command.replace(
        ClusterPaths().Data.root.as_posix(), LocalPaths().Data.root.as_posix()
    )

    # replace target path
    train_command = train_command.replace(ClusterPaths().Data.models.as_posix(), "tmp")
    train_command = train_command.replace(
        ClusterPaths().Data.evaluation.as_posix(), "tmp"
    )

    # replace device
    train_command = train_command.replace("cuda:[socket]", "cpu")

    return train_command


if __name__ == "__main__":
    model_configs = [
        parameterized_config(
            "zeroshot",
            wandb_group="superhetero",
            include_pg_opt_workloads=True,
            separate_node_type_per_plan_op=True,
        )
    ]

    datasets = [
        Dataset("imdb"),
    ]

    # for dataset in datasets:
    eval_commands = generate_training_commands(
        datasets=datasets,
        model_list=model_configs,
        python_version="",
        seeds=[0],
        wandb_project=None,
        wandb_entity=None,
        quick_run=True,
        pt_profiler=None,
        test_against="imdb",
        train_on_test=True,
    )

    # adjust paths of training commands
    eval_commands = [
        replace_file_paths_in_train_command(command) for command in eval_commands
    ]

    for cmd in eval_commands:
        print(cmd)
