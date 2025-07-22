from typing import List

from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync, KillAllScreens

from classes.classes import (ModelConfig)
from classes.model_config_helper import parameterized_config
from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers


def generate_commands(dataset: Dataset,
                      model: ModelConfig,
                      workloads: List[str] = None,
                      seeds=None):
    if seeds is None:
        seeds = [0]

    commands = []
    test_runs = ""
    for workload in workloads:
        test_runs += " " + model.assemble_wl_path(target_db=dataset,
                                                wl_name=workload,
                                                stratified_workloads=False,
                                                romans_runs=False)

    for seed in seeds:
        command = (f"python3 main.py "
                   f"--mode predict "
                   f"--model_type {model.name.NAME} "
                   f"--test_workload_runs {test_runs} "
                   f"--statistics_file {model.get_statistics(ClusterPaths(), dataset, model.input_format)} "
                   f"--model_dir {model.get_model_dir(ClusterPaths(), dataset)} "
                   f"--target_dir {model.get_eval_dir(ClusterPaths(), dataset)} "
                   f"--device {model.device} "
                   f"--seed {seed} ")

        if model.column_statistics:
            command += f" --column_statistics {model.get_column_stats(dataset)} "

        if model.word_embeddings:
            command += f" --word_embeddings {model.get_word_embeddings(source_path=ClusterPaths(), database=dataset)} "

        if model.hyperparameter:
            command += f" --hyperparameter_path {model.hyperparameter} "

        commands.append(command)
    return commands


if __name__ == '__main__':
    setup_steps = [Rsync(src=[str(LocalPaths().Models.src)], dest=[ClusterPaths().Models.root], update=True, put=True)]
    purge_steps = [KillAllScreens()]
    pickup_steps = [Rsync(src=[ClusterPaths().Data.evaluation], dest=[LocalPaths().data], put=False, update=True)]

    model_configs = []
    for model_type in ['postgres', 'zeroshot', 'dace', 'flat']:
        model_configs.extend([parameterized_config(model_type, wandb_group='baselines')])

    node = TrainingServers.NODE04
    seeds = [0]

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       python_version="rye")

    commands = generate_commands(dataset=Dataset('imdb'),
                                 model=model_configs[2],
                                 workloads=["join_order_job_0"],
                                 seeds=seeds)

    runner.run_exp(commands=commands,
                    set_up_steps=setup_steps,
                    purge_steps=purge_steps,
                    pickup_steps=pickup_steps,
                    offset=0)