from ldb_schema.datasets import dataset_list_dict
from ldb_schema.paths import ClusterPaths
from octopus.step import KillAllScreens, SetupVenv, RsyncCurrentRepo

from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == '__main__':
    install_venv = False

    setup_steps = [RsyncCurrentRepo(gitignore_path='.gitignore')]

    if install_venv:
        setup_steps.append(SetupVenv(use_requirements_txt=True,
                                     requirements_txt_filename=f'{ClusterPaths().Models.src}/requirements.txt',
                                     force=False,
                                     ))

    purge_steps = [KillAllScreens()]

    node = TrainingServers.NODE03

    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       python_version=node["python"])

    # for dataset in datasets:

    datasets = dataset_list_dict["zs"]

    dataset_str = " ".join([d.data_folder for d in datasets])
    file_name = 'feature_statistics'
    target_folder = f'{ClusterPaths().Data.root}/feature_statistics'

    eval_commands = [
        f'python3 gather_feature_stats.py '
        f'--database {" ".join([d.data_folder for d in datasets])} '
        f'--workload workload_200k_s1.json workload_200k_s1_full_hints.json join_order.json'
        f'--file_name {file_name} '
        f'--target_folder {target_folder} '
        f'--runs_dir {ClusterPaths().Data.runs} '
        f'--file_name feature_statistics',
    ]

    runner.run_exp(node_names=[node["hostname"]],
                   commands=eval_commands,
                   set_up_steps=setup_steps,
                   purge_steps=purge_steps,
                   pickup_steps=[],
                   screens_per_node=1,
                   offset=0)
