from ldb_schema.paths import ClusterPaths
from octopus.step import Cmd

from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == '__main__':
    install_venv = True

    tb_dir = ClusterPaths().Data.models / 'model_name'
    setup_steps = [
        Cmd(f'screen -S tensorboard -d -m bash -c "source venv_tb/bin/activate; tensorboard --logdir={tb_dir}"')
    ]

    node = TrainingServers.NODE09
    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       python_version=node["python"])

    runner.run_exp(node_names=[node["hostname"]],
                   commands=[],
                   set_up_steps=setup_steps,
                   purge_steps=[],
                   pickup_steps=[],
                   screens_per_node=1,
                   offset=0)
