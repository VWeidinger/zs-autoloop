from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync

from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == '__main__':
    install_venv = True

    tb_dir = '/mnt/labstore/SIGs/ML/learned_db/models/zeroshot/airline/zeroshot_42.prof'

    node = TrainingServers.NODE09
    runner = ExpRunner(replicate=False,
                       node_names=[node["hostname"]],
                       python_version=node["python"])

    runner.run_exp(node_names=[node["hostname"]],
                   commands=[],
                   set_up_steps=[],
                   purge_steps=[],
                   pickup_steps=[
                       Rsync(src=[f'{ClusterPaths().Data.models}/zeroshot/airline/zeroshot_42.prof'], dest=[
                           LocalPaths().Data.models / 'zeroshot/airline/'
                       ], put=False, update=True)
                   ],
                   screens_per_node=1,
                   offset=0)
