from ldb_schema.paths import ClusterPaths, LocalPaths
from octopus.step import Rsync

from scripts.exp_runner.exp_runner import ExpRunner, TrainingServers

if __name__ == "__main__":
    install_venv = True

    node = TrainingServers.NODE09
    runner = ExpRunner(
        replicate=False, node_names=[node["hostname"]], python_version=node["python"]
    )

    runner.run_exp(
        node_names=[node["hostname"]],
        commands=[],
        set_up_steps=[],
        purge_steps=[],
        pickup_steps=[
            Rsync(
                src=[
                    f"{ClusterPaths().Data.models}/pm-stealth-pgopt-layer_norm-hintidx-manyjoins-StealthWFilter-add_child_type_info_to_messages/imdb/pm-stealth-include_pg_opt_workloads-layer_norm-include_hint_idx_workloads-include_complex_many_joins_workloads-StealthWFilter-add_child_type_info_to_messages_3"
                ],
                dest=[
                    LocalPaths().Data.models
                    / "pm-stealth-pgopt-layer_norm-hintidx-manyjoins-StealthWFilter-add_child_type_info_to_messages/imdb"
                ],
                put=False,
                update=True,
            )
        ],
        screens_per_node=1,
        offset=0,
    )
