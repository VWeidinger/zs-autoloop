import json

from ldb_models.classes.model_config_helper import get_model_config
from ldb_models.training.training.provenance import (
    collect_training_info,
    file_fingerprint,
    read_training_info,
    write_training_info,
)


def test_fingerprint_tells_copies_with_the_same_name_apart(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    full, filtered = tmp_path / "a" / "workload.json", tmp_path / "b" / "workload.json"
    full.write_text('{"parsed_plans": [1, 2, 3]}')
    filtered.write_text('{"parsed_plans": [1]}')
    assert file_fingerprint(full)["head_tail_sha256"] != file_fingerprint(filtered)["head_tail_sha256"]
    assert file_fingerprint(tmp_path / "missing.json")["exists"] is False


def test_training_info_round_trip(tmp_path):
    workload = tmp_path / "workload.json"
    workload.write_text("{}")
    config = get_model_config("pm-zeroshot-include_hint_idx_workloads", {})
    info = collect_training_info(
        cli_args={"model_type": "pm-zeroshot-include_hint_idx_workloads", "seed": 9},
        config=config,
        train_workloads=[workload],
        test_workloads=[],
        other_files={"statistics_file": None},
    )
    json.dumps(info)  # everything must be JSON-serialisable
    assert info["model_config"]["include_hint_idx_workloads"] is True
    path = write_training_info(info, tmp_path, "pm-zeroshot-hi_9")
    assert read_training_info(path) == info
