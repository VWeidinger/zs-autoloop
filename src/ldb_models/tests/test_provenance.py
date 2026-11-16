import json

import numpy as np

from ldb_models.classes.model_config_helper import get_model_config
from ldb_models.training.dataset.dataset_creation import split_plans_in_train_val
from ldb_models.training.training.provenance import (
    collect_training_info,
    file_fingerprint,
    read_training_info,
    read_training_split,
    record_split,
    take_split,
    write_training_info,
    write_training_split,
)


def _plans(per_file=(5, 4)):
    """Plans as read_workload_runs leaves them: every other position of a file was
    filtered out, so run_file_pos is not the index in the loaded list."""
    return [
        {"run_file_id": file_id, "run_file_pos": 2 * i, "name": f"f{file_id}p{2 * i}"}
        for file_id, n in enumerate(per_file)
        for i in range(n)
    ]


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


def test_split_positions_name_the_plans_of_each_split(tmp_path):
    plans = _plans()
    np.random.seed(9)
    train_indexes, val_indexes = split_plans_in_train_val(
        plans, val_ratio=0.3, shuffle_before_split=True, cap_training_samples=None
    )
    record_split(["a.json", "b.json"], plans, train_indexes, val_indexes)
    split = take_split()
    assert take_split() is None  # handed out once

    by_position = {(p["run_file_id"], p["run_file_pos"]): p["name"] for p in plans}
    train_names = [by_position[tuple(pos)] for pos in split["train"]]
    val_names = [by_position[tuple(pos)] for pos in split["val"]]
    assert train_names == [plans[i]["name"] for i in train_indexes]  # dataset order is kept
    assert val_names == [plans[i]["name"] for i in val_indexes]
    assert not set(train_names) & set(val_names)
    assert set(train_names) | set(val_names) == set(by_position.values())

    summary = write_training_split(split, tmp_path, "pm-zeroshot-hi_9")
    assert (summary["n_train"], summary["n_val"]) == (len(train_indexes), len(val_indexes))
    assert read_training_split(summary["path"]) == split
    # the same split gives the same bytes, so the hash in the training info is stable
    assert write_training_split(split, tmp_path, "pm-zeroshot-hi_9")["sha256"] == summary["sha256"]


def test_split_of_per_file_pools_and_without_validation():
    per_file = [_plans((3,)), [dict(p, run_file_id=1) for p in _plans((2,))]]
    record_split(["a.json", "b.json"], per_file, [[2, 0], [1]], [[1], [0]])
    split = take_split()
    assert split["per_file_pools"] is True
    assert split["train"] == [[0, 4], [0, 0], [1, 2]]
    assert split["val"] == [[0, 2], [1, 0]]

    plans = _plans((2,))
    plans[1].pop("run_file_pos")  # a plan that was injected, not read from a file
    record_split(["a.json"], plans, [0, 1], None)
    assert take_split() == {"files": ["a.json"], "per_file_pools": False, "train": [[0, 0], [0, -1]], "val": []}
