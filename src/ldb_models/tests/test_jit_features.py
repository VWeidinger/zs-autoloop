"""
Tests of the JIT plan features (use_jit_features, parsed_plans_version 50).

The statistics file / dataloader / model tests run on a parsed_plans50 workload and the parsed_plans50 feature
statistics and are skipped if these files are missing. Configurable via environment variables:
- JIT_TEST_PARSED_PLANS: parsed_plans50 workload file (default: data/runs/parsed_plans50/imdb_scaled1/workload_200k_s1.json)
- JIT_TEST_FEATURE_STATISTICS: feature statistics file (default: data/feature_statistics/parsed_plans50_feature_statistics_v20261104.json)
- JIT_TEST_NUM_PLANS: number of plans used (default: 300)
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ldb_models.classes.configs.base import DataLoaderOptions
from ldb_models.classes.configs.zeroshot import ZeroShotModelConfig, ZSPGCostWithHintPlan
from ldb_models.classes.model_config_helper import get_model_config
from ldb_models.classes.workload_runs import WorkloadRuns
from ldb_models.models.zeroshot.encoding import encode
from ldb_models.models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from ldb_models.training.dataset.dataset_creation import create_zeroshot_dataloader
from ldb_models.training.featurizations import (
    JIT_PLAN_FEATURES,
    PostgresEstSystemCardDetail,
    PostgresEstWithPGCost,
)
from ldb_models.training.preprocessing.feature_statistics import FeatureType, gather_feature_statistics

REPO_ROOT = Path(__file__).resolve().parents[3]
JIT_TEST_PARSED_PLANS = Path(
    os.environ.get(
        "JIT_TEST_PARSED_PLANS",
        REPO_ROOT / "data/runs/parsed_plans50/imdb_scaled1/workload_200k_s1.json",
    )
)
JIT_TEST_FEATURE_STATISTICS = Path(
    os.environ.get(
        "JIT_TEST_FEATURE_STATISTICS",
        REPO_ROOT / "data/feature_statistics/parsed_plans50_feature_statistics_v20261104.json",
    )
)
JIT_TEST_NUM_PLANS = int(os.environ.get("JIT_TEST_NUM_PLANS", "300"))
JIT_FLAGS = ["jit", "jit_inlining", "jit_optimization", "jit_compiled_here"]


def test_flag_adds_jit_plan_features():
    config = ZeroShotModelConfig(parsed_plans_version=50, use_jit_features=True)
    assert config.featurization.__name__ == "PostgresEstSystemCardDetailWithJIT"
    assert config.featurization.PLAN_FEATURES == (
        PostgresEstSystemCardDetail.PLAN_FEATURES + JIT_PLAN_FEATURES
    )
    # all other node types are featurized as before
    assert config.featurization.FILTER_FEATURES == PostgresEstSystemCardDetail.FILTER_FEATURES
    assert config.featurization.COLUMN_FEATURES == PostgresEstSystemCardDetail.COLUMN_FEATURES
    # the base featurization is not modified
    assert "jit" not in PostgresEstSystemCardDetail.PLAN_FEATURES

    # the same class for every config (picklable by name) and works on top of other featurizations
    assert ZeroShotModelConfig(parsed_plans_version=50, use_jit_features=True).featurization is config.featurization
    other = ZSPGCostWithHintPlan(parsed_plans_version=50, use_jit_features=True)
    assert other.featurization.PLAN_FEATURES == PostgresEstWithPGCost.PLAN_FEATURES + JIT_PLAN_FEATURES


def test_default_and_parsed_plans50_without_flag_keep_featurization():
    assert ZeroShotModelConfig().featurization is PostgresEstSystemCardDetail
    assert ZeroShotModelConfig(parsed_plans_version=50).featurization is PostgresEstSystemCardDetail


def test_flag_requires_parsed_plans50():
    with pytest.raises(ValueError, match="parsed_plans_version 50"):
        ZeroShotModelConfig(use_jit_features=True)


def test_parsed_plans50_behaves_like_parsed_plans3():
    v3 = ZeroShotModelConfig(parsed_plans_version=3)
    v50 = ZeroShotModelConfig(parsed_plans_version=50)
    assert v3.do_operator_rewriting and v50.do_operator_rewriting
    assert not ZeroShotModelConfig(parsed_plans_version=2).do_operator_rewriting


def test_parsed_plans50_paths(monkeypatch):
    monkeypatch.setenv("CLUSTER_ROOT_PATH", "/cluster/root")
    monkeypatch.setenv("CLUSTER_DATA_PATH", "/cluster/data")
    monkeypatch.setenv("CLUSTER_KNOWN_HOSTS_PATH", "/cluster/known_hosts")
    from ldb_schema.paths import ClusterPaths

    config = ZeroShotModelConfig(parsed_plans_version=50, use_jit_features=True)
    # no stratified copy of parsed_plans50: train and test on the full workloads
    for stratified in [True, False]:
        assert config.get_data_base_dir(stratified_workloads=stratified) == Path(
            "/cluster/data/runs/parsed_plans50"
        )
    # own feature statistics (containing the JIT features)
    statistics = config.get_statistics(
        ClusterPaths(),
        dataset=None,
        format=config.input_format,
        suffix=config.get_feature_stats_suffix(),
    )
    assert statistics == Path(
        "/cluster/data/feature_statistics/parsed_plans50_feature_statistics_v20261104.json"
    )


def test_flag_in_model_string_and_name():
    config = get_model_config(
        "pm-zeroshot-include_pg_opt_workloads-parsed_plans_version50-use_jit_features", {}
    )
    assert config.use_jit_features
    assert config.parsed_plans_version == 50
    assert "jit_compiled_here" in config.featurization.PLAN_FEATURES
    assert config.name.NAME == "pm-zeroshot-pg_opt-PPV50-JIT"


def _jit_node(jit_functions: int, compiled_here: bool = False, jit: bool = True) -> dict:
    return dict(
        op_name="Seq Scan",
        jit=jit,
        jit_inlining=jit,
        jit_optimization=jit,
        jit_functions=jit_functions if jit else 0,
        jit_compiled_here=compiled_here,
    )


def _jit_block(jit_functions: int) -> dict:
    # JIT block of the whole plan as written by the plan parser (incl. measured timings)
    return dict(
        jit=jit_functions > 0,
        jit_functions=jit_functions,
        jit_inlining=False,
        jit_optimization=False,
        jit_expressions=True,
        jit_deforming=True,
        timing=dict(Generation=1.0, Inlining=0.0, Optimization=1.0, Emission=2.0, Total=4.0),
    )


def _assert_jit_statistics_encodable(statistics: dict):
    for flag in JIT_FLAGS:
        assert [encode(flag, {flag: value}, statistics) for value in [False, True]] == [0, 1], (
            f"{flag}: {statistics[flag]}"
        )
        if statistics[flag]["type"] == str(FeatureType.categorical):
            assert statistics[flag]["no_vals"] == 2
    assert statistics["jit_functions"]["type"] == str(FeatureType.numeric)
    assert statistics["jit_functions"]["min"] == 0 and statistics["jit_functions"]["scale"] > 0


def test_gather_feature_statistics_jit_features(tmp_path):
    # two plans: JIT with 10 functions (2 nodes, compiled in the leaf), no JIT (1 node)
    jit_plan = dict(
        plan_parameters=_jit_node(10),
        children=[dict(plan_parameters=_jit_node(10, compiled_here=True), children=[])],
        jit=_jit_block(10),
    )
    no_jit_plan = dict(plan_parameters=_jit_node(0, jit=False), children=[], jit=_jit_block(0))
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(dict(parsed_plans=[jit_plan, no_jit_plan])))
    target = tmp_path / "statistics.json"

    gather_feature_statistics([run_path.as_posix()], target.as_posix())
    statistics = json.loads(target.read_text())

    _assert_jit_statistics_encodable(statistics)
    # flags are stored with the keys encode() looks up (json would write true / false)
    assert statistics["jit_compiled_here"]["value_dict"] == {"False": 0, "True": 1}
    # statistics of the plan nodes only, not of the JIT block of the plan (10, 10, 0 - not 10, 10, 0, 10, 0)
    assert statistics["jit_functions"]["avg"] == pytest.approx(20 / 3)
    for key in ["jit_expressions", "jit_deforming", "Generation", "Total"]:
        assert key not in statistics


def test_feature_statistics_file_jit_features():
    if not JIT_TEST_FEATURE_STATISTICS.exists():
        pytest.skip(f"Feature statistics not found: {JIT_TEST_FEATURE_STATISTICS}")
    with open(JIT_TEST_FEATURE_STATISTICS) as f:
        statistics = json.load(f)
    _assert_jit_statistics_encodable(statistics)
    # the features of the other plan nodes are there as well
    for feature in PostgresEstSystemCardDetail.PLAN_FEATURES + ["operator", "aggregation", "data_type"]:
        assert feature in statistics


# --------------------------------------------------------------------------------------------------------------------
# dataloader / model on real parsed_plans50 plans


@pytest.fixture(scope="module")
def subset_plans(tmp_path_factory):
    missing = [p for p in [JIT_TEST_PARSED_PLANS, JIT_TEST_FEATURE_STATISTICS] if not p.exists()]
    if len(missing) > 0:
        pytest.skip(f"JIT test files not found: {missing}")

    with open(JIT_TEST_PARSED_PLANS) as f:
        run = json.load(f)
    # plans which are not filtered by read_workload_runs, in file order (test datasets are not shuffled)
    plans = [
        p
        for p in run["parsed_plans"]
        if not (p["timeout"] or p["invalid_hint"] or p["zero_card"])
    ][:JIT_TEST_NUM_PLANS]
    assert any(p["jit"]["jit"] for p in plans) and not all(p["jit"]["jit"] for p in plans)

    # keep the original dataset directory name (the database name is read from the plans)
    path = tmp_path_factory.mktemp("parsed_plans50") / JIT_TEST_PARSED_PLANS.parent.name / "subset.json"
    path.parent.mkdir()
    with open(path, "w") as f:
        json.dump(dict(parsed_plans=plans, database_stats=run["database_stats"]), f)
    return path, plans


def _plan_nodes(plan):
    yield plan["plan_parameters"]
    for child in plan["children"]:
        yield from _plan_nodes(child)


def _test_loader(path: Path, **config_kwargs):
    config = ZeroShotModelConfig(
        device="cpu", num_workers=0, batch_size=JIT_TEST_NUM_PLANS, parsed_plans_version=50, **config_kwargs
    )
    _, feature_statistics, _, _, test_loaders, _ = create_zeroshot_dataloader(
        workload_runs=WorkloadRuns(test_workload_runs=[path]),
        statistics_file=JIT_TEST_FEATURE_STATISTICS,
        model_config=config,
        data_loader_options=DataLoaderOptions(),
    )
    return config, feature_statistics, test_loaders[0]


def _plan_feature_matrix(features) -> np.ndarray:
    return np.concatenate([v.numpy() for k, v in sorted(features.items()) if k.startswith("plan")])


def test_dataloader_encodes_jit_features(subset_plans):
    path, plans = subset_plans
    _, feature_statistics, loader = _test_loader(path, use_jit_features=True)

    graph, features, labels, _ = next(iter(loader))
    assert len(labels) == len(plans)
    plan_features = _plan_feature_matrix(features)
    jit_columns = {
        feature: plan_features[:, len(PostgresEstSystemCardDetail.PLAN_FEATURES) + i]
        for i, feature in enumerate(JIT_PLAN_FEATURES)
    }
    nodes = [node for plan in plans for node in _plan_nodes(plan)]
    assert len(plan_features) == len(nodes)

    # flags are encoded as 0/1
    for flag in JIT_FLAGS:
        assert set(np.unique(jit_columns[flag])) <= {0.0, 1.0}
        assert jit_columns[flag].sum() == sum(node[flag] for node in nodes)
    # one node per JIT plan pays the JIT compilation
    assert jit_columns["jit_compiled_here"].sum() == sum(p["jit"]["jit"] for p in plans)
    # number of functions is robust scaled with the statistics of the file
    stats = feature_statistics["jit_functions"]
    unscaled = jit_columns["jit_functions"] * stats["scale"] + stats["center"]
    assert np.allclose(np.sort(unscaled), np.sort([node["jit_functions"] for node in nodes]))


def test_dataloader_keeps_existing_features(subset_plans):
    path, _ = subset_plans
    _, _, loader = _test_loader(path)
    _, features, labels, _ = next(iter(loader))
    _, _, jit_loader = _test_loader(path, use_jit_features=True)
    _, jit_features, jit_labels, _ = next(iter(jit_loader))

    assert torch.equal(torch.as_tensor(labels), torch.as_tensor(jit_labels))
    assert features.keys() == jit_features.keys()
    num_plan_features = len(PostgresEstSystemCardDetail.PLAN_FEATURES)
    for node_type in features:
        if node_type.startswith("plan"):
            assert jit_features[node_type].shape[1] == num_plan_features + len(JIT_PLAN_FEATURES)
            assert torch.equal(features[node_type], jit_features[node_type][:, :num_plan_features])
        else:
            assert torch.equal(features[node_type], jit_features[node_type])


def test_dataloader_artificial_root_aggregate(subset_plans, tmp_path):
    # remove the root aggregate - the collator re-adds an artificial one with the global JIT features of the query
    path, _ = subset_plans
    with open(path) as f:
        run = json.load(f)
    stripped = []
    for plan in run["parsed_plans"]:
        if plan["plan_parameters"]["op_name"] in ["Simple Aggregate", "Aggregate"] and len(plan["children"]) == 1:
            child = plan["children"][0]
            child.update({k: v for k, v in plan.items() if k not in ["plan_parameters", "children"]})
            plan = child
        stripped.append(plan)
    stripped_path = tmp_path / path.parent.name / "stripped.json"
    stripped_path.parent.mkdir()
    with open(stripped_path, "w") as f:
        json.dump(dict(parsed_plans=stripped, database_stats=run["database_stats"]), f)

    _, _, loader = _test_loader(stripped_path, use_jit_features=True, enforce_aggregation_node_at_root=True)
    _, features, _, _ = next(iter(loader))
    root_features = features["plan0"].numpy()
    jit_idx = len(PostgresEstSystemCardDetail.PLAN_FEATURES)
    assert root_features[:, jit_idx].sum() == sum(p["jit"]["jit"] for p in stripped)
    # the artificial aggregate does not pay the JIT compilation
    compiled_here = _plan_feature_matrix(features)[:, jit_idx + JIT_PLAN_FEATURES.index("jit_compiled_here")]
    assert compiled_here.sum() == sum(node["jit_compiled_here"] for p in stripped for node in _plan_nodes(p))


def test_model_forward_with_jit_features(subset_plans):
    path, plans = subset_plans
    config, feature_statistics, loader = _test_loader(path, use_jit_features=True)
    torch.manual_seed(0)
    model = PostgresZeroShotModel(model_config=config, feature_statistics=feature_statistics, label_norm=None)
    # the plan encoder sees the JIT features
    assert model.encoder.node_type_encoders["plan"].features == config.featurization.PLAN_FEATURES

    batch = next(iter(loader))
    model_input, labels, _ = config.batch_to_func(batch, "cpu", None)
    with torch.no_grad():
        predictions = model(model_input).plan_predictions
    assert predictions.shape == (len(plans), 1)
    assert torch.isfinite(predictions).all()
