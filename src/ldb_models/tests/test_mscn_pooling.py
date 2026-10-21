from types import SimpleNamespace

import dgl
import torch

from ldb_models.classes.configs.misc import MSCNModelConfig
from ldb_models.models.workload_driven.model.mscn_model import MSCNModel

DIMS = SimpleNamespace(input_dim_table=3, input_dim_pred=4, input_dim_agg=2, input_dim_join=5)


def _model() -> MSCNModel:
    torch.manual_seed(0)
    return MSCNModel(MSCNModelConfig(device="cpu", hidden_dim_plan=8), DIMS).eval()


def _batch(with_second_plan: bool):
    """Plan 0 has a table and an aggregate but no predicate and no join (like a plain scan
    subplan). Plan 1, if present, has every set type."""
    torch.manual_seed(1)
    table, agg = torch.randn(1, 3), torch.randn(1, 2)
    edges = {
        ("table", "table_plan", "plan"): [(0, 0)],
        ("pred", "pred_plan", "plan"): [],
        ("agg", "agg_plan", "plan"): [(0, 0)],
        ("join", "join_plan", "plan"): [],
    }
    feats = {"table": table, "agg": agg}  # the collator drops empty set types
    num_nodes = {"table": 1, "pred": 0, "agg": 1, "join": 0, "plan": 1}
    if with_second_plan:
        edges = {
            ("table", "table_plan", "plan"): [(0, 0), (1, 1)],
            ("pred", "pred_plan", "plan"): [(0, 1)],
            ("agg", "agg_plan", "plan"): [(0, 0), (1, 1)],
            ("join", "join_plan", "plan"): [(0, 1)],
        }
        feats = {
            "table": torch.cat([table, torch.randn(1, 3)]),
            "pred": torch.randn(1, 4),
            "agg": torch.cat([agg, torch.randn(1, 2)]),
            "join": torch.randn(1, 5),
        }
        num_nodes = {"table": 2, "pred": 1, "agg": 2, "join": 1, "plan": 2}
    graph = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    return graph, feats


def test_plan_without_predicate_or_join_is_batch_independent():
    model = _model()
    with torch.no_grad():
        alone = model(_batch(with_second_plan=False)).plan_predictions[0]
        batched = model(_batch(with_second_plan=True)).plan_predictions[0]
    assert torch.allclose(alone, batched, atol=1e-6)
