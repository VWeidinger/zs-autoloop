import dgl
import torch

from ldb_models.classes.configs.zeroshot import ZeroShotModelConfig
from ldb_models.classes.model_config_helper import get_model_config, shorten_model_name
from ldb_models.models.zeroshot.zero_shot_model import ZeroShotModelEncoder

HIDDEN = 8


def _encoder(reproducible: bool) -> ZeroShotModelEncoder:
    torch.manual_seed(0)
    config = ZeroShotModelConfig(
        device="cpu", hidden_dim=HIDDEN, reproducible_message_passing=reproducible
    )
    encoder = ZeroShotModelEncoder(
        model_config=config, add_tree_model_types=[], prepasses=[]
    )
    return encoder.eval()


def _root_of_plan_a(encoder, deeper_plan_in_batch: bool) -> torch.Tensor:
    """
    Plan A is a root over one leaf. Alone it spans plan0..plan1 (its leaf is on the
    deepest level of the batch); next to plan B (root -> mid -> leaf) the batch reaches
    plan2 and A's leaf is no longer the deepest node.
    """
    torch.manual_seed(1)
    a_root, a_leaf, b_root, b_mid, b_leaf, table = (torch.randn(1, HIDDEN) for _ in range(6))
    if deeper_plan_in_batch:
        g = dgl.heterograph(
            {
                ("plan1", "intra_plan", "plan0"): [(0, 0), (1, 1)],
                ("plan2", "intra_plan", "plan1"): [(0, 1)],
                ("table", "to_plan", "plan1"): [(0, 0)],
                ("table", "to_plan", "plan2"): [(0, 0)],
            }
        )
        g.max_depth = 2
        feats = {
            "plan0": torch.cat([a_root, b_root]),
            "plan1": torch.cat([a_leaf, b_mid]),
            "plan2": b_leaf,
            "table": table,
        }
    else:
        g = dgl.heterograph(
            {
                ("plan1", "intra_plan", "plan0"): [(0, 0)],
                ("table", "to_plan", "plan1"): [(0, 0)],
            }
        )
        g.max_depth = 1
        feats = {"plan0": a_root, "plan1": a_leaf, "table": table}
    g.max_pred_depth = None
    with torch.no_grad():
        return encoder.message_passing(g, feats)[0]


def test_reproducible_message_passing_is_batch_independent():
    encoder = _encoder(reproducible=True)
    alone = _root_of_plan_a(encoder, deeper_plan_in_batch=False)
    batched = _root_of_plan_a(encoder, deeper_plan_in_batch=True)
    assert torch.allclose(alone, batched, atol=1e-6)


def test_default_message_passing_depends_on_batch():
    # documents the behaviour the flag fixes (and that the default stays unchanged)
    encoder = _encoder(reproducible=False)
    alone = _root_of_plan_a(encoder, deeper_plan_in_batch=False)
    batched = _root_of_plan_a(encoder, deeper_plan_in_batch=True)
    assert not torch.allclose(alone, batched, atol=1e-6)


def test_flag_in_model_string_and_name():
    config = get_model_config(
        "pm-zeroshot-include_hint_idx_workloads-reproducible_message_passing", {}
    )
    assert config.reproducible_message_passing
    assert config.name.NAME == "pm-zeroshot-hi-RMP"
    assert shorten_model_name("pm-zeroshot-reproducible_message_passingFalse") == "pm-zeroshot-RMPFalse"
