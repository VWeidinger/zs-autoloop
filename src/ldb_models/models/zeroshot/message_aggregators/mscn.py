import math

import dgl
import dgl.function as fn
import torch
from models.zeroshot.message_aggregators.aggregator import MessageAggregator
from torch import nn


class MscnConv(MessageAggregator):
    """
    A message aggregator that sums up child messages and afterward combines them with the current hidden state of the
    parent node using an MLP
    """

    def __init__(
        self,
        node_type=None,
        hidden_dim=0,
        layer_norm: bool = False,
        join_repr_mode: str = "sum",
        **kwargs,
    ):
        # how to represent join / multi-child aggregations
        # supported modes: 'sum' (default), 'proj_then_concat', 'concat_reduce', 'pad_single'
        self.node_type = node_type
        if self.node_type == "intra_plan":
            self.join_repr_mode = join_repr_mode
        else:
            self.join_repr_mode = None
        # determine the dimensionality of the aggregated child message
        # most modes produce a child message of size `hidden_dim`, but `pad_single`
        # intentionally doubles the child message size (2 * hidden_dim)
        child_msg_dim = hidden_dim
        if self.join_repr_mode == "pad_single":
            child_msg_dim = 2 * hidden_dim

        # input_dim is concat of node feature (hidden_dim) and aggregated child message
        super().__init__(
            input_dim=hidden_dim + child_msg_dim, output_dim=hidden_dim, **kwargs
        )

        # projection used by 'proj_then_concat' to reduce each child to half dim
        if self.join_repr_mode == "proj_then_concat":
            half = max(1, hidden_dim // 2)
            self.child_proj = nn.Linear(hidden_dim, half)

        # linear to reduce concatenated children (expects two children)
        if self.join_repr_mode == "concat_reduce":
            # supports two children concatenated -> 2 * hidden_dim -> hidden_dim
            self.concat_reduce = nn.Linear(2 * hidden_dim, hidden_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        graph: dgl.DGLGraph = None,
        etypes=None,
        in_node_types=None,
        out_node_types=None,
        feat_dict=None,
    ):
        if len(etypes) == 0:
            return feat_dict

        with graph.local_scope():
            graph.ndata["h"] = feat_dict
            if self.node_type == "intra_plan" and self.join_repr_mode is not None:
                feat, out_node_types, rst = self.apply_join_method(
                    graph, etypes, out_node_types
                )
            else:
                # standard message passing: sum up all messages per node type
                graph.multi_update_all(
                    {
                        etype: (fn.copy_u("h", "m"), fn.sum("m", "ft"))
                        for etype in etypes
                    },
                    cross_reducer="sum",
                )

                feat = graph.ndata["h"]
                rst = graph.ndata["ft"]

            out_dict = self.combine(feat, out_node_types, rst)

            if self.layer_norm is not None:
                out_dict = {k: self.layer_norm(v) for k, v in out_dict.items()}

            return out_dict

    def apply_join_method(
        self,
        graph: dgl.DGLGraph,
        etypes,
        out_node_types,
    ):
        # message passing
        src_to_etypes = {}
        for src, e_name, dst in etypes:
            # only consider intra-plan edges for message passing
            src_to_etypes.setdefault(src, []).append((src, e_name, dst))

        # prepare per-src reduce field names
        update_dict = {}
        for src, ets in src_to_etypes.items():
            for et in ets:
                # write sums per source type into a dedicated field name
                field_name = f"ft_{src}"
                update_dict[et] = (fn.copy_u("h", "m"), fn.sum("m", field_name))

        # run the per-etype updates (this will produce graph.ndata fields like 'ft_plan', 'ft_scan', etc.)
        # DGL expects a string reducer name; use 'sum' as cross_reducer to merge if necessary
        graph.multi_update_all(update_dict, cross_reducer="sum")

        feat = graph.ndata["h"]

        # Build rst dict expected by combine(): for each out node type, create an aggregated tensor
        rst = {}
        for out_type in out_node_types:
            # collect child tensors per source type for this out_type
            child_tensors = []
            for src in src_to_etypes.keys():
                per_src_field = f"ft_{src}"
                per_src_dict = graph.ndata.get(per_src_field, None)
                if per_src_dict is None:
                    continue
                # per_src_dict is a dict mapping node types -> tensors
                t = per_src_dict.get(out_type, None)
                if t is not None:
                    child_tensors.append(t)
            # fallback to sum if children are not exactly 2 (we only implement special logic for binary joins)
            num_children = len(child_tensors)
            if num_children == 0:
                # no incoming messages for this node type
                continue
            elif num_children == 1:
                if self.join_repr_mode == "pad_single":
                    combined = torch.cat(
                        [child_tensors[0], torch.zeros_like(child_tensors[0])], dim=1
                    )
                else:
                    combined = child_tensors[0]
            elif num_children == 2:
                if self.join_repr_mode == "proj_then_concat":
                    # project each child to half dim then concat
                    projed = [self.child_proj(ct) for ct in child_tensors]
                    combined = torch.cat(projed, dim=1)
                elif self.join_repr_mode == "concat_reduce":
                    combined = self.concat_reduce(torch.cat(child_tensors, dim=1))
                elif self.join_repr_mode == "dot_gate":
                    # parameter-free scalar gate based on dot-product similarity
                    left, right = child_tensors[0], child_tensors[1]
                    d = left.shape[1]
                    # compute scalar compatibility per node
                    score = (left * right).sum(dim=1, keepdim=True) / math.sqrt(
                        max(1, d)
                    )
                    alpha = torch.sigmoid(score)
                    combined = alpha * left + (1.0 - alpha) * right
                elif self.join_repr_mode == "per_dim_gate":
                    # parameter-free per-dimension gating using elementwise product
                    left, right = child_tensors[0], child_tensors[1]
                    gate = torch.sigmoid(left * right)
                    combined = gate * left + (1.0 - gate) * right
                elif self.join_repr_mode == "pad_single":
                    # for two children, concatenate them to produce doubled output dimension
                    combined = torch.cat(child_tensors, dim=1)
                elif self.join_repr_mode == "sum":
                    # default for vanilla zero-shot MSCN: simply sum the two child tensors
                    combined = torch.sum(torch.stack(child_tensors, dim=0), dim=0)
            else:
                child_shapes = [tuple(ct.shape) for ct in child_tensors]
                # Attempt to include graph canonical etypes if available for debugging
                canonical = getattr(graph, "canonical_etypes", None)
                raise ValueError(
                    "Unexpected number of child tensors: {} for out_type={}.\n".format(
                        num_children, out_type
                    )
                    + f"child_shapes={child_shapes}\n"
                    + f"src_to_etypes_keys={list(src_to_etypes.keys())}\n"
                    + f"out_node_types={out_node_types}\n"
                    + f"current out_type={out_type}\n"
                    + f"etypes={etypes}\n"
                    + f"graph.canonical_etypes={canonical}\n"
                )

            rst[out_type] = combined

        return feat, out_node_types, rst
