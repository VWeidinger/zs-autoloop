from typing import Dict

import torch
from classes import MambaModelConfig
from models.white_mamba.mamba_dataset import (
    augment_feature_stats_for_mamba,
    expand_scan_featurization,
)
from models.white_mamba.mamba_subplan_runtime_helper import (
    mamba_compute_subplan_runtimes,
)
from models.zeroshot.subplan_runtime_helper import (
    extract_smaller_larger_preds_tensor_for_subplan_graphs,
)
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.zero_shot_model import NodeTypeEncoder
from torch import nn
from training.losses import LessThanLoss, RelativeLessThanLoss, create_loss
from training.preprocessing.feature_statistics import FeatureType
from training.training.outputs import LossOutput, ModelOutput


class DAGDualInputConv(nn.Module):
    def __init__(
        self,
        msg_layer_kwargs,
        hidden_dim: int,
        layer_norm: bool,
        add_child_type_info_to_messages: bool,
        residual_connection: bool,
    ):
        super().__init__()
        self.add_child_type_info_to_messages = add_child_type_info_to_messages
        self.residual_connection = residual_connection

        if self.add_child_type_info_to_messages:
            input_dim = 3 * hidden_dim + 2  # 2 for the child node types
        else:
            input_dim = 3 * hidden_dim

        # initialize the message passing layer
        self.mlp = FcOutModel(
            **msg_layer_kwargs, input_dim=input_dim, output_dim=hidden_dim
        )

        # initialize layer normalization if specified
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = None

    def forward(self, x, edge_index, left_right_edge_info, node_type_lookup):
        # Extract edge info
        src, tgt = edge_index

        assert src.shape == tgt.shape, (
            f"Source and target edges must have the same number of elements: src shape {src.shape}, tgt shape {tgt.shape}"
        )

        assert left_right_edge_info.shape[0] == src.shape[0], (
            "Left/right edge info must match the number of edges"
        )
        assert left_right_edge_info.shape == src.shape, (
            f"Left/right edge info must have the same shape as the edges: {left_right_edge_info.shape}, edges shape: {src.shape}"
        )

        left_mask = left_right_edge_info == 0
        right_mask = left_right_edge_info == 1

        # the masks should have the same number of elements set to true - it is counting edges, so the root node does not count
        assert left_mask.sum() + right_mask.sum() == src.shape[0], (
            f"Left mask has {left_mask.sum()} elements set to true, right mask has {right_mask.sum()} elements set to true. The sum of both masks must equal the number of edges: {src.shape[0]}."
        )

        # Initialize messages
        left_msg = torch.zeros_like(x)
        right_msg = torch.zeros_like(x)

        left_src_indices = src[left_mask]
        right_src_indices = src[right_mask]

        # align the left nodes by the dst of their edges - write parent messages to the parent node position in tensor
        # this is necessary to ensure that the left and right messages are correctly aligned
        if left_mask.any():
            left_vals = x.index_select(0, left_src_indices)
            left_tgt = tgt[left_mask]

            # write the left messages to the target nodes using scatter_
            left_msg.scatter_(0, left_tgt.unsqueeze(-1).expand_as(left_vals), left_vals)

        # align the edges by their dst node - write child messages to the parent node position in tensor
        # this is necessary to ensure that the left and right messages are correctly aligned
        if right_mask.any():
            right_vals = x.index_select(0, right_src_indices)
            right_tgt = tgt[right_mask]

            # # write the right messages to the target nodes
            right_msg.scatter_(
                0, right_tgt.unsqueeze(-1).expand_as(right_vals), right_vals
            )

        assert left_msg.shape == right_msg.shape, (
            "Left and right messages must have the same shape"
        )

        # extract only the relevant messages
        target_indices = tgt[left_mask]  # use the order of the left edges for ordering
        left_msg_raw = left_msg.index_select(0, target_indices)
        right_msg_raw = right_msg.index_select(0, target_indices)
        target_state = x.index_select(0, target_indices)
        assert (
            left_msg_raw.shape[0] == right_msg_raw.shape[0] == target_state.shape[0]
        ), (
            f"Left message shape {left_msg_raw.shape}, right message shape {right_msg_raw.shape}, x shape {x.shape}"
        )
        assert left_msg_raw.shape == right_msg_raw.shape, (
            f"Left message shape {left_msg_raw.shape}, right message shape {right_msg_raw.shape}"
        )

        # concatenate messages with the current node features
        if self.add_child_type_info_to_messages:
            left_ntypes = node_type_lookup.index_select(0, left_src_indices).unsqueeze(
                -1
            )  # shape: [num_left_edges, 1]
            right_ntypes = node_type_lookup.index_select(
                0, right_src_indices
            ).unsqueeze(-1)  # shape: [num_right_edges, 1]
            assert left_ntypes.shape == right_ntypes.shape, (
                f"Left node types shape {left_ntypes.shape} must match right node types shape {right_ntypes.shape}"
            )
            left_msg = torch.cat([left_msg_raw, left_ntypes], dim=-1)
            right_msg = torch.cat([right_msg_raw, right_ntypes], dim=-1)
        else:
            left_msg = left_msg_raw
            right_msg = right_msg_raw

        msg_input = torch.cat([left_msg, target_state, right_msg], dim=-1)

        # apply MLP to the concatenated messages
        out = self.mlp.fcout(msg_input)
        assert out.shape == target_state.shape, (
            f"Output shape {out.shape} must match target state shape {target_state.shape}"
        )

        # apply residual connection if specified
        if self.residual_connection:
            out = out + target_state + left_msg_raw + right_msg_raw

        # apply layer normalization if specified
        if self.norm is not None:
            out = self.norm(out)

        # write new node states back to the target nodes
        x[target_indices] = out

        # return node features tensor that includes the updated node states
        return x


class DAGLayerwiseNetworkBatched(torch.nn.Module):
    def __init__(
        self,
        msg_layer_kwargs: Dict,
        hidden_dim: int,
        layer_norm: bool,
        add_child_type_info_to_messages: bool,
        residual_connection: bool,
    ):
        super().__init__()
        self.mp_layer = DAGDualInputConv(
            msg_layer_kwargs=msg_layer_kwargs,
            hidden_dim=hidden_dim,
            layer_norm=layer_norm,
            add_child_type_info_to_messages=add_child_type_info_to_messages,
            residual_connection=residual_connection,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_depths: torch.Tensor,
        left_right_edge_info: torch.Tensor,
        node_type_lookup: torch.Tensor,
    ):
        # compute max depth
        max_depth = node_depths.max().item()

        sanity_edge_ctr = 0

        for layer in reversed(range(1, max_depth + 1)):
            assert edge_index.shape[0] == 2, (
                f"Edge index must be a 2D tensor with shape [2, num_edges], got shape {edge_index.shape}"
            )
            src, tgt = edge_index
            mask = node_depths[src] == layer
            sanity_edge_ctr += mask.sum().item()

            # filter edges based on the current layer
            edge_subset = edge_index[:, mask]

            # extract the corresponding left/right edge information
            left_right_edge_info_subset = left_right_edge_info[mask]

            if edge_subset.size(1) > 0:  # this must always be true, but just in case
                # apply message passing along these edges
                x = self.mp_layer(
                    x, edge_subset, left_right_edge_info_subset, node_type_lookup
                )

        assert sanity_edge_ctr == edge_index.shape[1], (
            f"Sanity check failed: {sanity_edge_ctr} edges processed, but {edge_index.shape[1]} edges in total."
        )

        return x


class MambaModelEncoder(torch.nn.Module):
    def __init__(self, model_config: MambaModelConfig, feature_statistics: Dict):
        super().__init__()

        featurization = model_config.featurization
        self.hidden_dim = model_config.hidden_dim

        # augment scan features
        scan_features = expand_scan_featurization(featurization)

        # create node type encoders
        model_config.node_type_kwargs.update(output_dim=model_config.hidden_dim)

        nte_config = {
            "feature_statistics": feature_statistics,
            "apply_dcn": model_config.use_dcn,
            "apply_layer_norm_before_dcn": model_config.apply_layer_norm_before_dcn,
            **model_config.node_type_kwargs,
        }
        scan_nte_config = {
            "features": scan_features,
            **nte_config,
        }
        join_nte_config = {
            "features": featurization.JOIN_FEATURES,
            **nte_config,
        }

        if model_config.scan_nte_relu:
            scan_nte_config["activation_class_name"] = "ReLU"

        self.scan_encoder = NodeTypeEncoder(
            **scan_nte_config,
        )
        self.join_encoder = NodeTypeEncoder(
            **join_nte_config,
        )

        self.dag_network = DAGLayerwiseNetworkBatched(
            msg_layer_kwargs=model_config.msg_layer_kwargs,
            hidden_dim=model_config.hidden_dim,
            layer_norm=model_config.layer_norm,
            add_child_type_info_to_messages=model_config.add_child_type_info_to_messages,
            residual_connection=model_config.residual_connection,
        )

    def assemble_node_features_from_scan_and_join(
        self, scan_encoded, join_encoded, node_type_lookup, node_type_id_lookup
    ):
        # retrieve masks
        scan_mask = node_type_lookup == 0
        join_mask = node_type_lookup == 1
        scan_indices = node_type_id_lookup[scan_mask]
        join_indices = node_type_id_lookup[join_mask]

        # reorder the node features based on the global node order
        scan_encoded_reordered = scan_encoded[scan_indices]
        join_encoded_reordered = join_encoded[join_indices]

        # make sure that during this reordering, no node gets lost
        assert scan_encoded_reordered.shape == scan_encoded.shape, (
            f"Scan encoded reordered shape {scan_encoded_reordered.shape} does not match scan features shape {scan_encoded.shape}"
        )
        assert join_encoded_reordered.shape == join_encoded.shape, (
            f"Join encoded reordered shape {join_encoded_reordered.shape} does not match join features shape {join_encoded.shape}"
        )

        node_features = torch.zeros(
            (node_type_lookup.shape[0], self.hidden_dim), device=scan_encoded.device
        )
        node_features[scan_mask] = scan_encoded_reordered
        node_features[join_mask] = join_encoded_reordered

        return node_features

    def forward(
        self,
        edge_ids,
        edge_left_right_information,
        plan_ids,
        node_depths,
        node_type_lookup,
        node_type_id_lookup,
        join_features,
        scan_features,
    ):
        # apply node type encoders
        scan_encoded = self.scan_encoder(scan_features)
        join_encoded = self.join_encoder(join_features)

        assert (
            scan_encoded.shape[0] == node_type_id_lookup[node_type_lookup == 0].shape[0]
        ), (
            f"Scan encoded shape {scan_encoded.shape} does not match the number of scan nodes {node_type_id_lookup[node_type_lookup == 0].shape[0]}"
        )
        assert (
            join_encoded.shape[0] == node_type_id_lookup[node_type_lookup == 1].shape[0]
        ), (
            f"Join encoded shape {join_encoded.shape} does not match the number of join nodes {node_type_id_lookup[node_type_lookup == 1].shape[0]}"
        )

        # bring the scan and join features together
        node_features = self.assemble_node_features_from_scan_and_join(
            scan_encoded, join_encoded, node_type_lookup, node_type_id_lookup
        )

        # Apply DAG layerwise network
        mp_results = self.dag_network(
            node_features,
            edge_ids,
            node_depths,
            edge_left_right_information,
            node_type_lookup,
        )

        # perform readout
        readout_mask = (
            node_depths == 0
        )  # Only consider join nodes for readout at depth 0
        readout_features = mp_results[readout_mask]

        assert readout_features.shape[1] == self.hidden_dim, (
            f"Readout features shape {readout_features.shape} does not match hidden dimension {self.hidden_dim}"
        )

        # in addition return the node types for the readout features
        node_types = node_type_lookup[readout_mask]

        return readout_features, node_types, mp_results


class MambaModel(nn.Module):
    def __init__(self, model_config: MambaModelConfig, feature_statistics: Dict):
        super(MambaModel, self).__init__()

        self.label_norm = None
        self.device = model_config.device

        # add pseudo entry for filter flat vector - no need to do any embeddings / one-hot encoding here
        local_feature_statistics = feature_statistics.copy()
        local_feature_statistics["filter_flat_vector"] = {
            "type": str(FeatureType.numeric)
        }
        local_feature_statistics["outcols_flat_vector"] = {
            "type": str(FeatureType.numeric)
        }

        augment_feature_stats_for_mamba(local_feature_statistics)

        # if model_config.loss_class_name is not None:
        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )

        self.return_all_node_states = (
            model_config.subplan_monotonicity_loss or model_config.subplan_estimation
        )

        # initialize encoder and decoder
        self.encoder = MambaModelEncoder(model_config, local_feature_statistics)

        # for queries with 2+ tables (i.e. processing JOIN latent spaces)
        self.join_decoder = FcOutModel(
            output_dim=model_config.output_dim,
            input_dim=model_config.hidden_dim,
            final_out_layer=True,
            **model_config.final_mlp_kwargs,
        )
        # for queries with 1 table (i.e. processing SCAN latent spaces)
        self.scan_decoder = FcOutModel(
            output_dim=model_config.output_dim,
            input_dim=model_config.hidden_dim,
            final_out_layer=True,
            **model_config.final_mlp_kwargs,
        )

        if (
            hasattr(model_config, "subplan_monotonicity_loss")
            and model_config.subplan_monotonicity_loss
        ):
            self.perform_subplan_monotonicity_loss = True

            # select loss function
            assert hasattr(model_config, "subplan_monotonicity_loss_fn")
            if model_config.subplan_monotonicity_loss_fn == "lt_loss":
                self.subplan_monotonicity_loss = LessThanLoss()
            elif model_config.subplan_monotonicity_loss_fn == "relative_lt_loss":
                self.subplan_monotonicity_loss = RelativeLessThanLoss()
            else:
                raise ValueError(
                    f"Unknown subplan monotonicity loss function: {model_config.subplan_monotonicity_loss_fn}"
                )
        else:
            self.perform_subplan_monotonicity_loss = False
            self.subplan_monotonicity_loss = None

    def forward(self, model_input, plan_dict=None, **kwargs) -> ModelOutput:
        # run encoder
        root_node_states, root_node_types, all_node_states = self.encoder(*model_input)

        def apply_decoder(hidden_states, node_type):
            # extract node type masks
            join_mask = node_type == 1
            scan_mask = node_type == 0

            outputs = torch.zeros(
                (hidden_states.shape[0], self.join_decoder.output_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if join_mask.any():
                outputs[join_mask] = self.join_decoder.fcout(hidden_states[join_mask])
            if scan_mask.any():
                outputs[scan_mask] = self.scan_decoder.fcout(hidden_states[scan_mask])

            return outputs

        # apply decoders
        outputs = apply_decoder(root_node_states, root_node_types)

        if self.return_all_node_states:
            subplan_graphs, subplan_predictions_tensor = mamba_compute_subplan_runtimes(
                all_node_states, model_input, apply_decoder
            )
            return ModelOutput(
                plan_predictions=outputs,
                subplan_graphs=subplan_graphs,
                subplan_predictions=subplan_predictions_tensor,
            )
        else:
            return ModelOutput(plan_predictions=outputs)

    def compute_loss(
        self, output: ModelOutput, query_stats: dict, labels: torch.Tensor
    ) -> LossOutput:
        # Loss computation
        plan_predictions = output.plan_predictions
        assert len(plan_predictions) == len(labels), (
            f"Output length {len(plan_predictions)} does not match label length {len(labels)}"
        )
        loss = plan_loss = self.loss_fxn(plan_predictions, labels)
        # co
        monotonicity_loss = None
        if self.perform_subplan_monotonicity_loss:
            assert hasattr(output, "subplan_graphs")
            assert hasattr(output, "subplan_predictions")
            # extract smaller and larger tensors - input for our monotonicity loss
            smaller_tensor, larger_tensor = (
                extract_smaller_larger_preds_tensor_for_subplan_graphs(
                    output.subplan_graphs, output.subplan_predictions
                )
            )

            # Loss computation
            monotonicity_loss = self.subplan_monotonicity_loss(
                smaller_tensor, larger_tensor
            )
            loss = loss + monotonicity_loss
        return LossOutput(
            loss=loss,
            plan_loss=plan_loss,
            monotonicity_loss=monotonicity_loss,
        )
