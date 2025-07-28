from typing import List

import numpy as np
import torch
from models.zeroshot.utils.embeddings import EmbeddingInitializer
from models.zeroshot.utils.fc_out_model import FcOutModel
from torch import nn
from training.preprocessing.feature_statistics import FeatureType


class NodeTypeEncoder(FcOutModel):
    """
    Model to encode one type of nodes in the graph (with particular features)
    """

    def __init__(
        self,
        features: List[str],
        feature_statistics: dict,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        one_hot_embeddings=True,
        apply_optimized_nte=False,
        apply_dcn: bool = False,
        apply_layer_norm_before_dcn: bool = False,
        num_dcn_layers: int = 1,
        **kwargs,
    ):
        self.apply_optimized_nte = apply_optimized_nte
        self.apply_dcn = apply_dcn  # apply deep cross network to the input features
        self.apply_layer_norm_before_dcn = apply_layer_norm_before_dcn
        self.num_dcn_layers = num_dcn_layers

        for f in features:
            if f not in feature_statistics:
                raise ValueError(f"Did not find {f} in feature statistics")

        self.features = features
        self.feature_types = [
            FeatureType[feature_statistics[feat]["type"]] for feat in features
        ]
        self.feature_idxs = []

        # initialize embeddings and input dimension

        self.input_dim = 0
        self.input_feature_idx = 0
        embeddings = dict()
        for i, (feat, type) in enumerate(zip(self.features, self.feature_types)):
            if type == FeatureType.numeric:
                # a single value is encoded here
                self.feature_idxs.append(
                    np.arange(self.input_feature_idx, self.input_feature_idx + 1)
                )
                self.input_feature_idx += 1

                self.input_dim += 1
            elif type == FeatureType.categorical:
                # similarly, a single value is encoded here
                self.feature_idxs.append(
                    np.arange(self.input_feature_idx, self.input_feature_idx + 1)
                )
                self.input_feature_idx += 1

                embd = EmbeddingInitializer(
                    num_embeddings=feature_statistics[feat]["no_vals"],
                    max_emb_dim=max_emb_dim,
                    p_dropout=kwargs["p_dropout"],
                    drop_whole_embeddings=drop_whole_embeddings,
                    one_hot=one_hot_embeddings,
                )
                embeddings[feat] = embd
                self.input_dim += embd.emb_dim
            else:
                raise NotImplementedError

        assert self.input_dim > 0, (
            "Input dimension must be greater than 0, but got 0. Check the features and feature statistics."
        )
        super().__init__(input_dim=self.input_dim, **kwargs)

        if self.apply_dcn:
            # add dcm
            self.dcn = DCNLayer(self.input_dim, num_layers=self.num_dcn_layers)

            if self.apply_layer_norm_before_dcn:
                self.dcn_layer_norm = nn.LayerNorm(self.input_dim)

        self.embeddings = nn.ModuleDict(embeddings)

    def forward(self, input):
        if self.no_input_required:
            return self.replacement_param.repeat(input.shape[0], 1)

        if len(input.shape) != 2:
            raise ValueError(
                f"Expected 2D input, got {len(input.shape)}D / shape {input.shape}"
            )

        if input.shape[0] == 0:
            return torch.reshape(input, (0, self.output_dim))

        assert input.shape[1] == self.input_feature_idx, (
            f"{input.shape} != {self.input_feature_idx} / features: {self.features}"
        )

        # Prepare masks for numeric and categorical features
        numeric_idxs = [
            i for i, t in enumerate(self.feature_types) if t == FeatureType.numeric
        ]
        categorical_idxs = [
            i for i, t in enumerate(self.feature_types) if t == FeatureType.categorical
        ]

        encoded_input = []
        if not self.apply_optimized_nte:
            for feat, feat_type, feat_idxs in zip(
                self.features, self.feature_types, self.feature_idxs
            ):
                feat_data = input[:, feat_idxs]
                if feat_type == FeatureType.numeric:
                    encoded_input.append(feat_data)
                elif feat_type == FeatureType.categorical:
                    feat_data = torch.reshape(feat_data, (-1,))
                    embd_data = self.embeddings[feat](feat_data.long())
                    encoded_input.append(embd_data)
                else:
                    raise NotImplementedError
        else:
            # apply optimized nte has been introduced with the white-mamba model
            # all other models trained earlier do not use it. Since it changes the input order, this is not compatible with the old models.
            if numeric_idxs:
                # Stack all numeric columns together
                numeric_feat_idxs = np.concatenate(
                    [self.feature_idxs[i] for i in numeric_idxs]
                )
                numeric_data = input[:, numeric_feat_idxs]
                encoded_input.append(numeric_data)

            if categorical_idxs:
                # Stack all categorical columns together
                cat_feat_idxs = [self.feature_idxs[i][0] for i in categorical_idxs]
                cat_data = input[:, cat_feat_idxs].long()  # shape: (batch, num_cat)
                # For each categorical feature, apply its embedding
                cat_embs = []
                for j, i in enumerate(categorical_idxs):
                    emb = self.embeddings[self.features[i]]
                    embd_data = emb(cat_data[:, j])
                    cat_embs.append(embd_data)
                encoded_input.extend(cat_embs)

        input_enc = torch.cat(encoded_input, dim=1)

        if self.apply_dcn:
            if self.apply_layer_norm_before_dcn:
                input_enc = self.dcn_layer_norm(input_enc)
            # Apply the deep cross network
            input_enc = self.dcn(input_enc)

        return self.fcout(input_enc)


class DCNLayer(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList()
        self.cross_bias = nn.ParameterList()
        for _ in range(num_layers):
            w = nn.Parameter(torch.empty(input_dim, 1))
            nn.init.xavier_uniform_(w)
            self.cross_weights.append(w)
            b = nn.Parameter(torch.zeros(input_dim))
            self.cross_bias.append(b)

    def forward(self, x0):
        x = x0
        for w, b in zip(self.cross_weights, self.cross_bias):
            xw = torch.matmul(x, w)  # (batch, 1)
            cross = x0 * xw + b.unsqueeze(0)  # (batch, input_dim)
            x = cross + x
        return x
