import torch
import torch.nn as nn
from classes import NeoModelConfig
from models.workload_driven.dataset.dataset_creation import NeoInputDims
from training.losses import create_loss
from training.training.outputs import LossOutput, ModelOutput


class TreeConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.
    Value is either cost or latency.
    """

    def __init__(
        self,
        model_config: NeoModelConfig,
        input_dims: NeoInputDims,
        feature_statistics: dict,
        max_plan_depth=25,
        label_size=1,
    ):
        super(TreeConvolution, self).__init__()

        self.label_norm = None
        self.device = model_config.device
        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )
        self.max_plan_depth = max_plan_depth

        self.num_cols = input_dims.num_cols
        self.num_tables = input_dims.num_tables
        feature_size = input_dims.num_cols + input_dims.num_tables
        feature_size = 4024

        # Network to embed query features
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )

        # Tree Convolution
        self.conv = nn.Sequential(
            TreeConv1d(32 + max_plan_depth, 512, device=self.device),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 256, device=self.device),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(256, 128, device=self.device),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),
        )

        # Fully connected network to predict cost
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.normal_(p, std=0.02)
            elif "bias" in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)

    def forward(self, model_input: tuple, **kwargs) -> ModelOutput:
        out = self._forward(
            query_features=model_input[0],
            plan_encodings=model_input[1],
            indexes=model_input[2],
        )
        return ModelOutput(plan_predictions=out)

    def compute_loss(
        self, output: ModelOutput, query_stats: dict, labels: torch.Tensor
    ) -> LossOutput:
        # Loss computation
        plan_predictions = output.plan_predictions
        assert len(plan_predictions) == len(labels), (
            f"Output length {len(plan_predictions)} does not match label length {len(labels)}"
        )
        loss = plan_loss = self.loss_fxn(plan_predictions, labels)

        return LossOutput(
            loss=loss,
            plan_loss=plan_loss,
        )

    def _forward(
        self,
        query_features: torch.Tensor,
        plan_encodings: torch.Tensor,
        indexes: torch.Tensor,
    ):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors. Shaped as [batch size, query dims].
          plan_encodings: The input plan features. Shaped as [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        # First embed query features
        query_embs = self.query_mlp(query_features.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = plan_encodings.shape[-1]

        # Concat query and plan features
        query_embs = query_embs.expand(
            query_embs.shape[0], query_embs.shape[1], max_subtrees
        )
        concat = torch.cat(tensors=(query_embs, plan_encodings), dim=1)

        # Convolution and pooling
        out = self.conv((concat, indexes))
        out = self.out_mlp(out)
        return out


class TreeConv1d(nn.Module):
    """Conv1d adapted to tree data."""

    def __init__(self, in_dims, out_dims, device):
        super().__init__()
        self.device = device
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=3, stride=3)

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = self.weights(
            torch.gather(data, 2, indexes.expand(-1, -1, self._in_dims).transpose(1, 2))
        )
        zeros = torch.zeros(
            (data.shape[0], self._out_dims), device=self.device
        ).unsqueeze(2)
        feats = torch.cat((zeros, feats), dim=2)
        return feats, indexes


class TreeMaxPool(nn.Module):
    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].max(dim=2).values


class TreeAct(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return self.activation(trees[0]), trees[1]


class TreeStandardize(nn.Module):
    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        mu = torch.mean(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        s = torch.std(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        standardized = (trees[0] - mu) / (s + 1e-5)
        return standardized, trees[1]
