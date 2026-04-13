from typing import Optional

import torch
import torch.nn as nn


class RelevancyGenerator(nn.Module):
    def __init__(
        self,
        args: dict,
        device=None,
        single_plan_mask=None,
        query_min_tensor=None,
        query_max_tensor=None,
    ):
        """Relevancy generator for ranking models.
        Args:
            args (dict): Dictionary of arguments containing:
                rl_data_min (float): Minimum value of the data (global fallback).
                rl_data_max (float): Maximum value of the data (global fallback).
                rl_scalar (float): Scaling factor for the relevancy scores.
                rl_log_degree (float): Degree of the logarithm to be applied.
                rl_use_log_transform (bool): If True, apply log transform. Otherwise, use linear scaling.
                rl_scope (str): Scope of normalization - 'global', 'group', or 'query'.
            device: Device to place tensors on (default: None, uses CPU)
        """

        super().__init__()
        self.device = device

        self.rl_use_log_transform = args.get("rl_use_log_transform", True)
        self.rl_scope = args.get("rl_scope", "global")
        self.mask_single_plan_queries = args.get("mask_single_plan_queries", True)

        self.register_buffer(
            "rl_data_min",
            torch.tensor(args["rl_data_min"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "rl_data_max",
            torch.tensor(args["rl_data_max"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "rl_scalar",
            torch.tensor(args["rl_scalar"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "group_size",
            torch.tensor(args["group_size"], dtype=torch.long, device=device),
        )
        if single_plan_mask is not None:
            self.register_buffer(
                "single_plan_mask",
                torch.tensor(single_plan_mask, dtype=torch.bool, device=device),
            )

        if self.rl_use_log_transform:
            self.register_buffer(
                "rl_log_degree",
                torch.tensor(args["rl_log_degree"], dtype=torch.float32, device=device),
            )
        if self.rl_scope == "query":
            self.register_buffer(
                "query_min_tensor",
                torch.tensor(query_min_tensor, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "query_max_tensor",
                torch.tensor(query_max_tensor, dtype=torch.float32, device=device),
            )

        log_degree_str = (
            f", rl_log_degree={self.rl_log_degree.item()}"
            if self.rl_use_log_transform
            else ""
        )
        print(
            f"Initialized RelevancyGenerator with: rl_data_min={self.rl_data_min.item()}, rl_data_max={self.rl_data_max.item()}, "
            f"rl_scalar={self.rl_scalar.item()}{log_degree_str}, "
            f"rl_use_log_transform={self.rl_use_log_transform}, rl_scope={self.rl_scope}"
        )

    def forward(
        self,
        inputs: torch.Tensor,
        query_stats: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute relevancy scores for inputs based on the configured scope.

        Args:
            inputs: Tensor of plan runtimes (in seconds).
            query_stats: Dictionary of query-level statistics. Required for 'query' scope.
                Should contain 'sample_idxs' - tensor of sample indices corresponding to inputs.
        Returns:
            Tensor of relevancy scores.
        """
        sample_idxs = query_stats.get("sample_idxs", None)
        sample_idxs = (
            torch.tensor(sample_idxs, dtype=torch.long, device=self.device)
            if sample_idxs is not None
            else None
        )

        # Exclude single-plan queries if single_plan_mask is provided and sample_idxs are available
        if self.mask_single_plan_queries and self.single_plan_mask is not None:
            if sample_idxs is not None:
                batch_mask = self.single_plan_mask[sample_idxs]
                inputs = inputs.clone()  # Avoid modifying original inputs
                # drop single-plan queries entirely
                inputs = inputs[~batch_mask]
                sample_idxs = sample_idxs[~batch_mask]

        if self.rl_scope == "global":
            relevancy_scores = self._compute_global(inputs)
        elif self.rl_scope == "group":
            relevancy_scores = self._compute_group(inputs)
        elif self.rl_scope == "query":
            if sample_idxs is None:
                raise ValueError(
                    "sample_idxs must be provided in query_stats for 'query' scope"
                )
            relevancy_scores = self._compute_query(inputs, sample_idxs)
        else:
            raise ValueError(
                f"Unknown rl_scope: {self.rl_scope}. Must be 'global', 'group', or 'query'."
            )

        return relevancy_scores

    def _compute_global(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute relevancy scores using global min/max."""
        data_min = self.rl_data_min
        data_max = self.rl_data_max
        return self._normalize_and_scale(inputs, data_min, data_max)

    def _compute_group(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute relevancy scores using group-level min/max. Fully vectorized."""
        n_plans = inputs.shape[0]
        n_groups = n_plans // self.group_size
        remainder = n_plans % self.group_size

        # Process complete groups with full vectorization
        if n_groups > 0:
            complete_inputs = inputs[: n_groups * self.group_size].reshape(
                n_groups, self.group_size
            )
            # Compute min/max per group along dim=1
            group_mins = torch.clamp(
                complete_inputs.min(dim=1, keepdim=True).values,
                min=self.rl_data_min,
                max=self.rl_data_max,
            )
            group_maxs = torch.clamp(
                complete_inputs.max(dim=1, keepdim=True).values,
                min=self.rl_data_min,
                max=self.rl_data_max,
            )

            # Expand to match group shape for broadcasting
            group_mins = group_mins.expand(-1, self.group_size)
            group_maxs = group_maxs.expand(-1, self.group_size)

            # Vectorized normalization for all complete groups at once
            complete_scores = self._normalize_and_scale(
                complete_inputs, group_mins, group_maxs
            ).reshape(-1)
        else:
            complete_scores = torch.empty(0, dtype=inputs.dtype)

        # Process remainder plans as a separate group
        if remainder > 0:
            remainder_inputs = inputs[n_groups * self.group_size :]
            remainder_min = torch.clamp(
                remainder_inputs.min(), min=self.rl_data_min, max=self.rl_data_max
            )
            remainder_max = torch.clamp(
                remainder_inputs.max(), min=self.rl_data_min, max=self.rl_data_max
            )
            remainder_scores = self._normalize_and_scale(
                remainder_inputs, remainder_min, remainder_max
            )
            return torch.cat([complete_scores, remainder_scores])

        return complete_scores

    def _compute_query(
        self, inputs: torch.Tensor, sample_idxs: torch.Tensor
    ) -> torch.Tensor:
        """Compute relevancy scores using pre-computed query-level min/max."""
        # Index pre-computed tensors by sample_idxs
        batch_query_mins = self.query_min_tensor[sample_idxs]
        batch_query_maxs = self.query_max_tensor[sample_idxs]

        # Clamp to be at least/most the global bounds
        batch_query_mins = torch.clamp(
            batch_query_mins, min=self.rl_data_min, max=self.rl_data_max
        )
        batch_query_maxs = torch.clamp(
            batch_query_maxs, min=self.rl_data_min, max=self.rl_data_max
        )

        # Compute relevancy scores using vectorized normalization
        relevancy_scores = self._normalize_and_scale(
            inputs, batch_query_mins, batch_query_maxs
        )

        return relevancy_scores

    def _normalize_and_scale(
        self, inputs: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor
    ) -> torch.Tensor:
        """Apply normalization and scaling to inputs."""
        # Clamp inputs to be within [data_min, data_max]
        inputs_clamped = torch.clamp(inputs, min=data_min, max=data_max)

        if self.rl_use_log_transform:
            # Ensure rl_log_degree is > 0 to avoid math errors
            log_degree = torch.clamp(self.rl_log_degree, min=0.0001)

            # Log-transform the runtimes and the boundaries
            inputs_transformed = torch.log1p(log_degree * inputs_clamped)
            min_transformed = torch.log1p(log_degree * data_min)
            max_transformed = torch.log1p(log_degree * data_max)
        else:
            # No log transform, use linear scaling
            inputs_transformed = inputs_clamped
            min_transformed = data_min
            max_transformed = data_max

        # Normalize the inputs
        denominator = max_transformed - min_transformed
        inputs_transformed_normalized = (inputs_transformed - min_transformed) / (
            denominator + 1e-8
        )

        # Invert the normalized inputs (so smaller runtimes get higher relevancy)
        inputs_transformed_normalized_inverted = 1 - inputs_transformed_normalized

        # Apply rl_scalar factor
        relevancy_scores = self.rl_scalar * inputs_transformed_normalized_inverted

        return relevancy_scores

    def __repr__(self):
        log_degree_str = (
            f", rl_log_degree={self.rl_log_degree.item()}"
            if self.rl_use_log_transform
            else ""
        )
        return (
            f"RelevancyGenerator(rl_data_min={self.rl_data_min.item()}, rl_data_max={self.rl_data_max.item()}, "
            f"rl_scalar={self.rl_scalar.item()}{log_degree_str}, "
            f"rl_use_log_transform={self.rl_use_log_transform}, rl_scope={self.rl_scope})"
        )
