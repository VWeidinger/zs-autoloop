import torch
import torch.nn as nn


class RelevancyGenerator(nn.Module):
    def __init__(self, args, device="cpu"):
        """Relevancy generator for ranking models.
        Args:
            args (dict): Dictionary of arguments containing:
                rl_data_min (float): Minimum value of the data.
                rl_data_max (float): Maximum value of the data.
                rl_scalar (float): rl_scalar factor for the relevancy scores.
                rl_log_degree (int): Degree of the logarithm to be applied.
                rl_use_log_transform (bool): If True, apply log transform. Otherwise, use linear scaling.
                adaptive (bool): If True, rl_scalar and rl_log_degree are learnable parameters.
            device (str): Device to run the model on.
        """

        super().__init__()

        self.adaptive = args["adaptive"]
        self.rl_use_log_transform = args.get(
            "rl_use_log_transform", True
        )  # Default to True for backward compatibility

        self.rl_data_min = torch.tensor(
            args["rl_data_min"], dtype=torch.float32, device=device
        )
        self.rl_data_max = torch.tensor(
            args["rl_data_max"], dtype=torch.float32, device=device
        )

        if self.adaptive:
            self.rl_scalar = torch.nn.Parameter(
                torch.tensor(args["rl_scalar"], dtype=torch.float32, device=device),
                requires_grad=True,
            )
            # Only make rl_log_degree a parameter if we are actually using it
            if self.rl_use_log_transform:
                self.rl_log_degree = torch.nn.Parameter(
                    torch.tensor(
                        args["rl_log_degree"], dtype=torch.float32, device=device
                    ),
                    requires_grad=True,
                )
            else:
                # Still define it as a tensor for attribute consistency, but it won't be used
                self.rl_log_degree = torch.tensor(
                    args.get("rl_log_degree", 2.0), dtype=torch.float32, device=device
                )
        else:
            self.rl_scalar = torch.tensor(
                args["rl_scalar"], dtype=torch.float32, device=device
            )
            self.rl_log_degree = torch.tensor(
                args.get("rl_log_degree", 2.0), dtype=torch.float32, device=device
            )

        self.device = device

        print(
            f"Initialized RelevancyGenerator with: rl_data_min={self.rl_data_min.item()}, rl_data_max={self.rl_data_max.item()}, "
            f"rl_scalar={self.rl_scalar.item()}, rl_log_degree={self.rl_log_degree.item()}, "
            f"rl_use_log_transform={self.rl_use_log_transform}, adaptive={self.adaptive}"
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        x = torch.clamp(x, min=self.rl_data_min, max=self.rl_data_max)

        if self.rl_use_log_transform:
            # Ensure rl_log_degree is > 1 to avoid math errors (log(1)=0)
            log_base = torch.log(torch.clamp(self.rl_log_degree, min=1.0001))

            # Log-transform the runtimes and the boundaries
            labels_transformed = torch.log(x + 1) / log_base
            min_transformed = torch.log(self.rl_data_min + 1) / log_base
            max_transformed = torch.log(self.rl_data_max + 1) / log_base
        else:
            # No log transform, use linear scaling
            labels_transformed = x
            min_transformed = self.rl_data_min
            max_transformed = self.rl_data_max

        # Normalize the labels
        # Add a small epsilon to the denominator to prevent division by zero if rl_data_max == rl_data_min
        denominator = max_transformed - min_transformed
        labels_transformed_normalized = (labels_transformed - min_transformed) / (
            denominator + 1e-8
        )

        # Invert the normalized labels (so smaller runtimes get higher relevancy)
        labels_transformed_normalized_inverted = 1 - labels_transformed_normalized

        # Apply rl_scalar factor
        relevancy_scores = self.rl_scalar * labels_transformed_normalized_inverted

        return relevancy_scores

    def revert(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Revert rl_scalar factor
        labels_transformed_normalized_inverted = x / self.rl_scalar

        # Revert inversion
        labels_transformed_normalized = 1 - labels_transformed_normalized_inverted

        if self.rl_use_log_transform:
            log_base = torch.log(torch.clamp(self.rl_log_degree, min=1.0001))
            min_transformed = torch.log(self.rl_data_min + 1) / log_base
            max_transformed = torch.log(self.rl_data_max + 1) / log_base

            # Denormalize the labels
            denominator = max_transformed - min_transformed
            labels_transformed = (
                labels_transformed_normalized * (denominator + 1e-8) + min_transformed
            )

            # Revert the log-transform
            original_values = torch.exp(labels_transformed * log_base) - 1
        else:
            min_transformed = self.rl_data_min
            max_transformed = self.rl_data_max

            # Denormalize the labels (linear)
            denominator = max_transformed - min_transformed
            original_values = (
                labels_transformed_normalized * (denominator + 1e-8) + min_transformed
            )

        # Clamp the values to the original range to correct for any floating point inaccuracies
        reverted_x = torch.clamp(
            original_values, min=self.rl_data_min, max=self.rl_data_max
        )

        return reverted_x

    def __repr__(self):
        return (
            f"RelevancyGenerator(rl_data_min={self.rl_data_min.item()}, rl_data_max={self.rl_data_max.item()}, "
            f"rl_scalar={self.rl_scalar.item()}, rl_log_degree={self.rl_log_degree.item()}, "
            f"rl_use_log_transform={self.rl_use_log_transform})"
        )
