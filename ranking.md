# Technical Report: Experimental Approaches for Zero-Shot Ranking Model

## 1. Model Configurations and Parameters

### 1.1. `ZeroShotRankingModelBaseConfig`

This is the foundational configuration class from which all other ranking configurations inherit. It defines the common parameters for dataset handling, training, and loss computation.

**Parameters:**

*   **`name`**: `ModelName` enum value to identify the specific model configuration.
*   **Dataset Parameters:**
    *   `group_size: int = 16`: The number of plans to be grouped together for listwise ranking. The ranking loss is computed over this group.
    *   `batch_group_size: int = 16`: The number of such groups to include in a single training batch.
    *   `batch_size: int`: The total number of plans in a batch, automatically calculated as `group_size * batch_group_size`.
    *   `num_workers: int = 20`: The number of subprocesses to use for data loading.
    *   `include_W_workloads: bool = False`: Flag to include the "W" workloads in the training dataset.
*   **Training Parameters:**
    *   `lr: float = 1e-3`: The learning rate for the optimizer.
    *   `early_stopping_patience: int = 30`: The number of epochs to wait for improvement in the validation metric before halting training.
    *   `epochs: int = 125`: The maximum number of epochs to train for.
*   **Loss Parameters:**
    *   `regression_loss_name: str = "QLoss"`: The name of the pointwise regression loss component (e.g., 'MSELoss').
    *   `ranking_loss_name: str = "LambdaLoss"`: The name of the listwise ranking loss component (e.g., 'ListMLELoss').
    *   `regression_weight: float = 0`: The weight multiplier for the regression loss in the final composite loss. A value of `0` effectively disables it.
    *   `ranking_weight: float = 1.0`: The weight multiplier for the ranking loss.
    *   `loss_class_name: str = "LearnToRankWithRegressionLoss"`: The name of the class that implements the composite loss logic.
*   **Scheduling Parameters:**
    *   `scheduler_class_name: str = None`: The name of an optional learning rate scheduler class.
*   **Derived Parameters (`__attrs_post_init__`):**
    *   `loss_class_kwargs: dict`: A dictionary automatically constructed to pass all relevant loss parameters to the loss function's constructor.
    *   `optimizer_kwargs: dict`: A dictionary automatically constructed to pass `lr` to the optimizer's constructor.

### 1.2. Experiment-Specific Configurations

These classes inherit from the base configuration and introduce specific modifications to test different hypotheses.

#### `ZeroShotRankingModelConfig`
This is the vanilla ranking model, using the defaults from the base class without any further modifications. It serves as a standard baseline.

#### `LatentZeroShotRankingModelConfig`
This configuration tests a two-stage prediction process. Instead of predicting cost directly, the main model produces a latent vector, which is then fed to a smaller ranking head.

*   `latent: bool = True`: A flag that signals the `RankModel` to use the latent representation pathway.
*   `output_dim_rank: int = 2`: The dimension of the output from the final ranking head (`rank_model`).

#### `ZeroShotRankingModelConfigWithRelevancy`
This configuration introduces the concept of transforming latency values into "relevancy scores" before computing the ranking loss.

*   `use_relevancy_generator: bool = True`: Enables the `RelevancyGenerator` module for label transformation.
*   `relevancy_generator_args: dict`: A dictionary containing hyperparameters for the `RelevancyGenerator`, such as data bounds and scaling factors.

#### `ZeroShotRankingModelConfigWithSubplan`
This configuration adds an auxiliary loss term based on the costs of query subplans to provide a richer training signal.

*   `subplans_reg_loss: bool = True`: If `True`, the loss calculated on subplans is a pure regression loss, ignoring the ranking component.
*   `subplan_estimation: bool = True`: Master switch to enable or disable the entire subplan loss calculation.
*   `subplan_weight: float = 0.1`: The weight of the subplan loss when adding it to the main loss.
*   The `__attrs_post_init__` method is overridden to add `subplan_weight` to the `loss_class_kwargs` dictionary.

---

## 2. Core Modules and Logic

### 2.1. Loss Function: `LearnToRankWithRegressionLoss`

This class implements a flexible, composite loss function.

*   **Initialization**: It instantiates a ranking loss and a regression loss based on the names and arguments provided in the configuration.
*   **Forward Pass Logic**:
    1.  It calculates a pointwise `regression_loss` between model predictions and true labels. This loss is weighted by `regression_weight`.
    2.  It calculates a listwise `ranking_loss`. This can be computed either on the raw predictions or on transformed `relevancy_scores` if they are provided. This loss is weighted by `ranking_weight`.
    3.  The `only_regression` flag forces the function to compute only the regression component, which is used for the subplan loss.
    4.  The `subplan` flag indicates that the final calculated loss should be multiplied by `subplan_weight`.
    5.  The total loss is the weighted sum of the regression and ranking components.

### 2.2. Label Transformation: `RelevancyGenerator`

This module addresses the highly skewed distribution of query execution times by transforming them into well-behaved relevancy scores.

*   **Transformation (`forward` method)**:
    1.  Clamps the input latencies to a predefined [`rl_data_min`, `rl_rl_data_max`].
    2.  Applies a logarithmic function: `log(x + 1) / log(rl_log_degree)`.
    3.  Normalizes the result to a [0, 1] range.
    4.  Inverts and scales the result: `rl_scalar * (1 - normalized_value)`. This makes lower latencies correspond to higher relevancy scores.

### 2.3. Training Loop and Model Integration

The `RankModel` class and `train_epoch_rank` function integrate these components.

*   **`RankModel`**: This is the main `nn.Module` passed to the trainer.
    *   It wraps the underlying graph neural network or transformer model.
    *   Based on the config, it conditionally initializes the `RelevancyGenerator` or the secondary `rank_model` for the latent approach.
    *   Its `calculate_loss` method orchestrates the entire loss computation: it groups tensors, optionally calls the `relevancy_generator`, and passes the correct set of predictions and labels to the `loss_fxn`.
*   **`train_epoch_rank`**: This function defines the logic for a single training epoch.
    *   It correctly handles model outputs, which can be a single tensor of predictions or a tuple containing main predictions and subplan predictions.
    *   It calls `rank_model.calculate_loss` for both the main plans and, if enabled, the subplans.
    *   It sums the main loss and the subplan loss to get the final loss value for backpropagation.
    *   It includes a standard gradient clipping step for training stability.