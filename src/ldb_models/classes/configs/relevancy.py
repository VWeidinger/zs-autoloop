import attrs as attrs


@attrs.define(frozen=True, slots=False)
class RelevancyConfigMixin:
    """Shared relevancy-generator options for ranking-style configs."""

    use_relevancy_generator: bool = True
    rl_use_log_transform: bool = True
    relevancy_generator_args: dict = attrs.field(init=False)

    """
    Relevancy generator is now handled in the collate function after sampling.
    Modes for relevancy score normalization (rl_scope):
        - 'global': Use global min/max (default, current version)
        - 'group': Use min/max per group in a batch
        - 'query': Use min/max per query (query_min can be precomputed in the sampler)
    If mask_single_plan_queries is True, single-plan queries get relevancy score -1 (masked from ranking loss).
    """
    rl_scope: str = "global"  # 'global', 'group', or 'query'
    rl_scalar: float = 10.0
    rl_data_min: float = 1e-8
    rl_data_max: float = 180.0
    rl_log_degree: float = 5.0
    mask_single_plan_queries: bool = (
        True  # If True, set relevancy score to -1 for single-plan queries
    )

    # Per-class behavior toggles (kept as plain class attributes on purpose).
    _include_mask_single_plan_queries = True

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if (
            hasattr(self, "ranking_weight")
            and self.ranking_weight == 0.0
            and self.use_relevancy_generator
        ):
            object.__setattr__(self, "use_relevancy_generator", False)

        if not self.use_relevancy_generator:
            object.__setattr__(self, "relevancy_generator_args", None)
            return

        args = dict(
            rl_scope=self.rl_scope,
            rl_data_min=self.rl_data_min,
            rl_data_max=self.rl_data_max,
            rl_scalar=self.rl_scalar,
            rl_log_degree=self.rl_log_degree,
            rl_use_log_transform=self.rl_use_log_transform,
        )
        if self._include_mask_single_plan_queries:
            args["mask_single_plan_queries"] = self.mask_single_plan_queries

        object.__setattr__(self, "relevancy_generator_args", args)
