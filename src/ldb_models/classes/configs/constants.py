from enum import Enum

import attrs as attrs
from attr import field


@attrs.define(frozen=True, slots=False)
class Name:
    NAME: str = field(default=None)
    DISPLAY_NAME: str = field(default=None)


class InputFormats(Enum):
    parsed = "parsed_plans"
    baseline_parsed = "augmented_plans"
    json = "json"


class ModelName:
    POSTGRES: Name = Name("postgres", "Sc. Postgres")
    POSTGRES_STRATIFIED = Name("postgres_stratified", "Sc. Postgres (stratified)")
    FLAT: Name = Name("flat", "Flat Vector")
    FLAT_ACT_CARDS = Name("flat_act_cards", "Flat Vector (act. card.)")
    FLAT_STRATIFIED = Name("flat_stratified", "Flat Vector (stratified)")
    SPN = Name("spn", "SPN")
    MSCN = Name("mscn", "MSCN")
    E2E = Name("e2e", "E2E")

    ZEROSHOT = Name("zeroshot", "Zero-Shot")
    ZEROSHOT_RANKING_BASE = Name("zs_rank_base", "Zero-Shot (Ranking, Base)")
    ZEROSHOT_RANKING_LATENT = Name("zs_rank_lt", "Zero-Shot (Ranking, Latent)")
    ZEROSHOT_RANKING_LATENT_RELEVANCY = Name(
        "zs_rank_lt_rl", "Zero-Shot (Ranking, Latent, Relevancy)"
    )
    ZEROSHOT_RANKING = Name("zs_rank", "Zero-Shot (Ranking)")
    ZEROSHOT_RANKING_ACT_CARD = Name("zs_rank_ac", "Zero-Shot (Ranking, Act. Card)")
    ZEROSHOT_RANKING_RELEVANCY = Name("zs_rank_rl", "Zero-Shot (Ranking, Relevancy)")
    ZEROSHOT_RANKING_SUBPLAN = Name("zs_rank_sp", "Zero-Shot (Ranking, Subplan)")

    ZEROSHOT_RANKING_LATENT_CASCADE = Name(
        "zs_rank_lt_casc", "Zero-Shot (Ranking, Latent, Cascade)"
    )
    ZEROSHOT_RANKING_LATENT_TOURNAMENT = Name(
        "zs_rank_lt_tour", "Zero-Shot (Ranking, Latent, Tournament)"
    )
    ZEROSHOT_RANKING_LATENT_UNIFIED = Name(
        "zs_rank_lt_unif", "Zero-Shot (Ranking, Latent, Unified)"
    )

    MAMBA_RANKING = Name("mamba_rank", "Mamba (Ranking)")

    SUMO = Name("zs_sumo_enc", "Sum of Operators")
    SUMO_RELEVANCY = Name("zs_sumo_rl", "Sum of Operators (Relevancy)")
    SUM_OF_OPERATORS = Name("zs_sop", "Sum of Operators")
    SUM_OF_OPERATORS_RELEVANCY = Name("zs_sop_rl", "Sum of Operators (Relevancy)")
    SUM_OF_OPERATORS_NO_CARD = Name("zs_sop_nc", "Sum of Operators (No Card)")
    SUM_OF_OPERATORS_NO_RANKING = Name("zs_sop_nr", "Sum of Operators (No Ranking)")
    SUM_OF_OPERATORS_LOOP = Name("zs_sop_loop", "Sum of Operators (Loop-aware)")
    SUMO_BASELINE = Name("zs_sumo", "Sum of Operators (Baseline)")
    MAMBA_SUM_OF_OPERATORS = Name("mamba_sop", "Mamba (Sum of Operators)")

    ZS_LOW_MIN = Name("zs_low_min", "Zero-Shot (low min)")
    ZEROSHOT_ACT_CARD = Name("zeroshot_act_card", "Zero-Shot (act. card)")
    ZS_FROM_TIMEOUT = Name(
        "zeroshot_learn_from_timeout", "Zero-Shot (learn from timeout)"
    )
    ZS_RANK_LOSS = Name("zeroshot_rank_loss", "Zero-Shot (Rank Loss)")
    ZS_STRATIFIED = Name("zeroshot_stratified", "Zero-Shot (stratified)")
    ZS_XL = Name("zeroshot_xl", "Zero-Shot (XL)")
    ZS_XL_LAYER_NORM = Name("zeroshot_xl_layer_norm", "Zero-Shot (XL, Layer Norm)")
    ZS_LAYER_NORM = Name("zeroshot_layer_norm", "Zero-Shot (Layer Norm)")
    ZS_PGCOST = Name("zeroshot_pgcost", "Zero-Shot (pgcost)")
    ZS_WITH_HINTS = Name("zeroshot_with_hints", "Zero-Shot (with hints)")
    ZS_ONLY_HINTS = Name("zeroshot_only_hints", "Zero-Shot (only hints)")
    ZS_2DIM = Name("zeroshot_2dim", "Zero-Shot (2dim)")
    ZS_STRATIFIED_PG_COST_WITH_HINT_PLAN = Name(
        "zeroshot_stratified_pg_cost_with_hint_plan",
        "Zero-Shot (stratified, PG cost, with hint plan)",
    )
    ZS_PG_COST_WITH_HINT_PLAN = Name(
        "zeroshot_pg_cost_with_hint_plan", "Zero-Shot (PG cost, with hint plan)"
    )
    ZS_VAE = Name("zeroshot_vae", "Zero-Shot (VAE)")
    QPP_NET = Name("qppnet", "QPP-Net")
    QPP_NET_NO_COSTS = Name("qppnet_no_costs", "QPP-Net\n(w/o PG costs)")
    QPP_NET_ACT_CARDS = Name("qppnet_act_cards", "QPP-Net (act. card.)")
    DACE = Name("dace", "DACE")
    DACE_ACT_CARDS = Name("dace_act_cards", "DACE (act. card.)")
    DACE_NO_COSTS = Name("dace_no_costs", "DACE\n(w/o PG costs)")
    DACE_STRATIFIED = Name("dace_stratified", "DACE (stratified)")
    DACE_WITH_HINTS = Name("dace_with_hints", "DACE (with hints)")
    DACE_ONLY_HINTS = Name("dace_only_hints", "DACE (only hints)")
    QUERY_FORMER = Name("query_former", "QueryFormer")
    T3 = Name("t3", "T3")
    T3Flat = Name("t3_flat", "T3 (Flat)")
    T3Tree = Name("t3_tree", "T3 (Tree)")
    NEO = Name("neo", "Neo")
    MAMBA = Name("mamba", "Mamba")
    MAMBA_TANH = Name("mamba_tanh", "Mamba (Tanh)")
    MAMBA_RELU = Name("mamba_relu", "Mamba (ReLU)")


class ModelType:
    WL_DRIVEN = "wl_driven"
    WL_AGNOSTIC = "wl_agnostic"
