from typing import Dict

pk_for_table_cache = dict()


def find_pk_for_table(table_name: str, db_schema_info: Dict, database_id: int) -> str:
    if (table_name, database_id) in pk_for_table_cache:
        return pk_for_table_cache[(table_name, database_id)]
    pk_candidates = []
    for col_name, stats in db_schema_info[database_id]["table_col_info"][
        table_name
    ].items():
        if stats["pk"]:
            pk_candidates.append(col_name)

    # there might be composite primary keys, so we return the first one - this is a simplification
    assert len(pk_candidates) >= 1, (
        f"Expected at least one primary key for table {table_name}, but found {len(pk_candidates)}: {pk_candidates}"
    )

    # return the first primary key candidate
    pk_for_table_cache[(table_name, database_id)] = pk_candidates[0]
    return pk_candidates[0]


col_stats_cache = dict()


def get_col_stats_for_idx_name(
    idx_name: str,
    table_name: str,
    db_schema_info: Dict,
    database_id: int,
    db_statistics: Dict,
) -> dict:
    if idx_name is None:
        return None

    if (idx_name, database_id) in col_stats_cache:
        return col_stats_cache[(idx_name, database_id)]

    if idx_name.endswith("_pkey"):
        # special case for primary key index
        col_name = find_pk_for_table(table_name, db_schema_info, database_id)
        idx_col = f"{table_name}_{col_name}"
    else:
        assert idx_name.startswith("idx_"), (
            f"Index name {idx_name} does not start with 'idx_'."
        )
        idx_col = idx_name[4:]  # remove 'idx_' prefix

    for stats in db_statistics[database_id]["column_stats"]:
        if (
            idx_col.lower()
            == f"{stats['tablename'].lower()}_{stats['attname'].lower()}"
        ):
            col_stats_cache[(idx_name, database_id)] = stats
            return stats

    raise ValueError(
        f"Column stats for index {idx_name} not found in database {database_id}."
    )
