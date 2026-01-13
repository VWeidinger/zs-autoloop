from typing import Dict, Tuple

from sympy import re


def get_general_table_col_to_ids_dict(
    db_statistics: Dict,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    # extract table and column statistics from first run file statistics
    first_run_table_stats = db_statistics[0]["table_stats"]
    first_run_column_stats = db_statistics[0]["column_stats"]

    # build column name to id mapping
    table_col_names_list = []
    for i, col_stat in enumerate(first_run_column_stats):
        assert "tablename" in col_stat, (
            f"Column stats entry does not have table_name: {col_stat}"
        )
        assert "attname" in col_stat, (
            f"Column stats entry does not have column_name: {col_stat}"
        )
        table_name = col_stat["tablename"]

        if re.match(r"^(pg_|sql_)", table_name):
            # skip system tables
            continue

        assert not table_name.startswith("pg_"), (
            f"System table found in column stats: {table_name}"
        )

        col_name = col_stat["attname"]
        table_col_names_list.append((table_name, col_name))

    # sort to have consistent ordering
    table_col_names_list = sorted(table_col_names_list)

    # assign ids
    col_name_to_id_dict = {}
    for i, key in enumerate(table_col_names_list):
        col_name_to_id_dict[key] = i

    # build table name to id mapping
    table_names = [s["relname"] for s in first_run_table_stats]
    table_names = [tn for tn in table_names if not re.match(r"^(pg_|sql_)", tn)]
    table_names = sorted(table_names)

    table_name_to_id_dict = {}
    for i, table_name in enumerate(table_names):
        table_name_to_id_dict[table_name] = i

    return table_name_to_id_dict, col_name_to_id_dict
