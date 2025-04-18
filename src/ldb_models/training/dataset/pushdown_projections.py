from functools import lru_cache
from typing import Dict, List

from ldb_schema.load_schema import get_json_schema


def extract_cols_from_aggr(aggr_list: List[Dict], only_w_aggr: bool = True) -> List[Dict]:
    cols = []
    for aggr in aggr_list:
        # check if it is an aggregation
        if only_w_aggr:
            if aggr['aggregation'] is None or aggr['aggregation'] == 'None':
                continue

        cols.extend(aggr['columns'])

    return cols


def extract_cols_from_filter(filter: Dict) -> List[Dict]:
    # extract the columns from the filter
    if 'children' in filter:
        cols = []
        for child in filter['children']:
            cols.extend(extract_cols_from_filter(child))
        return cols
    else:
        return [{
            'table_name': filter['table_name'],
            'col_name': filter['col_name'],
            'col_id': filter['col_id'],
            'child_ref': None
        }]


def extract_cols_from_join(join: Dict) -> List[Dict]:
    if join['operator'] in ['AND']:
        # call recursively on children
        cols = []
        for child in join['children']:
            cols.extend(extract_cols_from_join(child))
        return cols

    assert 'table_name1' in join, join

    # extract the columns from the join condition
    return [{
        'table_name': join['table_name1'],
        'col_name': join['column_name1'],
        'col_id': join['col_id1'],
        'child_ref': None
    },
        {
            'table_name': join['table_name2'],
            'col_name': join['column_name2'],
            'col_id': join['col_id2'],
            'child_ref': None
        }
    ]


def apply_projection_pushdown(plan: Dict, database_statistics: Dict, outcol_list: List[Dict] = None,
                              dataset: str = None, db_id: int = None) -> (Dict, List[str]):
    if outcol_list is None:
        is_root = True
        outcol_list = []
        dataset = plan['dataset']
        db_id = plan['database_id']
    else:
        is_root = False

    # create a copy of the outcol list
    parent_col_list = outcol_list[:]

    # update columns necessary to process this operator
    if 'join' in plan['plan_parameters']:
        outcol_list.extend(extract_cols_from_join(plan['plan_parameters']['join']))

    if 'filter' in plan['plan_parameters']:
        outcol_list.extend(extract_cols_from_filter(plan['plan_parameters']['filter']))

    if 'output_columns' in plan['plan_parameters']:
        outcol_list.extend(extract_cols_from_aggr(plan['plan_parameters']['output_columns']))

    # remove duplicates identified by col id
    outcol_list = list({col['col_id']: col for col in outcol_list}.values())

    # sort by col id
    outcol_list.sort(key=lambda x: x['col_id'])

    # extract tables that are accessed by children
    child_tables = []
    if 'table_name' in plan['plan_parameters']:
        child_tables.append(plan['plan_parameters']['table_name'])

    # process children
    if 'children' in plan:
        for child in plan['children']:
            child_tables.extend(
                apply_projection_pushdown(child, database_statistics=database_statistics, outcol_list=outcol_list,
                                          dataset=dataset, db_id=db_id)[1])

    if not is_root and len(parent_col_list) == 0:
        # this might happen because of count(*) on top - keep just the first output column

        cols = []
        if 'output_columns' in plan['plan_parameters'] and len(plan['plan_parameters']['output_columns']) > 0:
            # reconstruct from out cols
            cols = extract_cols_from_aggr(plan['plan_parameters']['output_columns'], only_w_aggr=False)

        if len(cols) == 0 and 'join' in plan['plan_parameters']:
            # reconstruct from join
            cols = extract_cols_from_join(plan['plan_parameters']['join'])

        if len(cols) == 0 and 'filter' in plan['plan_parameters']:
            # reconstruct from filter
            cols = extract_cols_from_filter(plan['plan_parameters']['filter'])

        if len(cols) == 0 and len(plan['children']) == 0:
            # still no cols extracted and no child exists: we just use the pk column
            # lookup pk col name & id
            pk_col_name, pk_col_id = get_pk_col_of_table(plan, plan['plan_parameters']['table_name'],
                                                         database_statistics, dataset=dataset, db_id=db_id)
            cols = [{
                'col_id': pk_col_id,
                'child_ref': None,
                'table_name': plan['plan_parameters']['table_name'],
                'table_alias': plan['plan_parameters']['table_alias'] if 'table_alias' in plan[
                    'plan_parameters'] else None,
                'col_name': pk_col_name,
            }]

        if len(cols) == 0:
            # still no cols extracted: this is a problem
            raise Exception(f'No output column can be reconstructed for {plan}')

        assert len(cols) > 0, plan
        parent_col_list.append(cols[0])

    node_output_cols = []
    cols_already_covered_in_outcol = []

    # manually add aggregation columns (from existing output columns list of node)
    if 'output_columns' in plan['plan_parameters']:
        for outcol in plan['plan_parameters']['output_columns']:
            # keep all output cols that are aggregations - or keep them if they are in the root node
            if (outcol['aggregation'] is not None and outcol['aggregation'] != 'None') or is_root:
                # keep it: it is an aggregation
                if outcol not in node_output_cols:
                    node_output_cols.append(outcol)
                    cols_already_covered_in_outcol.extend(outcol['columns'])

    # check which of the columns needed by parent operators can be produced by this (+child) nodes
    for col in parent_col_list:
        # check if col is from table scanned in children
        if col['table_name'] in child_tables and col not in cols_already_covered_in_outcol:
            node_output_cols.append({
                "aggregation": "None",
                "columns": [col
                            ],
                "udf_name": None,
                "udf_output": "False",
                "child_ref": False
            })
            cols_already_covered_in_outcol.append(col)

    if len(plan['children']) == 0 and len(node_output_cols) == 0:
        if is_root:
            # this is a "SELECT * FROM table WHERE ... " case

            # lookup pk col name & id
            pk_col_name, pk_col_id = get_pk_col_of_table(plan, plan['plan_parameters']['table_name'],
                                                         database_statistics)

            node_output_cols.append({
                "aggregation": "None",
                "columns": [{
                    'col_id': pk_col_id,
                    'child_ref': None,
                    'table_name': plan['plan_parameters']['table_name'],
                    'table_alias': plan['plan_parameters']['table_alias'] if 'table_alias' in plan[
                        'plan_parameters'] else None,
                    'col_name': pk_col_name,
                }
                ],
                "udf_name": None,
                "udf_output": "False",
                "child_ref": False
            })
        else:
            # this could be an index only scan - columns will be used in join, but join has been moved to idx scan so no usage in parent
            # keep existing output columns
            node_output_cols.extend(plan['plan_parameters']['output_columns'])

    # update the output columns
    assert len(node_output_cols) > 0, f'there always has to be an output column: {plan} / {parent_col_list}'
    plan['plan_parameters']['output_columns'] = node_output_cols

    return plan, child_tables


@lru_cache(maxsize=128)
def get_schema_wrapper(dataset: str):
    return get_json_schema(dataset=dataset)


def get_pk_col_of_table(plan, table_name, dataset_statistics, dataset: str = None, db_id: int = None) -> str:
    if dataset is None:
        assert 'dataset' in plan, f'does not found dataset info in {plan}'
        dataset = plan['dataset']
        db_id = plan['database_id']

    schema = get_schema_wrapper(dataset=dataset)

    col_list = schema['table_col_info'][table_name]
    assert len(col_list) > 0, f'no columns found for table {table_name} in {schema}'

    # filter out pk cols
    pk_cols = [col_name for col_name, col in col_list.items() if col['pk']]
    assert len(pk_cols) > 0, f'no pk cols found for table {table_name} in {schema}'
    pk_col_name = pk_cols[0]

    # lookup pk col id
    assert 'column_stats' in dataset_statistics[db_id], f'no column stats found in {dataset_statistics[db_id]}'
    cols = [i for i, c in enumerate(dataset_statistics[db_id]['column_stats']) if
            c['tablename'] == table_name and c['attname'] == pk_col_name]
    assert len(
        cols) == 1, f'not exactly one match found. Isntead: {cols} for {dataset_statistics[db_id]["column_stats"]}'
    return pk_cols[0], cols[0]
