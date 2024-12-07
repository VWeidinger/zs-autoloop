import collections
import json
import multiprocessing
import os
from enum import Enum

import numpy as np
import orjson
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def assemble_join_str(join_dict):
    if 'children' in join_dict:
        children_str = [f'({assemble_join_str(child)})' for child in join_dict['children']]

        join_str = f' {join_dict["operator"]} '.join(children_str)
        return join_str
    else:
        # sort by alphabet
        table_str = [f'{join_dict["table_name1"]}.{join_dict["column_name1"]}', f'{join_dict["table_name2"]}.{join_dict["column_name2"]}']
        return f" {join_dict['comparison_operator']} ".join(sorted(table_str))


def gather_values_recursively(json_dict, value_dict=None):
    if value_dict is None:
        value_dict = collections.defaultdict(list)

    if isinstance(json_dict, dict):
        for k, v in json_dict.items():
            if k == 'join':
                # manually construct join key
                value_dict['join'].append(assemble_join_str(v))
            elif k in ['hint_notices', 'sql', 'Index Cond', 'Filter','sql_string','Output']:
                # skip these values
                pass
            elif not (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, dict)):
                value_dict[k].append(v)
            elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) > 0 and \
                    (isinstance(v[0], int) or isinstance(v[0], float) or isinstance(v[0], str)):
                for v_e in v:
                    value_dict[k].append(v_e)
            else:
                gather_values_recursively(v, value_dict=value_dict)
    elif isinstance(json_dict, tuple) or isinstance(json_dict, list):
        for e in json_dict:
            gather_values_recursively(e, value_dict=value_dict)

    return value_dict


class FeatureType(Enum):
    numeric = 'numeric'
    categorical = 'categorical'

    def __str__(self):
        return self.value

def gather_values_fn(run_path):
    with open(run_path) as json_file:
        runs =orjson.loads(json_file.read())

    value_dict = gather_values_recursively(runs)
    return value_dict


def gather_feature_statistics(workload_run_paths, target):
    """
    Traverses a JSON object and gathers metadata for each key. Depending on whether the values of the key are
    categorical or numerical, different statistics are collected. This is later on used to automate the feature
    extraction during the training (e.g., how to consistently map a categorical value to an index).
    """

    with multiprocessing.Pool() as pool:
        value_dicts = list(tqdm(pool.imap(gather_values_fn, workload_run_paths), desc="Reading Runs", total=len(workload_run_paths)))

    # combine value dicts
    value_dict = collections.defaultdict(list)
    for vd in value_dicts:
        for k, v in vd.items():
            value_dict[k].extend(v)

    print("Saving")
    # save unique values for categorical features and scale and center of RobustScaler for numerical ones
    statistics_dict = dict()
    for k, values in value_dict.items():
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue

        if all([isinstance(v, int) or isinstance(v, float) or v is None for v in values]):
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)

            statistics_dict[k] = dict(max=float(np_values.max()),
                                      scale=scaler.scale_.item(),
                                      center=scaler.center_.item(),
                                      type=str(FeatureType.numeric))
        else:
            unique_values = set(values)
            statistics_dict[k] = dict(value_dict={v: id for id, v in enumerate(unique_values)},
                                      no_vals=len(unique_values),
                                      type=str(FeatureType.categorical))

    # save as json
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w') as outfile:
        json.dump(statistics_dict, outfile)
    os.chmod(target, 0o777)
