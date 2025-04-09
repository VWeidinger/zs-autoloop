import json
import urllib.parse
from typing import Dict

import requests
from dotenv import load_dotenv

import wandb

load_dotenv()
file_cache = dict()
file_cache_parsed = dict()


def load_job_file():
    job_file = 'imdb_scaled1/join_order_job_w_idxs_combined.json'

    if job_file in file_cache:
        return file_cache[job_file], file_cache_parsed[job_file]

    print(f'Loading job file {job_file}')
    with open(LocalPaths().Data.json_plans.as_posix() + '/' + job_file, 'r') as f:
        job_data = orjson.loads(f.read())

    print(f'Loaded parsed job file {job_file}')
    with open(LocalPaths().Data.parsed_plans.as_posix() + '/' + job_file, 'r') as f:
        job_data_parsed = orjson.loads(f.read())

    file_cache[job_file] = job_data
    file_cache_parsed[job_file] = job_data_parsed
    return job_data, job_data_parsed


def create_explain_dalibo_entry(query, plan, title):
    if isinstance(plan, Dict):
        plan = json.dumps(plan)

    url = 'https://explain.dalibo.com/new'
    form_data = {
        'query': urllib.parse.quote(query),
        'title': urllib.parse.quote(title),
        'plan': plan
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=form_data
    )

    if response.status_code == 200:
        return response.url
    else:
        raise Exception(f'Failed to create explain entry: {response.text}')


def search_plan(target_sql, target_hint):
    job_data, job_data_parsed = load_job_file()

    matching_files = []

    # find raw plan
    for plan in job_data['query_list']:
        sql = plan['sql']
        hint = plan['hint']

        if sql == target_sql and hint == target_hint:
            matching_files.append(plan)

    matching_parsed_plans = []
    # find parsed plan
    for plan in job_data_parsed['parsed_plans']:
        sql = plan['sql']
        hint = plan['hint']

        if sql == target_sql and hint == target_hint:
            matching_parsed_plans.append(plan)

    return matching_files, matching_parsed_plans


def get_table_df(url: str, label: str, api):
    print(url)
    artifact = api.artifact(url)
    artifact.download()

    table = artifact.get(label)
    return table.get_dataframe()


def get_dataframes_from_run(run: str, df_cache: Dict, api, wandb_project: str = 'learneddb/lcm') -> Dict[str, str]:
    if run in df_cache:
        print(f'Found run {run} in cache')
        return df_cache[run]

    if wandb.run is None:
        wandb.init()

    df_dict = {}

    instance = api.run(f'{wandb_project}/{run}')

    for key in instance.summary.keys():
        if key.endswith('pred_table'):
            df_dict[key] = get_table_df(f'{wandb_project}/run-{run}-{key}:latest', key, api=api)

    df_cache[run] = df_dict

    # save cache to file
    import pickle
    with open('wandb_df_cache.pkl', 'wb') as f:
        pickle.dump(df_cache, f)

    return df_dict


def compute_optimal_picked_instances(df):
    grouped_df = df.groupby('sql')

    opt_runtimes = []
    picked_runtimes = []
    opt_plan_instances = []
    picked_plan_instances = []
    sqls = []

    for sql, group in grouped_df:
        min_runtime = group['label'].min()

        # get the picked plan runtime (the one with the lowest prediction)
        picked_plan = group.loc[group['prediction'].idxmin()]
        picked_plan_runtime = picked_plan['label']

        opt_runtimes.append(min_runtime)
        opt_plan_instances.append(group.loc[group['label'].idxmin()])
        picked_runtimes.append(picked_plan_runtime)
        picked_plan_instances.append(picked_plan)
        sqls.append(sql)

    return opt_runtimes, picked_runtimes, opt_plan_instances, picked_plan_instances, sqls


from ldb_schema.paths import LocalPaths
from ldb_schema.load_schema import load_json
import orjson

runs_file_cache = dict()


def find_plan_by_sql(sql: str, hint: str, path):
    if path in runs_file_cache:
        run = runs_file_cache[path]
    else:
        try:
            run = load_json(str(path))
            runs_file_cache[path] = run
        except orjson.JSONDecodeError:
            print(f'Error reading {path} - skip')
            return None

    plans = run['parsed_plans']

    matching_plans = []
    for p in plans:
        if p['sql'] == sql and p['hint'] == hint:
            matching_plans.append(p)

    return matching_plans
