import json
import urllib.parse
from typing import Dict

import orjson
import requests
from dotenv import load_dotenv
from ldb_schema.paths import LocalPaths

load_dotenv()
file_cache = dict()


def load_job_file():
    job_file = 'imdb_scaled1/join_order_job_w_idxs_combined.json'

    if job_file in file_cache:
        return file_cache[job_file]

    print(f'Loading job file {job_file}')
    with open(LocalPaths().Data.json_plans.as_posix() + '/' + job_file, 'r') as f:
        job_data = orjson.loads(f.read())

    file_cache[job_file] = job_data
    return job_data


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
    job_data = load_job_file()

    matching_files = []

    # find raw plan
    for plan in job_data['query_list']:
        sql = plan['sql']
        hint = plan['hint']

        if sql == target_sql and hint == target_hint:
            matching_files.append(plan)

    return matching_files
