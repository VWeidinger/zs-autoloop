import argparse
import os
import glob

from training.preprocessing.feature_statistics import gather_feature_statistics


def get_all_json_files(directory):
    # Use os.path.join to make the path OS-independent
    if os.path.isfile(directory) and directory.endswith('.json'):
        json_files = [directory]
    else:
        path = os.path.join(directory, '**', '*.json')
        json_files = glob.glob(path, recursive=True)
        json_files = [file for file in json_files if "feature_statistics" not in file]
    assert len(json_files) > 0, f"No json files found in {directory}"
    return json_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=None, nargs="*")
    parser.add_argument('--workload', default=None, nargs="*")
    parser.add_argument('--target', default=None)
    args = parser.parse_args()
    json_files = []
    for workload in args.workload:
        for database in args.database:
            json_files += get_all_json_files(f'../data/runs/parsed_plans/{database}/{workload}')  # ToDo: Load correct scaling name
    assert len(json_files) == len(args.database) * len(args.workload), "Some workload files were not found"
    gather_feature_statistics(json_files, args.target)
