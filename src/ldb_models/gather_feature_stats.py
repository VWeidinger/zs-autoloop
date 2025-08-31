import argparse
import glob
import os

from classes.classes import InputFormats
from training.preprocessing.feature_statistics import gather_feature_statistics


def get_all_json_files(directory):
    if not os.path.exists(directory):
        print(f"Skipping directory {directory}: does not exist")
        return []

    # Use os.path.join to make the path OS-independent
    if os.path.isfile(directory) and directory.endswith(".json"):
        json_files = [directory]
    else:
        path = os.path.join(directory, "**", "*.json")
        json_files = glob.glob(path, recursive=True)
        json_files = [file for file in json_files if "feature_statistics" not in file]
    assert len(json_files) > 0, f"No json files found in {directory}"
    return json_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=None, nargs="*")
    parser.add_argument("--workload", default=None, nargs="*")
    parser.add_argument("--file_name", default=None, type=str)
    parser.add_argument("--target_folder", default=None)
    parser.add_argument("--runs_dir", default=None)
    args = parser.parse_args()

    for mode in [InputFormats.json, InputFormats.parsed]:
        json_files = []
        for workload in args.workload:
            for database in args.database:
                json_files += get_all_json_files(
                    f"{args.runs_dir}/{mode}/{database}/{workload}"
                )

        assert len(json_files) > 0, "No json files found"

        mode_str = f"{mode}" if mode != InputFormats.parsed else "parsed_plans2"

        target_path = f"{args.target_folder}/{mode_str}_{args.file_name}.json"
        print(
            f"Saving feature statistics for {json_files} workload files to {target_path}"
        )
        gather_feature_statistics(json_files, target_path)

    print("Done")
