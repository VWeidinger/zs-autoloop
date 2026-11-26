import argparse
import glob
import os

from ldb_models.classes import InputFormats
from ldb_models.training.preprocessing.feature_statistics import gather_feature_statistics


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
    parser.add_argument("--parsed_plans_dirname", default=None, type=str)
    parser.add_argument(
        "--formats", default=["json", "parsed"], nargs="+", choices=["json", "parsed"]
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing statistics files"
    )
    args = parser.parse_args()

    for mode in [InputFormats[f] for f in args.formats]:
        mode_str = (
            f"{mode}" if mode != InputFormats.parsed else args.parsed_plans_dirname
        )

        json_files = []
        for workload in args.workload:
            assert workload.endswith(".json"), (
                f"Expected workload to end with .json, got {workload}"
            )
            for database in args.database:
                json_files += get_all_json_files(
                    f"{args.runs_dir}/{mode_str}/{database}/{workload}"
                )

        assert len(json_files) > 0, (
            f"No json files found: {args.runs_dir}/{mode_str}/{{database}}/{{workload}}"
        )

        target_path = f"{args.target_folder}/{mode_str}_{args.file_name}.json"
        if os.path.exists(target_path) and not args.overwrite:
            raise FileExistsError(
                f"{target_path} already exists - pass --overwrite to replace it"
            )
        print(
            f"Saving feature statistics for {json_files} workload files to {target_path}"
        )
        gather_feature_statistics(json_files, target_path)

    print("Done")
