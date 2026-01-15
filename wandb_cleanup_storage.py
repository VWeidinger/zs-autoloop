import traceback
from multiprocessing import Pool

from tqdm import tqdm

import wandb

# -------------------------------
# Configuration
# -------------------------------
ENTITY = "learneddb"  # user or team
PROJECT = "lcm"

ARTIFACT_REGEX = "%per_epoch/col_plots/%"
FILE_SUFFIX = ".png"  # e.g. ".png", ".json", ".pt"

# -------------------------------

api = wandb.Api()

deleted = 0

print(f"Scanning artifacts in {ENTITY}/{PROJECT}")
print(f"Name regex: '{ARTIFACT_REGEX}'")
print(f"File suffix: '{FILE_SUFFIX}'")
print("-" * 60)

artifact_types = api.artifact_types(f"{ENTITY}/{PROJECT}")

deleted = 0

# fetch all runs in the project
runs = wandb.Api().runs(f"{ENTITY}/{PROJECT}")
print(f"Fetched {len(runs)} runs.")

run_ids = [run.id for run in runs]


def process_run(run_id):
    try:
        run_deleted = 0
        print(f"Processing run: {run_id}")
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        except Exception as e:
            print(f"Failed to fetch run {run_id}: {e}")
            return run_deleted
        for file in tqdm(run.files(pattern=ARTIFACT_REGEX), desc=run_id):
            try:
                file_name = file.name
                if file_name.endswith(FILE_SUFFIX):
                    file.delete()
                    run_deleted += 1
            except Exception:
                continue
        print(f"Run {run_id} - Deleted files: {run_deleted}")
        return run_deleted
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
        traceback.print_exc()
        return run_deleted


if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(process_run, run_ids)
        deleted = sum(results)

print("-" * 60)
print(f"Deleted artifacts:  {deleted}")
