from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_run(run):
    run_deleted = 0
    print(f"Processing run: {run.id}")
    for file in tqdm(run.files(pattern=ARTIFACT_REGEX), desc=run.id):
        file_name = file.name
        if file_name.endswith(FILE_SUFFIX):
            file.delete()
            run_deleted += 1
    print(f"Run {run.id} - Deleted files: {run_deleted}")
    return run_deleted


with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_run, run) for run in runs]
    for future in as_completed(futures):
        deleted += future.result()

print("-" * 60)
print(f"Deleted artifacts:  {deleted}")
