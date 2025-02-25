import unittest
from pathlib import Path

from dotenv import load_dotenv
from ldb_schema.datasets import dataset_list_dict
from ldb_schema.paths import LocalPaths

from classes.classes import T3ModelConfig
from classes.workload_runs import WorkloadRuns
from models.t3.t3_dataloader import create_t3_dataloader


class MyTestCase(unittest.TestCase):


    def setUp(self):
        super().setUp()
        load_dotenv()

    def test_create_t3_dataloader(self):
        sample_wl_file = Path(f'{LocalPaths().Data.parsed_plans.as_posix()}/imdb_scaled1/workload_200k_s1_reduced.json')
        config = T3ModelConfig()
        target_dataset = [db for db in dataset_list_dict['zs'] if db.db_name == 'imdb'][0]
        statistics_path = config.get_statistics(source_path=LocalPaths(),
                                                dataset=target_dataset,
                                                format=config.input_format).as_posix()

        wl_runs = WorkloadRuns(train_workload_runs=[sample_wl_file])
        queries = create_t3_dataloader(workload_runs=wl_runs, model_config=config)


if __name__ == '__main__':
    unittest.main()
