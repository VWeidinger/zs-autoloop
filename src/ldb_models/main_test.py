import unittest

from ldb_models.main import get_model_config


class TestMain(unittest.TestCase):
    def test_get_model_config(self):
        config = get_model_config("zeroshot_pgcost", dict())
        self.assertEqual(config.name.NAME, "zeroshot_pgcost")


if __name__ == "__main__":
    unittest.main()
