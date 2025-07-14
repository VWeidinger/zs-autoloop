import unittest
from ldb_models.models.white_mamba.mamba_dataset import construct_filter_flat_vector


class TestMambaDataset(unittest.TestCase):
    def test_construct_filter_flat_vector(self):
        filter_dict = {
            "operator": "AND",
            "is_index_cond": False,
            "children": [
                {
                    "table_name1": "char_name",
                    "table_alias1": "chn",
                    "column_name1": "surname_pcode",
                    "col_id1": 92,
                    "operator": "=",
                    "table_name2": "aka_name",
                    "table_alias2": "ak",
                    "column_name2": "name_pcode_nf",
                    "col_id2": 70,
                    "is_index_cond": False,
                },
                {
                    "table_name1": "name",
                    "table_alias1": "n",
                    "column_name1": "name_pcode_nf",
                    "col_id1": 13,
                    "operator": "=",
                    "table_name2": "aka_name",
                    "table_alias2": "ak",
                    "column_name2": "name_pcode_cf",
                    "col_id2": 69,
                    "is_index_cond": False,
                },
            ],
        }

        flat_vector = construct_filter_flat_vector(filter_dict)
        expected_vector = [
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        self.assertEqual(flat_vector, expected_vector)


if __name__ == "__main__":
    unittest.main()
