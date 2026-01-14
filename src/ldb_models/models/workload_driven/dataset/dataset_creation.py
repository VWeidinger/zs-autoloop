import functools
from pathlib import Path
from typing import Dict, Tuple

from classes import (
    DataLoaderOptions,
    E2EModelConfig,
    ModelConfig,
    MSCNModelConfig,
    NeoModelConfig,
    QueryFormerModelConfig,
    TestSumModelConfig,
    TlstmModelConfig,
    TPoolModelConfig,
)
from classes.workload_runs import WorkloadRuns
from gensim.models import KeyedVectors
from ldb_schema.load_schema import load_json
from models.neo.neo_dataloader import neo_collator
from models.query_former.dataloader import query_former_plan_collator
from models.utils import get_general_table_col_to_ids_dict
from models.workload_driven.dataset.mscn_batching import mscn_plan_collator
from models.workload_driven.dataset.plan_tree_batching import (
    baseline_plan_collator,
    extract_dimensions,
)
from torch.utils.data import DataLoader
from training.dataset.dataset_creation import create_datasets


class InputDims:
    input_dim_plan: int
    input_dim_pred: int


class PlanModelInputDims(InputDims):
    def __init__(
        self,
        feature_statistics,
        dim_word_embedding,
        dim_word_hash,
        dim_bitmaps,
        dim_tables: int,
        dim_cols: int,
    ):
        dim_ops, dim_pred_op = extract_dimensions(feature_statistics)

        self.input_dim_pred = (
            dim_cols + dim_pred_op + dim_word_embedding + dim_word_hash
        )
        self.input_dim_plan = dim_cols + dim_tables + dim_ops + dim_bitmaps


class MSCNInputDims(InputDims):
    def __init__(
        self,
        feature_statistics,
        dim_word_embedding,
        dim_word_hash,
        dim_bitmaps,
        dim_tables: int,
        dim_cols: int,
    ):
        dim_ops, dim_pred_op, dim_joins, dim_aggs = extract_dimensions(
            feature_statistics, extended=True
        )

        # set input dimensions
        self.input_dim_table = dim_tables + dim_bitmaps
        self.input_dim_pred = (
            dim_cols + dim_pred_op + dim_word_embedding + dim_word_hash
        )
        self.input_dim_join = dim_joins
        self.input_dim_agg = dim_aggs + dim_cols


class NeoInputDims(InputDims):
    def __init__(
        self,
        dim_tables: int,
        dim_cols: int,
    ):
        self.num_cols = dim_cols
        self.num_tables = dim_tables


def make_filters_consistent(fc, db_col_mapping):
    if "col_id" in fc.keys():
        fc["col_id"] = db_col_mapping[fc["col_id"]]

    if "children" in fc.keys():
        for c in fc["children"]:
            make_filters_consistent(c, db_col_mapping)
    return fc


def get_collate_func(
    model_config: ModelConfig,
    database_statistics: dict,
    feature_statistics: dict,
    column_statistics: dict,
    word_embeddings: KeyedVectors,
    dim_word_hash: int,
    dim_word_emdb: int,
    dim_bitmaps: int,
    database_schema_info: dict,
    table_id_dict: Dict[str, int],
    col_id_dict: Dict[Tuple[str, str], int],
):
    collate_kwargs = {
        "db_statistics": database_statistics,
        "feature_statistics": feature_statistics,
        "column_statistics": column_statistics,
        "db_schema_info": database_schema_info,
        "table_id_dict": table_id_dict,
        "col_id_dict": col_id_dict,
    }

    dim_tables = len(table_id_dict)
    dim_cols = len(col_id_dict)

    if isinstance(
        model_config,
        (E2EModelConfig, TPoolModelConfig, TlstmModelConfig, TestSumModelConfig),
    ):
        collate_kwargs.update(
            dict(
                word_embeddings=word_embeddings,
                dim_word_hash=dim_word_hash,
                dim_word_emdb=dim_word_emdb,
                dim_bitmaps=dim_bitmaps,
            )
        )
        plan_collator_func = baseline_plan_collator
        input_dims: InputDims = PlanModelInputDims(
            feature_statistics,
            dim_word_emdb,
            dim_word_hash,
            dim_bitmaps,
            dim_cols=dim_cols,
            dim_tables=dim_tables,
        )

    elif isinstance(model_config, MSCNModelConfig):
        collate_kwargs.update(
            dict(
                word_embeddings=word_embeddings,
                dim_word_hash=dim_word_hash,
                dim_word_emdb=dim_word_emdb,
                dim_bitmaps=dim_bitmaps,
            )
        )
        plan_collator_func = mscn_plan_collator
        input_dims: InputDims = MSCNInputDims(
            feature_statistics,
            dim_word_emdb,
            dim_word_hash,
            dim_bitmaps,
            dim_cols=dim_cols,
            dim_tables=dim_tables,
        )

    elif isinstance(model_config, QueryFormerModelConfig):
        collate_kwargs.update(
            dict(
                word_embeddings=word_embeddings,
                dim_word_hash=dim_word_hash,
                dim_word_embedding=dim_word_emdb,
                histogram_bin_size=model_config.histogram_bin_number,
                max_num_filters=model_config.max_num_filters,
                dim_bitmaps=dim_bitmaps,
            )
        )
        plan_collator_func = query_former_plan_collator
        input_dims: InputDims = PlanModelInputDims(
            feature_statistics,
            dim_word_emdb,
            dim_word_hash,
            dim_bitmaps,
            dim_cols=dim_cols,
            dim_tables=dim_tables,
        )

    elif isinstance(model_config, NeoModelConfig):
        plan_collator_func = neo_collator
        input_dims: InputDims = NeoInputDims(
            dim_tables,
            dim_cols,
        )

    else:
        raise NotImplementedError

    collate_fn = functools.partial(plan_collator_func, **collate_kwargs)
    return collate_fn, input_dims


def create_baseline_dataloader(
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    column_statistics: Path,
    word_embeddings,
    model_config: ModelConfig,
    data_loader_options: DataLoaderOptions,
) -> Tuple[dict, DataLoader, DataLoader, DataLoader, InputDims]:
    train_loader, val_loader, test_loaders = None, None, None

    dim_bitmaps = 1000
    feature_statistics = load_json(str(statistics_file), namespace=False)
    assert feature_statistics != {}, (
        f"Feature statistics file is empty! / {statistics_file}"
    )
    dataloader_args = dict(
        batch_size=model_config.batch_size,
        shuffle=data_loader_options.shuffle,
        num_workers=model_config.num_workers,
        pin_memory=data_loader_options.pin_memory,
    )

    if column_statistics:
        column_statistics = load_json(str(column_statistics), namespace=False)
    if word_embeddings:
        word_embeddings = KeyedVectors.load(str(word_embeddings), mmap="r")
        dim_word_emdb = word_embeddings.vector_size
        dim_word_hash = dim_word_emdb
    else:
        dim_word_emdb = 0
        dim_word_hash = 0

    if workload_runs.train_workload_runs:
        # assert workload_runs.test_workload_runs == [], ("Unseen Test workload runs are not allowed when training "
        #                                                "workload driven models")
        print("Create dataloader for training, validation and test data")

        label_norm, train_dataset, val_dataset, database_statistics, db_schema_info = (
            create_datasets(
                workload_run_paths=workload_runs.train_workload_runs,
                model_config=model_config,
                val_ratio=data_loader_options.val_ratio,
                stratify_dataset=data_loader_options.stratify_dataset,
                stratify_per_wl_file=data_loader_options.stratify_per_wl_file,
            )
        )

        test_dataset, val_dataset = val_dataset.split(0.5)

        test_dataset = test_dataset.remove_timeouts()

        print(
            f"Created datasets of size: train {len(train_dataset)}, validation: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        # create run_id independent mapping of table names to ids
        table_id_dict, col_id_dict = get_general_table_col_to_ids_dict(
            database_statistics
        )

        # plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
        # database statistics but also feature statistics

        collate_fn, input_dims = get_collate_func(
            model_config=model_config,
            database_statistics=database_statistics,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            word_embeddings=word_embeddings,
            dim_word_hash=dim_word_hash,
            dim_word_emdb=dim_word_emdb,
            dim_bitmaps=dim_bitmaps,
            database_schema_info=db_schema_info,
            table_id_dict=table_id_dict,
            col_id_dict=col_id_dict,
        )
        dataloader_args.update(collate_fn=collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        test_loaders = [DataLoader(test_dataset, **dataloader_args)]
    else:
        raise Exception(
            "no training workloads provided. This is a problem since training workloads are globally used for table & col-id mapping."
        )

    if workload_runs.test_workload_runs:
        test_loaders = []
        for test_workload in workload_runs.test_workload_runs:
            _, _, test_dataset, database_statistics, db_schema_info = create_datasets(
                workload_run_paths=[test_workload],
                model_config=model_config,
                val_ratio=1.0,
                is_test_dataset=True,
                stratify_dataset=data_loader_options.stratify_dataset,
                stratify_per_wl_file=data_loader_options.stratify_per_wl_file,
            )

            test_collate_fn, input_dims = get_collate_func(
                model_config=model_config,
                database_statistics=database_statistics,
                feature_statistics=feature_statistics,
                column_statistics=column_statistics,
                word_embeddings=word_embeddings,
                dim_word_hash=dim_word_hash,
                dim_word_emdb=dim_word_emdb,
                dim_bitmaps=dim_bitmaps,
                database_schema_info=db_schema_info,
                table_id_dict=table_id_dict,
                col_id_dict=col_id_dict,
            )

            dataloader_args.update(collate_fn=test_collate_fn)
            test_loaders.append(DataLoader(test_dataset, **dataloader_args))

    return (
        None,
        feature_statistics,
        train_loader,
        val_loader,
        test_loaders,
        input_dims,
    )
