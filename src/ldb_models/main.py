import argparse
import copy
import functools
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import attrs as attrs
import numpy as np
import torch
import torch.optim as opt
import wandb
from ldb_models.classes import (
    CascadeRankingModelConfig,
    DACEModelConfig,
    DataLoaderOptions,
    E2EModelConfig,
    FlatModelConfig,
    MambaModelConfig,
    MambaRankingModelConfig,
    MambaSumOfOperatorsModelConfig,
    ModelConfig,
    MSCNModelConfig,
    NeoModelConfig,
    QPPNetModelConfig,
    QueryFormerModelConfig,
    ScaledPostgresModelConfig,
    SPNModelConfig,
    SumOfOperatorsModelConfig,
    SUMOModelConfig,
    T3ModelConfig,
    TabularModelConfig,
    TlstmModelConfig,
    ZeroShotModelConfig,
    ZeroShotRankingModelBaseConfig,
    ZeroShotVAEModelConfig,
)
from ldb_models.classes.model_config_helper import get_model_config, shorten_model_name
from ldb_models.classes.workload_runs import WorkloadRuns
from ldb_models.models.dace.dace_dataset import create_dace_dataloader
from ldb_models.models.dace.dace_model import DACELora
from ldb_models.models.neo.neo_model import TreeConvolution
from ldb_models.models.qppnet.qppnet_dataloader import create_qppnet_dataloader
from ldb_models.models.qppnet.qppnet_model import QPPNet
from ldb_models.models.query_former.model import QueryFormer
from ldb_models.models.ranking.ranking import RankModel
from ldb_models.models.sum_of_operators import SumOfOperatorsModel
from ldb_models.models.sumo.sumo import SUMO
from ldb_models.models.tabular.train_tabular_baseline import (
    predict_tabular_model,
    train_tabular_model,
)
from ldb_models.models.white_mamba.mamba_model import MambaModel
from ldb_models.models.white_mamba.mamba_sum_of_operators import MambaSumOfOperatorsModel
from ldb_models.models.workload_driven.dataset.dataset_creation import create_baseline_dataloader
from ldb_models.models.workload_driven.model.e2e_model import E2EModel
from ldb_models.models.workload_driven.model.mscn_model import MSCNModel
from ldb_models.models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from ldb_models.models.zeroshot.specific_models.postgres_zero_shot_vae import (
    PostgresZeroShotModelVAE,
)
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from ldb_models.train import train_model
from ldb_models.training.bound_predictions_with_history import PredictionBounder
from ldb_models.training.dataset.dataset_creation import (
    DataloaderOutputKwargs,
    create_zeroshot_dataloader,
)
from ldb_models.training.dataset.plan_dataset import OHEMPlanDataset
from ldb_models.training.scheduler import init_scheduler
from ldb_models.training.training.checkpoint import load_checkpoint
from ldb_models.training.training.provenance import collect_training_info, write_training_info
from ldb_models.training.training.metrics import Metric, init_metrics
from ldb_models.training.training.test import test_model
from wandb.apis.public import Run


def init_model(
    statistics_file: Path,
    column_statistics: Path,
    config: ModelConfig,
    workload_runs: WorkloadRuns,
    word_embeddings: Path,
    quick_run: bool = False,
    assemble_data_up_front: bool = False,
) -> Tuple[Optional[nn.Module], DataLoader, DataLoader, List[DataLoader], Dict]:
    if isinstance(
        config, (ZeroShotModelConfig, ZeroShotVAEModelConfig, MambaModelConfig)
    ):
        shuffle = (
            config.shuffle_train_data if hasattr(config, "shuffle_train_data") else True
        )

        # dataloader_output contains single_plan_mask, query_min_tensor, query_max_tensor
        dataloader_output = DataloaderOutputKwargs()
        (
            label_norm,
            feature_statistics,
            train_loader,
            val_loader,
            test_loaders,
            dataloader_output,
        ) = create_zeroshot_dataloader(
            workload_runs=workload_runs,
            statistics_file=statistics_file,
            model_config=config,
            data_loader_options=DataLoaderOptions(shuffle=shuffle),
            quick_run=quick_run,
            assemble_graphs_up_front=assemble_data_up_front,
        )

        print(f"Model device: {config.device}", flush=True)

        if isinstance(config, CascadeRankingModelConfig):
            # Instantiate the main RANKING model (the second, more complex stage)
            ZSmodel = PostgresZeroShotModel(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )

            # Instantiate and load the pretrained FILTER model (the first stage)
            print("--- Initializing cascade filter model for evaluation ---")
            # Pass device at construction time since ModelConfig is frozen and
            # cannot be mutated after instantiation.
            filter_model_config = get_model_config(
                model_type=config.cascade_filter_model_type,
                m_args={"device": config.device},
            )
            if isinstance(filter_model_config, SUMOModelConfig):
                filter_model_shell = SUMO(
                    model_config=filter_model_config,
                    feature_statistics=feature_statistics,
                    label_norm=label_norm,
                )
            elif isinstance(filter_model_config, SumOfOperatorsModelConfig):
                filter_model_shell = SumOfOperatorsModel(
                    model_config=filter_model_config,
                    feature_statistics=feature_statistics,
                    label_norm=label_norm,
                )
            elif isinstance(filter_model_config, ZeroShotRankingModelBaseConfig):
                filter_model_shell = PostgresZeroShotModel(
                    model_config=filter_model_config,
                    feature_statistics=feature_statistics,
                    label_norm=label_norm,
                )
            else:
                raise NotImplementedError(
                    f"Cascade filter model type '{config.cascade_filter_model_type}' is not supported."
                )
            filter_model_shell = RankModel(
                model=filter_model_shell,
                model_config=filter_model_config,
                **dataloader_output.as_kwargs(),
            )

            print(
                f"Loading filter model '{config.cascade_filter_model_name}' from '{config.cascade_filter_model_dir}'"
            )
            _, _, _, loaded_filter_model, _, _, _, _, _ = load_checkpoint(
                model=filter_model_shell,
                manual_model_dir=config.cascade_filter_model_dir,
                config=filter_model_config,
                manual_model_name=config.cascade_filter_model_name,
                target_path=None,
                optimizer=None,
                metrics=[],
            )

            loaded_filter_model.eval()

            model = RankModel(
                model=ZSmodel,
                model_config=config,
                filter_model=loaded_filter_model,
                **dataloader_output.as_kwargs(),
            )

        elif isinstance(config, SUMOModelConfig):
            ZSmodel: nn.Module = SUMO(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )
            model: nn.Module = RankModel(
                model=ZSmodel,
                model_config=config,
                **dataloader_output.as_kwargs(),
            )
        elif isinstance(config, SumOfOperatorsModelConfig):
            ZSmodel: nn.Module = SumOfOperatorsModel(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )
            model: nn.Module = RankModel(
                model=ZSmodel,
                model_config=config,
                **dataloader_output.as_kwargs(),
            )
        elif isinstance(config, MambaSumOfOperatorsModelConfig):
            ZSmodel: nn.Module = MambaSumOfOperatorsModel(
                model_config=config, feature_statistics=feature_statistics
            )
            model: nn.Module = RankModel(
                model=ZSmodel,
                model_config=config,
                **dataloader_output.as_kwargs(),
            )
        elif isinstance(config, MambaRankingModelConfig):
            ZSmodel: nn.Module = MambaModel(
                model_config=config, feature_statistics=feature_statistics
            )
            model: nn.Module = RankModel(
                model=ZSmodel,
                model_config=config,
                **dataloader_output.as_kwargs(),
            )
        elif isinstance(config, ZeroShotRankingModelBaseConfig):
            ZSmodel: nn.Module = PostgresZeroShotModel(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )
            model: nn.Module = RankModel(
                model=ZSmodel,
                model_config=config,
                **dataloader_output.as_kwargs(),
            )
        elif isinstance(config, ZeroShotVAEModelConfig):
            model: nn.Module = PostgresZeroShotModelVAE(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )
        elif isinstance(config, ZeroShotModelConfig):
            model: nn.Module = PostgresZeroShotModel(
                model_config=config,
                feature_statistics=feature_statistics,
                label_norm=label_norm,
            )
        elif isinstance(config, MambaModelConfig):
            model: nn.Module = MambaModel(
                model_config=config, feature_statistics=feature_statistics
            )
        else:
            raise NotImplementedError(f"Model type {config.name.NAME} not implemented")

    elif isinstance(
        config,
        (E2EModelConfig, MSCNModelConfig, TlstmModelConfig, QueryFormerModelConfig),
    ):
        # assert not assemble_data_up_front, "assemble_data_up_front is not supported for this model type"
        assemble_data_up_front = (
            False  # assemble_data_up_front is not supported for this model type
        )

        _, feature_statistics, train_loader, val_loader, test_loaders, input_dims = (
            create_baseline_dataloader(
                workload_runs=workload_runs,
                statistics_file=statistics_file,
                column_statistics=column_statistics,
                word_embeddings=word_embeddings,
                model_config=config,
                data_loader_options=DataLoaderOptions(),
            )
        )

        if isinstance(config, E2EModelConfig):
            model: nn.Module = E2EModel(model_config=config, input_dims=input_dims)

        elif isinstance(config, MSCNModelConfig):
            model: nn.Module = MSCNModel(model_config=config, input_dims=input_dims)

        elif isinstance(config, QueryFormerModelConfig):
            model: nn.Module = QueryFormer(
                config=config,
                input_dims=input_dims,
                label_norm=None,
                feature_statistics=feature_statistics,
            )

    elif isinstance(config, NeoModelConfig):
        _, feature_statistics, train_loader, val_loader, test_loaders, input_dims = (
            create_baseline_dataloader(
                workload_runs=workload_runs,
                statistics_file=statistics_file,
                column_statistics=column_statistics,
                word_embeddings=word_embeddings,
                model_config=config,
                data_loader_options=DataLoaderOptions(),
            )
        )

        model: nn.Module = TreeConvolution(
            model_config=config,
            input_dims=input_dims,
            feature_statistics=feature_statistics,
        )

    elif isinstance(config, QPPNetModelConfig):
        # assert not assemble_data_up_front, "assemble_data_up_front is not supported for this model type"
        assemble_data_up_front = (
            False  # assemble_data_up_front is not supported for this model type
        )
        label_norm, feature_statistics, train_loader, val_loader, test_loaders = (
            create_qppnet_dataloader(
                workload_runs=workload_runs,
                statistics_file=statistics_file,
                column_statistics=column_statistics,
                model_config=config,
                data_loader_options=DataLoaderOptions(),
            )
        )

        model: nn.Module = QPPNet(
            model_config=config,
            workload_runs=workload_runs,
            feature_statistics=feature_statistics,
            label_norm=label_norm,
        )

    elif isinstance(config, DACEModelConfig):
        # assert not assemble_data_up_front, "assemble_data_upfront is not supported for this model type"
        assemble_data_up_front = (
            False  # assemble_data_up_front is not supported for this model type
        )
        feature_statistics, train_loader, val_loader, test_loaders = (
            create_dace_dataloader(
                workload_runs=workload_runs,
                statistics_file=statistics_file,
                model_config=config,
                dataloader_options=DataLoaderOptions(),
            )
        )
        model: nn.Module = DACELora(config=config)

    elif isinstance(
        config,
        (ScaledPostgresModelConfig, FlatModelConfig, T3ModelConfig, SPNModelConfig),
    ):
        # assert not assemble_data_up_front, "assemble_data_upfront is not supported for this model type"
        assemble_data_up_front = (
            False  # assemble_data_up_front is not supported for this model type
        )
        model = None
        train_loader, val_loader, test_loaders = None, None, None
        feature_statistics = None
        model = None

    else:
        raise NotImplementedError

    def gather_dataloader_stats(dataloader, string: str):
        # Count the number of entries in train_loader where timeout happened
        overall_count = 0
        timeout_count = 0
        if assemble_data_up_front:
            for batch in dataloader.dataset:
                # extract statistics
                per_batch_labels = batch[2]
                overall_count += len(per_batch_labels)
                timeout_count += sum([1 for label in per_batch_labels if label is None])
        else:
            for label, plan in dataloader.dataset:  # dataloader is batched
                overall_count += 1
                timeout_count += 1 if plan["timeout"] else 0

        print(
            f"Timeout count: {timeout_count} of {overall_count} "
            f"({timeout_count / overall_count * 100:.2f}%) in {string}"
        )

    if (
        hasattr(config, "online_hard_example_mining")
        and not config.online_hard_example_mining
    ) or not hasattr(config, "online_hard_example_mining"):
        if train_loader:
            gather_dataloader_stats(train_loader, "train_loader")
            gather_dataloader_stats(val_loader, "val_loader")
        if test_loaders:
            for test_loader in test_loaders:
                gather_dataloader_stats(test_loader, "test_loader")
    print("\n")

    return model, train_loader, val_loader, test_loaders, feature_statistics


def use_model(
    mode: str,
    wl_runs: WorkloadRuns,
    statistics_file: Path,
    column_statistics: Path,
    word_embeddings: Path,
    model_dir: Path,
    target_dir: Path,
    config: ModelConfig,
    wandb_run: Run,
    pt_profiler: bool,
    bound_predictions: bool,
    is_pretrained: bool,
    enable_checkpoint_loading: bool = False,
    run_qo_test_pipeline_after_each_epoch: bool = True,
    pretrained_model_dir: Optional[str] = None,
    pretrained_model_name: Optional[str] = None,
    quick_run: bool = False,
    assemble_data_up_front: bool = False,
    training_info: Optional[Dict] = None,
):
    # Seed for reproducibility
    print("Found model config:")
    pprint(attrs.asdict(config), sort_dicts=True)
    # basic seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # hash seed for Python hashing (affects some randomized containers)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # ADD dgl seed!
    try:
        import dgl

        dgl.seed(config.seed)
        dgl.random.seed(config.seed)
    except ImportError:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # newer API
        torch.use_deterministic_algorithms(True)
    except Exception:
        # best-effort: if the runtime doesn't support this, continue
        pass

    # Add test workloads to workload_runs
    wl_runs.update_test_workloads(target_dir, config.seed, config.version)

    # Check if training and testing was previously executed
    if wl_runs.check_if_done(config.name.NAME):
        return

    # Do sanity check on the training data paths
    wl_runs.check_model_compability(config, mode, is_pretrained=is_pretrained)

    # Initialize metrics
    metrics: List[Metric] = init_metrics(config.loss_class_name)

    # Initialize models and data loaders
    print(
        f"Initializing model {config.name.NAME} with seed {config.seed} and version {config.version} and load datasets",
        flush=True,
    )
    model, train_loader, val_loader, test_loaders, feature_stats = init_model(
        config=config,
        statistics_file=statistics_file,
        column_statistics=column_statistics,
        workload_runs=wl_runs,
        word_embeddings=word_embeddings,
        quick_run=quick_run,
        assemble_data_up_front=assemble_data_up_front,
    )

    if test_loaders:
        # extract test data for testing after each epoch
        if run_qo_test_pipeline_after_each_epoch:
            test_after_epoch_dataloaders = test_loaders
            test_after_epoch_testfiles = wl_runs.target_test_csv_paths
        else:
            test_after_epoch_dataloaders = []
            test_after_epoch_testfiles = []
            for dl, wl_file in zip(test_loaders, wl_runs.target_test_csv_paths):
                str_wl_file = str(wl_file)
                if "join_order" in str_wl_file:
                    continue
                if "physical_operator" in str_wl_file:
                    continue
                if "access_path_selection" in str_wl_file:
                    continue
                test_after_epoch_dataloaders.append(dl)
                test_after_epoch_testfiles.append(wl_file)
    else:
        test_after_epoch_dataloaders = []
        test_after_epoch_testfiles = []

    found = False
    rank_found = False
    finished = False

    if model is not None:
        # Move model to device
        print(f"Model device: {model.device}", flush=True)
        model = model.to(model.device)
        print(f"Model is now on device: {next(model.parameters()).device}", flush=True)
        print(f"Optimizer kwargs: {config.optimizer_kwargs}", flush=True)

        # Init optimizer
        if config.optimizer_class_name == "AdamWScheduleFree":
            optimizer: Optimizer = AdamWScheduleFree(
                model.parameters(), **config.optimizer_kwargs
            )
        else:
            optimizer: Optimizer = opt.__dict__[config.optimizer_class_name](
                model.parameters(), **config.optimizer_kwargs
            )

        scheduler, target_epochs = init_scheduler(config, optimizer)

        has_pretrained_model_dir = (
            pretrained_model_dir is not None and pretrained_model_name is not None
        )

        print(
            f"enable_checkpoint_loading={enable_checkpoint_loading}, has_pretrained_model_dir={has_pretrained_model_dir}"
        )
        if has_pretrained_model_dir:
            print(
                f"Pretrained model specified: {pretrained_model_name} in {pretrained_model_dir}"
            )

        if enable_checkpoint_loading and not has_pretrained_model_dir:
            # for continuing training from a checkpoint of the same experiment, we look for checkpoints in model_dir
            print(
                f"Try to load model with name {config.name.NAME} and seed {config.seed} from {model_dir}",
                flush=True,
            )
            (
                csv_stats,
                epochs_wo_improvement,
                epoch,
                model,
                optimizer,
                metrics,
                finished,
                found,
                rank_found,
            ) = load_checkpoint(
                model=model,
                target_path=(
                    model_dir if not quick_run else None
                ),  # do not load checkpoint in quick_run
                config=config,
                optimizer=optimizer,
                metrics=metrics,
                filetype=".pt",
                load_last_checkpoint=True,
            )
        elif has_pretrained_model_dir:
            # for loading a pretrained model (potentially trained in a different experiment), we look for checkpoints in the specified pretrained_model_dir
            # no checkpoint found, try to load the model from the pretrained_model_dir
            print(
                f"Try to initialize model from {pretrained_model_dir}/{pretrained_model_name}",
                flush=True,
            )
            # We capture both 'found' and 'rank_found' because load_checkpoint returns found=False for rank models.
            _, _, _, model, _, _, _, found_base, found_rank = load_checkpoint(
                model=model,
                manual_model_dir=pretrained_model_dir,
                manual_model_name=pretrained_model_name,
                config=None,
                optimizer=None,
                metrics=copy.deepcopy(metrics),
                filetype=".pt",
                load_last_checkpoint=False,
                target_path=None,
            )  # load best checkpoint

            # Consider the model "found" if either a base or a rank model was loaded.
            found = found_base or found_rank

            assert found, (
                f"Pretrained model not found in {pretrained_model_dir}/{pretrained_model_name}"
            )
        else:
            print(
                "Skipping loading of checkpoint as per enable_checkpoint_loading flag",
                flush=True,
            )
            csv_stats = []
            epochs_wo_improvement = 0
            epoch = 0
            finished = False
            found = False
            rank_found = False

        if rank_found:
            print("Found pretrained model (trained with ranking)", flush=True)
        if found:
            print("Found pretrained model", flush=True)

    if is_pretrained and not isinstance(config, TabularModelConfig):
        assert found
        if not finished:
            print(
                "Pretrained model not finished. Though still considering it as finished because of `is_pretrained` flag is set.",
                flush=True,
            )
        finished = True

    ###
    # Do deferred setup of OHEM - it was easier to do it here, because we need the model and criterion (not yet available during dataset creation)
    ###
    if (
        hasattr(config, "online_hard_example_mining")
        and config.online_hard_example_mining
    ):
        assert isinstance(config, (ZeroShotModelConfig, MambaModelConfig)), (
            "OHEM is only supported for ZeroShot and Mamba at the moment"
        )
        loss_fn = functools.partial(model.loss_fxn, return_aggregated=False)

        # update datasets with model / criterion information
        train_loader.dataset.deferred_setup(
            model=model,
            criterion=loss_fn,
            initialize_difficulty=config.ohem_initialize_difficulty,
        )
        val_loader.dataset.deferred_setup(
            model=model,
            criterion=loss_fn,
            initialize_difficulty=config.ohem_initialize_difficulty,
        )

        for test_loader in test_loaders:
            if isinstance(test_loader, OHEMPlanDataset):
                test_loader.dataset.deferred_setup(
                    model=model,
                    criterion=loss_fn,
                    initialize_difficulty=config.ohem_initialize_difficulty,
                )

    # prepare the prediction bounder
    if bound_predictions:
        prediction_bounder = PredictionBounder(history_dataset=train_loader.dataset)
    else:
        prediction_bounder = None

    # Train model
    if mode == "retrain":
        epochs_wo_improvement = 0

    rerun_predict = False
    epoch = 0
    if mode in ["train", "retrain"]:
        try:
            wandb_run.watch(model, log="all")
        except Exception:
            print("Could not watch model with wandb")

        if isinstance(model_config, TabularModelConfig):
            # check if model is already pre-trained -> do not execute this
            if not is_pretrained:
                model = train_tabular_model(
                    workload_runs=wl_runs,
                    config=config,
                    statistics_file=statistics_file,
                    model_dir=model_dir,
                    bound_predictions=False,
                    prediction_bounder=prediction_bounder,
                    dataloader_options=DataLoaderOptions(),
                )
        else:
            train_model(
                model_dir=model_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loaders=test_after_epoch_dataloaders,
                test_csv_paths=test_after_epoch_testfiles,
                config=config,
                csv_stats=csv_stats,
                epochs_wo_improvement=epochs_wo_improvement,
                epoch=epoch,
                epoch_offset=0 if not found else epoch,
                target_epochs=target_epochs,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                wandb_run=wandb_run,
                pt_profiler=pt_profiler,
                bound_predictions=False,
                prediction_bounder=prediction_bounder,
                quick_run=quick_run,
                scheduler=scheduler,
                training_info=training_info,
            )

        # perform validation - after each training stage
        if isinstance(model_config, TabularModelConfig):
            predict_tabular_model(
                config=config,
                workload_runs=wl_runs,
                statistics_file=statistics_file,
                model_dir=model_dir,
                wandb_run=wandb_run,
                bound_predictions=bound_predictions,
                prediction_bounder=prediction_bounder,
                t3_model=model,
            )
        else:
            test_model(
                config=config,
                test_csv_paths=wl_runs.target_test_csv_paths,
                test_loaders=test_loaders,
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                metrics=metrics,
                wandb_run=wandb_run,
                log_prefix="test",
                file_suffix="",
                bound_predictions=bound_predictions,
                prediction_bounder=prediction_bounder,
                quick_run=quick_run,
            )

        # reset stats
        epoch = 0
        epochs_wo_improvement = 0

    print(
        f"--Run {config.name.NAME} ended at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} --"
    )

    if mode in ["predict"] or rerun_predict:
        print(f"Predicting model {config.name.NAME} with seed {config.seed}")
        if isinstance(model_config, TabularModelConfig):
            predict_tabular_model(
                config=config,
                workload_runs=wl_runs,
                statistics_file=statistics_file,
                model_dir=model_dir,
                wandb_run=wandb_run,
                bound_predictions=bound_predictions,
                prediction_bounder=prediction_bounder,
                t3_model=model,
            )
        else:
            test_model(
                config=config,
                test_csv_paths=wl_runs.target_test_csv_paths,
                test_loaders=test_loaders,
                model_dir=model_dir,
                epoch=epoch,
                model=model,
                metrics=metrics,
                wandb_run=wandb_run,
                file_suffix="",
                bound_predictions=bound_predictions,
                prediction_bounder=prediction_bounder,
                log_prefix="test",
                quick_run=quick_run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "predict", "explain", "retrain"], required=True
    )
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--workload_runs", default=None, nargs="+")
    parser.add_argument("--test_workload_runs", default=None, nargs="+")
    parser.add_argument("--statistics_file", default=None, required=True)
    parser.add_argument("--column_statistics", default=None)
    parser.add_argument("--model_dir", default=None, required=True)
    parser.add_argument("--target_dir", default=None, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hyperparameter_path", default=None)
    parser.add_argument("--word_embeddings", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--cap_training_samples", type=int, default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_resume_id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--quick_run", action="store_true")
    parser.add_argument("--pt_profiler", action="store_true")
    parser.add_argument("--bound_predictions", action="store_true")
    parser.add_argument("--qo_eval_after_each_epoch", action="store_true")
    parser.add_argument("--is_pretrained", action="store_true")
    # pre-trained model, used to initialize weights
    parser.add_argument("--pretrained_model_dir", default=None)
    parser.add_argument("--pretrained_model_name", default=None)

    parser.add_argument("--enable_checkpoint_loading", action="store_true")

    # assemble data up front
    parser.add_argument("--assemble_data_up_front", action="store_true")

    args = parser.parse_args()

    assert args.device == "cpu" or args.device.startswith("cuda"), (
        "Device must be either 'cpu' or 'cuda'"
    )
    assert (
        "[socket]" not in args.device
    )  # make the sure the socket placeholder was correctly replaced during cmd generation

    print(f"Device used: {args.device}")

    # Init model configuration by either reading out hyperparameters or using standard config
    model_args = dict(
        device=args.device,
        cap_training_samples=args.cap_training_samples,
        seed=args.seed,
        version=args.version,
        num_workers=args.num_workers,
    )

    if args.quick_run:
        model_args.update(limit_plans_per_file=2000, epochs=3)

    # Create workload runs
    if args.workload_runs:
        train_runs = [Path(raw_path) for raw_path in args.workload_runs]
    else:
        train_runs = []
    if args.test_workload_runs:
        test_runs = [Path(raw_path) for raw_path in args.test_workload_runs]
    else:
        test_runs = []

    if args.quick_run:
        train_runs = train_runs[:1]

    runs = WorkloadRuns(train_workload_runs=train_runs, test_workload_runs=test_runs)

    model_config = get_model_config(model_type=args.model_type, m_args=model_args)

    # record how this model is trained, beside and inside its checkpoint (provenance.py)
    training_info = None
    if args.mode == "train" and not args.quick_run:
        training_info = collect_training_info(
            cli_args=vars(args),
            config=model_config,
            train_workloads=train_runs,
            test_workloads=test_runs,
            other_files={
                "statistics_file": args.statistics_file,
                "column_statistics": args.column_statistics,
                "word_embeddings": args.word_embeddings,
            },
        )
        info_path = write_training_info(
            training_info,
            args.model_dir,
            shorten_model_name(f"{model_config.name.NAME}_{model_config.seed}"),
        )
        print(f"Training info written to {info_path}")

    # extract model_type information
    model_type = args.model_type
    if model_type.startswith("pm-"):
        model_type = model_type[3:].split("-")[0]
    else:
        model_type = model_type.split("-")[0]
    if model_type.startswith("postgres"):
        model_type = "postgres"
    elif model_type.startswith("zeroshot"):
        model_type = "zeroshot"
    elif model_type.startswith("zs"):
        model_type = "zeroshot"
    elif model_type.startswith("dace"):
        model_type = "dace"
    elif model_type.startswith("flat"):
        model_type = "flat"
    elif model_type.startswith("t3"):
        model_type = "t3"
    elif model_type.startswith("mscn"):
        model_type = "mscn"
    elif model_type.startswith("query_former"):
        model_type = "query_former"
    elif model_type.startswith("qppnet"):
        model_type = "qppnet"
    elif model_type.startswith("neo"):
        model_type = "neo"
    elif model_type.startswith("e2e"):
        model_type = "e2e"
    elif model_type.startswith("mamba"):
        model_type = "mamba"
    elif model_type.startswith("spn"):
        model_type = "spn"
    else:
        raise ValueError(f"Model type {model_type} not recognized")

    # initialize wandb
    wandb_run: Optional[Run] = None
    if args.wandb_project:
        wandb_name = args.wandb_name
        if wandb_name.startswith("pm-"):  # remove the pm- prefix
            wandb_name = wandb_name[3:]

        if args.quick_run:
            wandb_name = f"quick_{wandb_name}"

        config_dict = attrs.asdict(model_config)
        # add wandb-visible name and model_type
        config_dict["name"] = wandb_name
        config_dict["model_type"] = model_type
        # expose the base config name (for parameterized models 'pm-<base>-...')
        full_model_name = model_config.name.NAME
        if full_model_name.startswith("pm-"):
            base_config_name = full_model_name[3:].split("-")[0]
        else:
            base_config_name = full_model_name.split("-")[0]
        config_dict["base_config_name"] = base_config_name
        config_dict["seed"] = args.seed
        config_dict["version"] = args.version
        config_dict["quick_run"] = args.quick_run
        config_dict["device"] = args.device

        if args.wandb_resume_id is not None:
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                id=args.wandb_resume_id,
                resume="must",
            )
        else:
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=wandb_name,
                config=config_dict,
                group=args.wandb_group,
                tags=[model_type],
            )

    args = dict(
        mode=args.mode,
        wl_runs=runs,
        statistics_file=Path(args.statistics_file),
        model_dir=Path(args.model_dir),
        target_dir=Path(args.target_dir),
        config=model_config,
        word_embeddings=Path(args.word_embeddings) if args.word_embeddings else None,
        column_statistics=Path(args.column_statistics)
        if args.column_statistics
        else None,
        pt_profiler=args.pt_profiler,
        bound_predictions=args.bound_predictions,
        enable_checkpoint_loading=args.enable_checkpoint_loading,
        wandb_run=wandb_run,
        run_qo_test_pipeline_after_each_epoch=args.qo_eval_after_each_epoch,
        is_pretrained=args.is_pretrained,
        pretrained_model_dir=args.pretrained_model_dir,
        pretrained_model_name=args.pretrained_model_name,
        quick_run=args.quick_run,
        assemble_data_up_front=args.assemble_data_up_front,
        training_info=training_info,
    )

    use_model(**args)
    if wandb_run:
        wandb_run.finish()
