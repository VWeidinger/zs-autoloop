import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import attrs as attrs
import numpy as np
import torch
import torch.optim as opt
import wandb
from attrs import evolve
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from wandb.apis.public import Run

from classes.classes import ModelConfig, E2EModelConfig, ZeroShotModelConfig, MSCNModelConfig, DataLoaderOptions, \
    TlstmModelConfig, ModelName, FlatModelConfig, ScaledPostgresModelConfig, TabularModelConfig, TPoolModelConfig, \
    QPPNetModelConfig, DACEModelConfig, DACEModelNoCostsConfig, QPPModelNoCostsConfig, FlatModelActCardModelConfig, \
    QPPModelActCardsConfig, ZeroShotModelActCardConfig, DACEModelActCardConfig, QueryFormerModelConfig, \
    ZSLearnFromTimeoutConfig, ZSRankLossConfig, FlatStratifiedModelConfig, ZSStratifiedModelConfig, \
    DACEStratifiedModelConfig, ScaledPostgresStratifiedModelConfig
from classes.workload_runs import WorkloadRuns
from models.dace.dace_dataset import create_dace_dataloader
from models.dace.dace_model import DACELora
from models.qppnet.qppnet_dataloader import create_qppnet_dataloader
from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.tabular.train_tabular_baseline import train_tabular_model, predict_tabular_model
from models.workload_driven.dataset.dataset_creation import create_baseline_dataloader
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from train import train_model
from training.bound_predictions_with_history import PredictionBounder
from training.dataset.dataset_creation import create_zeroshot_dataloader
from training.training.checkpoint import load_checkpoint
from training.training.metrics import Metric, RankAccuracy, init_metrics
from training.training.test import test_model


def init_model(statistics_file: Path,
               column_statistics: Path,
               config: ModelConfig,
               workload_runs: WorkloadRuns,
               word_embeddings: Path) \
        -> (Optional[nn.Module], DataLoader, DataLoader, List[DataLoader]):
    if isinstance(config, ZeroShotModelConfig):
        label_norm, feature_statistics, train_loader, val_loader, test_loaders = \
            create_zeroshot_dataloader(workload_runs=workload_runs,
                                       statistics_file=statistics_file,
                                       model_config=config,
                                       data_loader_options=DataLoaderOptions())

        model: nn.Module = PostgresZeroShotModel(model_config=config,
                                                 feature_statistics=feature_statistics,
                                                 label_norm=label_norm)

    elif isinstance(config, (E2EModelConfig, MSCNModelConfig, TlstmModelConfig, QueryFormerModelConfig)):
        _, feature_statistics, train_loader, val_loader, test_loaders, input_dims = \
            create_baseline_dataloader(workload_runs=workload_runs,
                                       statistics_file=statistics_file,
                                       column_statistics=column_statistics,
                                       word_embeddings=word_embeddings,
                                       model_config=config,
                                       data_loader_options=DataLoaderOptions())

        if isinstance(config, E2EModelConfig):
            model: nn.Module = E2EModel(model_config=config,
                                        input_dims=input_dims)

        elif isinstance(config, MSCNModelConfig):
            model: nn.Module = MSCNModel(model_config=config,
                                         input_dims=input_dims)

        elif isinstance(config, QueryFormerModelConfig):
            model: nn.Module = QueryFormer(config=config,
                                           input_dims=input_dims,
                                           label_norm=None,
                                           feature_statistics=feature_statistics)

    elif isinstance(config, QPPNetModelConfig):
        label_norm, feature_statistics, train_loader, val_loader, test_loaders = \
            create_qppnet_dataloader(workload_runs=workload_runs,
                                     statistics_file=statistics_file,
                                     column_statistics=column_statistics,
                                     model_config=config,
                                     data_loader_options=DataLoaderOptions())

        model: nn.Module = QPPNet(model_config=config,
                                  workload_runs=workload_runs,
                                  feature_statistics=feature_statistics,
                                  label_norm=label_norm)

    elif isinstance(config, DACEModelConfig):
        feature_statistics, train_loader, val_loader, test_loaders = \
            create_dace_dataloader(workload_runs=workload_runs,
                                   statistics_file=statistics_file,
                                   model_config=config,
                                   dataloader_options=DataLoaderOptions())
        model: nn.Module = DACELora(config=config)

    elif isinstance(config, (ScaledPostgresModelConfig, FlatModelConfig)):
        model = None
        train_loader, val_loader, test_loaders = None, None, None
        feature_statistics = None
    else:
        raise NotImplementedError

    def gather_dataloader_stats(dataloader, string: str):
        # Count the number of entries in train_loader where timeout happened
        overall_count = 0
        timeout_count = 0
        for label, plan in dataloader.dataset:  # dataloader is batched
            overall_count += 1
            timeout_count += 1 if plan['timeout'] else 0

        print(f"Timeout count: {timeout_count} of {overall_count} "
              f"({timeout_count / overall_count * 100:.2f}%) in {string}")

    if train_loader:
        gather_dataloader_stats(train_loader, "train_loader")
        gather_dataloader_stats(val_loader, "val_loader")
    if test_loaders:
        for test_loader in test_loaders:
            gather_dataloader_stats(test_loader, "test_loader")
    print('\n')

    return model, train_loader, val_loader, test_loaders, feature_statistics


def use_model(mode: str,
              wl_runs: WorkloadRuns,
              statistics_file: Path,
              column_statistics: Path,
              word_embeddings: Path,
              model_dir: Path,
              ft_with_ranking: bool,
              target_dir: Path,
              config: ModelConfig,
              wandb_run: Run,
              pt_profiler: bool,
              bound_predictions: bool,
              run_qo_test_pipeline_after_each_epoch: bool = True):
    # Seed for reproducibility
    print(f"Found model config {config}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Add test workloads to workload_runs
    wl_runs.update_test_workloads(target_dir, config.seed)

    # Check if training and testing was previously executed
    if wl_runs.check_if_done(config.name.NAME):
        return

    # Do sanity check on the training data paths
    wl_runs.check_model_compability(config, mode)

    # Initialize metrics
    metrics: List[Metric] = init_metrics(config.loss_class_name)

    # Initialize models and data loaders
    print(f'Initializing model {config.name.NAME} with seed {config.seed} and load datasets', flush=True)
    model, train_loader, val_loader, test_loaders, feature_stats = init_model(config=config,
                                                                              statistics_file=statistics_file,
                                                                              column_statistics=column_statistics,
                                                                              workload_runs=wl_runs,
                                                                              word_embeddings=word_embeddings)
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
                if 'join_order' in str_wl_file:
                    continue
                if 'physical_operator' in str_wl_file:
                    continue
                if 'access_path_selection' in str_wl_file:
                    continue
                test_after_epoch_dataloaders.append(dl)
                test_after_epoch_testfiles.append(wl_file)

    found = False
    rank_found = False
    finished = False
    if model:
        # Move model to device
        print(f"Model device: {model.device}")
        model = model.to(model.device)

        # Init optimizer
        optimizer: Optimizer = opt.__dict__[config.optimizer_class_name](model.parameters(),
                                                                         **config.optimizer_kwargs)

        print(f"Try to load model with name {config.name.NAME} and seed {config.seed} from {model_dir}")
        csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished, found, rank_found = \
            load_checkpoint(model=model,
                            target_path=model_dir,
                            config=config,
                            optimizer=optimizer,
                            metrics=metrics,
                            filetype='.pt',
                            prefer_ft_rank=ft_with_ranking)

        if rank_found:
            print("Found pretrained model (trained with ranking)")
        if found:
            print("Found pretrained model")

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
        # assemble train to do list - FALSE=no ranking, TRUE=ranking
        if ft_with_ranking:
            if rank_found:
                todo_list = [True]
            else:
                todo_list = [False, True]
        else:
            todo_list = [False]

        if finished:
            print(f"Training already finished for {config.name.NAME} with seed {config.seed}")
            todo_list.pop(0)  # remove the first element - it has already finished

            # if there is still something to do: start from the beginning the next stage
            if len(todo_list) > 0:
                epoch = 0
                epochs_wo_improvement = 0
            else:
                rerun_predict = True

        for train_w_rank_model in todo_list:
            print(f'Train Model using ranking: {train_w_rank_model}')

            if train_w_rank_model:
                rank_metric = RankAccuracy(early_stopping_metric=True)

                # set early stopping metric to False for other metrics
                for metric in metrics:
                    metric.early_stopping_metric = False

                metrics.append(rank_metric)
            else:
                rank_metric = None

            if isinstance(model_config, TabularModelConfig):
                train_tabular_model(workload_runs=wl_runs,
                                    config=config,
                                    statistics_file=statistics_file,
                                    model_dir=model_dir,
                                    bound_predictions=False,
                                    prediction_bounder=prediction_bounder,
                                    dataloader_options=DataLoaderOptions())
            else:
                train_model(model_dir=model_dir,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loaders=test_after_epoch_dataloaders,
                            test_csv_paths=test_after_epoch_testfiles,
                            config=config,
                            csv_stats=csv_stats,
                            epochs_wo_improvement=epochs_wo_improvement,
                            epoch=epoch,
                            epoch_offset=0 if not train_w_rank_model else config.epochs,
                            model=model,
                            optimizer=optimizer,
                            metrics=metrics,
                            wandb_run=wandb_run,
                            pt_profiler=pt_profiler,
                            train_w_rank_model=train_w_rank_model,
                            rank_metric=rank_metric,
                            bound_predictions=False,
                            prediction_bounder=prediction_bounder)

            # perform validation - after each training stage
            if isinstance(model_config, TabularModelConfig):
                predict_tabular_model(config=config,
                                      workload_runs=wl_runs,
                                      statistics_file=statistics_file,
                                      model_dir=model_dir,
                                      wandb_run=wandb_run,
                                      bound_predictions=bound_predictions,
                                      prediction_bounder=prediction_bounder, )
            else:
                test_model(config=config,
                           test_csv_paths=wl_runs.target_test_csv_paths,
                           test_loaders=test_loaders,
                           model_dir=model_dir,
                           epoch=epoch,
                           model=model,
                           metrics=metrics,
                           wandb_run=wandb_run,
                           log_prefix='test',
                           file_suffix=f"{'_rankmodel' if train_w_rank_model else ''}",
                           bound_predictions=bound_predictions,
                           prediction_bounder=prediction_bounder)

            # reset stats
            epoch = 0
            epochs_wo_improvement = 0

        print(f'--Run {config.name.NAME} ended at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} --')

    if mode in ["predict"] or rerun_predict:
        print(f"Predicting model {config.name.NAME} with seed {config.seed}")
        if isinstance(model_config, TabularModelConfig):
            predict_tabular_model(config=config,
                                  workload_runs=wl_runs,
                                  statistics_file=statistics_file,
                                  model_dir=model_dir,
                                  target_path=target_dir,
                                  wandb_run=wandb_run,
                                  bound_predictions=bound_predictions,
                                  prediction_bounder=prediction_bounder,
                                  dataloader_options=DataLoaderOptions()
                                  )
        else:
            test_model(config=config,
                       test_csv_paths=wl_runs.target_test_csv_paths,
                       test_loaders=test_loaders,
                       model_dir=model_dir,
                       epoch=epoch,
                       model=model,
                       metrics=metrics,
                       wandb_run=wandb_run,
                       file_suffix='',
                       bound_predictions=bound_predictions,
                       prediction_bounder=prediction_bounder,
                       log_prefix='test'
                       )


def get_model_config(model_type: str, m_args: dict) -> ModelConfig:
    model_config_map = {
        ModelName.POSTGRES.NAME: ScaledPostgresModelConfig,
        ModelName.MSCN.NAME: MSCNModelConfig,
        ModelName.ZEROSHOT.NAME: ZeroShotModelConfig,
        ModelName.ZS_RANK_LOSS.NAME: ZSRankLossConfig,
        ModelName.E2E.NAME: TPoolModelConfig,
        ModelName.QPP_NET.NAME: QPPNetModelConfig,
        ModelName.DACE.NAME: DACEModelConfig,
        ModelName.FLAT.NAME: FlatModelConfig,
        ModelName.FLAT_ACT_CARDS.NAME: FlatModelActCardModelConfig,
        ModelName.QUERY_FORMER.NAME: QueryFormerModelConfig,

        # Stratified Models
        ModelName.FLAT_STRATIFIED.NAME: FlatStratifiedModelConfig,
        ModelName.ZS_STRATIFIED.NAME: ZSStratifiedModelConfig,
        ModelName.DACE_STRATIFIED.NAME: DACEStratifiedModelConfig,
        ModelName.POSTGRES_STRATIFIED.NAME: ScaledPostgresStratifiedModelConfig,

        # learn from timeout
        ModelName.ZS_FROM_TIMEOUT.NAME: ZSLearnFromTimeoutConfig,

        # Ablation models without PG costs
        ModelName.DACE_NO_COSTS.NAME: DACEModelNoCostsConfig,
        ModelName.QPP_NET_NO_COSTS.NAME: QPPModelNoCostsConfig,

        # Act card models
        ModelName.QPP_NET_ACT_CARDS.NAME: QPPModelActCardsConfig,
        ModelName.ZEROSHOT_ACT_CARD.NAME: ZeroShotModelActCardConfig,
        ModelName.DACE_ACT_CARDS.NAME: DACEModelActCardConfig,
    }

    if model_type in model_config_map:
        config = model_config_map[model_type](**m_args)

        return config

    else:
        raise Exception(f"Model name {model_type} not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "predict", "explain", "retrain"], required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--workload_runs', default=None, nargs='+', required=True)
    parser.add_argument('--test_workload_runs', default=None, nargs='+')
    parser.add_argument('--statistics_file', default=None, required=True)
    parser.add_argument('--column_statistics', default=None)
    parser.add_argument('--model_dir', default=None, required=True)
    parser.add_argument('--ft_with_ranking', action='store_true')
    parser.add_argument('--target_dir', default=None, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hyperparameter_path', default=None)
    parser.add_argument('--word_embeddings', default=None)
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--cap_training_samples', type=int, default=None)
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--pt_profiler', action='store_true')
    parser.add_argument('--bound_predictions', action='store_true')
    parser.add_argument('--qo_eval_after_each_epoch', action='store_true')
    args = parser.parse_args()

    # Init model configuration by either reading out hyperparameters or using standard config
    model_args = dict(device=args.device,
                      cap_training_samples=args.cap_training_samples,
                      seed=args.seed,
                      num_workers=args.num_workers)

    if args.quick_run:
        model_args.update(limit_plans_per_file=2000, epochs=3, batch_size=64)

    # Create workload runs
    train_runs = [Path(raw_path) for raw_path in args.workload_runs]
    if args.test_workload_runs:
        test_runs = [Path(raw_path) for raw_path in args.test_workload_runs]
    else:
        test_runs = []

    if args.quick_run:
        train_runs = train_runs[:1]

    runs = WorkloadRuns(train_workload_runs=train_runs, test_workload_runs=test_runs)

    model_config = get_model_config(model_type=args.model_type, m_args=model_args)

    # If retraining, set specific model parameters
    if args.mode == "retrain":
        model_config = evolve(model_config, epochs=model_config.epochs + 100)
        # Disable penalty for small values in case of retraining
        if model_config.loss_class_name == "QLoss":
            final_mlp_kwargs = model_config.final_mlp_kwargs
            loss_class_kwargs = final_mlp_kwargs.pop('loss_class_kwargs')
            loss_class_kwargs.update(penalty_negative=0)
            final_mlp_kwargs.update(loss_class_kwargs=loss_class_kwargs)
            model_config = evolve(model_config, final_mlp_kwargs=final_mlp_kwargs)

    # initialize wandb
    wandb_run: Optional[Run] = None
    if args.wandb_project:
        wandb_run = wandb.init(entity=args.wandb_entity,
                         project=args.wandb_project,
                         name=args.wandb_name,
                         config=attrs.asdict(model_config))

    args = dict(
        mode=args.mode,
        wl_runs=runs,
        statistics_file=Path(args.statistics_file),
        model_dir=Path(args.model_dir),
        ft_with_ranking=args.ft_with_ranking,
        target_dir=Path(args.target_dir),
        config=model_config,
        word_embeddings=Path(args.word_embeddings) if args.word_embeddings else None,
        column_statistics=Path(args.column_statistics) if args.column_statistics else None,
        pt_profiler=args.pt_profiler,
        bound_predictions=args.bound_predictions,
        wandb_run=wandb_run,
        run_qo_test_pipeline_after_each_epoch=args.qo_eval_after_each_epoch)

    use_model(**args)

    if wandb_run:
        wandb_run.finish()
