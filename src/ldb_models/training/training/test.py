import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from classes import MambaRankingModelConfig, ModelConfig, ZeroShotRankingModelBaseConfig
from dgl.dataloading import DataLoader
from torch import nn
from training.bound_predictions_with_history import PredictionBounder
from training.evaluate_query_optimization import (
    evaluate_access_path_selection,
    evaluate_join_order,
    evaluate_physical_plan_selection,
)
from training.training.checkpoint import save_csv
from training.training.metrics import Metric
from training.training.utils import find_early_stopping_metric, prefix_dict
from training.training.validate import validate_model
from wandb.apis.public import Run


def test_model(
    config: ModelConfig,
    test_loaders: List[DataLoader],
    test_csv_paths: List[Path],
    model_dir: Path,
    metrics: List[Metric],
    model: nn.Module,
    epoch: int,
    wandb_run: Run,
    file_suffix: str,
    bound_predictions: bool = False,
    prediction_bounder: Optional[PredictionBounder] = None,
    load_best_checkpoint: bool = True,
    log_prefix: str = "test",
    is_per_epoch: bool = False,
    quick_run: bool = False,
):
    # initialize variables for aggregated reporting across all experiments
    job_total_selected_runtime = -1
    job_avg_slowdown = None
    job_light_total_selected_runtime = -1
    job_light_avg_slowdown = None
    job_complex_total_selected_runtime = -1
    job_complex_avg_slowdown = None
    physop_selected_runtime = -1
    physop_avg_slowdown = None
    physop_idx_selected_runtime = -1
    physop_idx_avg_slowdown = None
    accesspath_selected_runtime = -1
    accesspath_avg_slowdown = None
    accesspath_w_agg_selected_runtime = -1
    accesspath_w_agg_avg_slowdown = None

    predictions_dicts = dict()

    if test_loaders is not None:
        if not (model_dir is None or config.name.NAME is None):
            # In case of retraining, do not load the totally best model but the latest one
            if load_best_checkpoint:
                early_stop_m = find_early_stopping_metric(metrics)
                if early_stop_m.best_model is not None:
                    if isinstance(
                        config,
                        (ZeroShotRankingModelBaseConfig, MambaRankingModelConfig),
                    ):
                        model.load_state_dict(early_stop_m.best_model, strict=False)
                    else:
                        model.load_state_dict(early_stop_m.best_model)

            for test_path, test_loader in zip(test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = dict()
                wandb_plots_dict = dict() if wandb_run is not None else None

                _, predictions_dict = validate_model(
                    val_loader=test_loader,
                    model=model,
                    epoch=epoch,
                    epoch_stats=test_stats,
                    wandb_plots_dict=wandb_plots_dict,
                    metrics=metrics,
                    custom_batch_to=config.batch_to_func,
                    log_all_queries=True,
                    target_path=test_path,
                    bound_predictions=bound_predictions,
                    prediction_bounder=prediction_bounder,
                    update_metrics=False,  # do not update best seen metrics - this is determined by val set
                    store_model_snapshot=False,  # do not update best model snapshot - this is determined by val set
                    log_prefix=log_prefix,
                    show_best_seen=False,  # do not show - it was computed during val i.e. not relevant
                    model_config=config,
                    quick_run=quick_run,
                )

                # extract filename from path
                filename = test_path.stem
                # remove _seed from name (this is different from the _s0 suffix)
                filename = "_".join(filename.split("_")[:-1])

                predictions_dicts[filename] = predictions_dict

                if wandb_run is not None:
                    prefixed = prefix_dict(test_stats, f"{filename}.json/")
                    if is_per_epoch:
                        # prefixed_plots = prefix_dict(wandb_plots_dict, f'{filename}.json/') # do not log plots per epoch
                        wandb_run.log(prefixed, step=epoch)
                        # wandb_run.log(prefixed_plots, step=epoch)
                    else:
                        prefixed_plots = prefix_dict(
                            wandb_plots_dict, f"{filename}.json/"
                        )
                        wandb_run.summary.update(prefixed)
                        wandb_run.log(prefixed_plots)

                if not is_per_epoch:
                    save_csv(
                        [test_stats], str(test_path) + f"_test_stats{file_suffix}.csv"
                    )

                if (
                    "join_order" in filename
                    or "job_" in filename
                    or "JOBExtended" in filename
                ):
                    if filename == "join_order":
                        wl_name = "job_light"  # job_light
                    elif filename in [
                        "join_order_job",
                        "join_order_job_w_idxs_combined",
                        "job_w_optimal_w_idxs",
                    ]:
                        wl_name = "job"
                    elif "JOBExtended" in filename or "job_complex" in filename:
                        wl_name = "job_complex"
                    else:
                        wl_name = None
                        print(f"Unknown workload name: {filename}")

                    if wl_name is not None:
                        rt, slowdown = evaluate_join_order(
                            predictions=predictions_dict,
                            is_per_epoch=is_per_epoch,
                            epoch=epoch,
                            wandb_run=wandb_run,
                            wl_name=wl_name,
                        )

                        if wl_name == "job":
                            job_total_selected_runtime = rt
                            job_avg_slowdown = slowdown
                        elif wl_name == "job_light":
                            job_light_total_selected_runtime = rt
                            job_light_avg_slowdown = slowdown
                        elif wl_name == "job_complex":
                            job_complex_total_selected_runtime = rt
                            job_complex_avg_slowdown = slowdown
                        else:
                            raise ValueError(f"Unknown workload name: {wl_name}")

                if "physical_operator" in str(test_path):
                    with_idxs = "_w_idx" in str(test_path)
                    rt, slowdown = evaluate_physical_plan_selection(
                        predictions=predictions_dict,
                        with_idxs=with_idxs,
                        is_per_epoch=is_per_epoch,
                        epoch=epoch,
                        wandb_run=wandb_run,
                    )
                    if with_idxs:
                        physop_idx_selected_runtime = rt
                        physop_idx_avg_slowdown = slowdown
                    else:
                        physop_selected_runtime = rt
                        physop_avg_slowdown = slowdown
                if "access_path_selection" in str(test_path):
                    if "with_aggregate" in str(test_path):
                        ap_w_aggregate = True
                        wl_name = "w_aggregate"
                    else:
                        ap_w_aggregate = False
                        wl_name = ""

                    res = evaluate_access_path_selection(
                        predictions=predictions_dict,
                        is_per_epoch=is_per_epoch,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        wl_name=wl_name,
                    )

                    if not ap_w_aggregate:
                        (
                            accesspath_selected_runtime,
                            accesspath_avg_slowdown,
                        ) = res
                    else:
                        (
                            accesspath_w_agg_selected_runtime,
                            accesspath_w_agg_avg_slowdown,
                        ) = res
        else:
            print("Skipping saving the test stats")

    # aggregate metrics to qo_total
    if (
        job_total_selected_runtime != -1
        and job_light_total_selected_runtime != -1
        and job_complex_total_selected_runtime != -1
        and physop_selected_runtime != -1
        and physop_idx_selected_runtime != -1
        and accesspath_selected_runtime != -1
        and accesspath_w_agg_selected_runtime != -1
    ):
        print("Aggregating total selected runtimes and slowdowns across all workloads")
        print(
            "-job_total_selected_runtime:",
            job_total_selected_runtime,
            "\tAvg slowdown:",
            job_avg_slowdown,
        )
        print(
            "-job_light_total_selected_runtime:",
            job_light_total_selected_runtime,
            "\tAvg slowdown:",
            job_light_avg_slowdown,
        )
        print(
            "-job_complex_total_selected_runtime:",
            job_complex_total_selected_runtime,
            "\tAvg slowdown:",
            job_complex_avg_slowdown,
        )
        print(
            "-physop_selected_runtime:",
            physop_selected_runtime,
            "\tAvg slowdown:",
            physop_avg_slowdown,
        )
        print(
            "-physop_idx_selected_runtime:",
            physop_idx_selected_runtime,
            "\tAvg slowdown:",
            physop_idx_avg_slowdown,
        )
        print(
            "-accesspath_selected_runtime:",
            accesspath_selected_runtime,
            "\tAvg slowdown:",
            accesspath_avg_slowdown,
        )
        print(
            "-accesspath_w_agg_selected_runtime:",
            accesspath_w_agg_selected_runtime,
            "\tAvg slowdown:",
            accesspath_w_agg_avg_slowdown,
        )
        # if not all 5 workloads are present, do not aggregate
        assert job_total_selected_runtime is not None
        assert job_light_total_selected_runtime is not None
        assert job_complex_total_selected_runtime is not None
        assert physop_selected_runtime is not None
        assert physop_idx_selected_runtime is not None
        assert accesspath_selected_runtime is not None
        assert accesspath_w_agg_selected_runtime is not None

        join_order_total_selected_runtime = (
            job_total_selected_runtime
            + job_light_total_selected_runtime
            + job_complex_total_selected_runtime
        )
        access_path_total_selected_runtime = (
            accesspath_w_agg_selected_runtime + accesspath_selected_runtime
        )
        physop_total_selected_runtime = (
            physop_selected_runtime + physop_idx_selected_runtime
        )

        jo_ap_po_total_selected_runtime = (
            join_order_total_selected_runtime
            + access_path_total_selected_runtime
            + physop_total_selected_runtime
        )

        ac_po_total_selected_runtime = (
            access_path_total_selected_runtime + physop_total_selected_runtime
        )

        total_selected_runtime = (
            job_total_selected_runtime
            + job_light_total_selected_runtime
            + physop_selected_runtime
            + physop_idx_selected_runtime
            + accesspath_selected_runtime
        )

        avg_slowdown = (
            job_avg_slowdown
            + job_light_avg_slowdown
            + physop_avg_slowdown
            + physop_idx_avg_slowdown
            + accesspath_avg_slowdown
        ) / 5

        if wandb_run is not None:
            if is_per_epoch:
                wandb_run.log(
                    {
                        "qo_total_per_epoch/selected_runtime": total_selected_runtime,
                        "qo_total_per_epoch/avg_slowdown": avg_slowdown,
                        "qo_total_with_complex_per_epoch/selected_runtime": total_selected_runtime
                        + job_complex_total_selected_runtime,
                        "qo_total_per_epoch/jo_ap_po_selected_runtime": jo_ap_po_total_selected_runtime,
                        "qo_total_per_epoch/ac_po_selected_runtime": ac_po_total_selected_runtime,
                        "qo_total_per_epoch/jo_selected_runtime": join_order_total_selected_runtime,
                        "qo_total_per_epoch/ap_selected_runtime": access_path_total_selected_runtime,
                        "qo_total_per_epoch/po_selected_runtime": physop_total_selected_runtime,
                    },
                    step=epoch,
                )
            else:
                wandb_run.summary.update(
                    {
                        "qo_total/selected_runtime": total_selected_runtime,
                        "qo_total/avg_slowdown": avg_slowdown,
                        "qo_total_with_complex/selected_runtime": total_selected_runtime
                        + job_complex_total_selected_runtime,
                        "qo_total/jo_ap_po_selected_runtime": jo_ap_po_total_selected_runtime,
                        "qo_total/ac_po_selected_runtime": ac_po_total_selected_runtime,
                        "qo_total/jo_selected_runtime": join_order_total_selected_runtime,
                        "qo_total/ap_selected_runtime": access_path_total_selected_runtime,
                        "qo_total/po_selected_runtime": physop_total_selected_runtime,
                    }
                )

    # create plot: cardinality estimation error vs. q-error
    import matplotlib.pyplot as plt

    # Collect data for plotting
    x_qerror = []
    y_card_error = []
    colors = []
    color_map = {}
    color_list = plt.cm.tab10.colors  # Up to 10 datasets

    for dataset_name, query_stats_dicts in predictions_dicts.items():
        card_error = [q["card_error_root_qerror"] for q in query_stats_dicts]
        runtime_error = [q["qerror"] for q in query_stats_dicts]

        x_qerror.extend(runtime_error)
        y_card_error.extend(card_error)

        # Assign a color to the dataset
        if dataset_name not in color_map:
            color_map[dataset_name] = color_list[len(color_map) % len(color_list)]

        colors.extend([color_map[dataset_name]] * len(card_error))

    # Improved plotting with marginals using seaborn's jointplot
    # Prepare DataFrame for seaborn
    df = pd.DataFrame(
        {
            "Runtime Q-Error": np.log10(x_qerror),
            "Cardinality Q-Error": np.log10(y_card_error),
            "Dataset": [
                next(ds for ds, col in color_map.items() if col == color)
                for color in colors
            ],
        }
    )

    # Use seaborn's jointplot for scatter and marginals, start axes at 1
    jp = sns.jointplot(
        data=df,
        x="Runtime Q-Error",
        y="Cardinality Q-Error",
        hue="Dataset",
        palette=color_map,
        kind="scatter",
        height=10,
        marginal_kws={
            "fill": False,
            "bw_adjust": 1,
            "common_norm": False,
        },
        alpha=0.7,
        s=4,
    )

    # Clear the original marginals (we will re-plot them)
    jp.ax_marg_x.cla()
    jp.ax_marg_y.cla()

    # Plot normalized marginal histograms manually
    # Plot normalized marginal histograms manually, without legend
    sns.kdeplot(
        data=df,
        x="Runtime Q-Error",
        hue="Dataset",
        ax=jp.ax_marg_x,
        legend=False,
        common_norm=False,
        palette=color_map,
    )
    sns.kdeplot(
        data=df,
        y="Cardinality Q-Error",
        hue="Dataset",
        ax=jp.ax_marg_y,
        legend=False,
        common_norm=False,
        palette=color_map,
    )

    # Add frame to marginals
    jp.ax_marg_x.spines["top"].set_visible(True)
    jp.ax_marg_x.spines["right"].set_visible(True)
    jp.ax_marg_x.spines["left"].set_visible(True)
    jp.ax_marg_x.spines["bottom"].set_visible(True)

    jp.ax_marg_y.spines["top"].set_visible(True)
    jp.ax_marg_y.spines["right"].set_visible(True)
    jp.ax_marg_y.spines["left"].set_visible(True)
    jp.ax_marg_y.spines["bottom"].set_visible(True)

    # Move legend below the plot with increased dot size
    handles, labels = jp.ax_joint.get_legend_handles_labels()
    for h in handles:
        # For PathCollection (scatter), set_sizes sets the legend marker size
        if hasattr(h, "set_sizes"):
            h.set_sizes([200])  # Increase dot size in legend (default is ~20)
        # For Line2D (sometimes used), set_markersize sets the legend marker size
        elif hasattr(h, "set_markersize"):
            h.set_markersize(6)  # Increase dot size in legend

    jp.ax_joint.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        scatterpoints=1,  # Ensures legend uses scatter markers
    )

    # Set cutoff at 95% of the datapoints as upper limits for axes
    cutoff_x = np.percentile(df["Runtime Q-Error"], 99)
    cutoff_y = np.percentile(df["Cardinality Q-Error"], 99)

    if not np.isfinite(cutoff_x):
        # set to max entry
        cutoff_x = df["Runtime Q-Error"][np.isfinite(df["Runtime Q-Error"])].max()

    if not np.isfinite(cutoff_y):
        # set to max entry
        cutoff_y = df["Cardinality Q-Error"][
            np.isfinite(df["Cardinality Q-Error"])
        ].max()

    # add logscale
    # jp.ax_joint.set_xscale("log")
    # jp.ax_joint.set_yscale("log")

    jp.ax_joint.set_xlim(left=-0.01, right=cutoff_x)
    jp.ax_joint.set_ylim(bottom=-0.04, top=cutoff_y)
    jp.ax_marg_x.set_xlim(left=-0.01, right=cutoff_x)
    jp.ax_marg_y.set_ylim(bottom=-0.04, top=cutoff_y)

    jp.ax_joint.set_xlabel("Runtime Q-Error (log$_{10}$ scale)")
    jp.ax_joint.set_ylabel("Cardinality Q-Error (log$_{10}$ scale)")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    # log to wandb if available
    if wandb_run is not None:
        if is_per_epoch:
            wandb_run.log(
                {"cardinality_vs_qerror": wandb.Image(plt)},
                step=epoch,
            )
        else:
            wandb_run.log({"cardinality_vs_qerror": wandb.Image(plt)})

    # create a boxplot for runtime and cardinality q-error
    plt.close()  # Close the plot to free memory
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

    # Boxplot: Runtime Q-Error
    sns.boxplot(
        data=df,
        x="Dataset",
        y="Runtime Q-Error",
        palette=color_map,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_ylabel("Runtime Q-Error (log$_{10}$ scale)")
    axes[0].set_title("Runtime Q-Error Boxplot")
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")

    # Boxplot: Cardinality Q-Error
    sns.boxplot(
        data=df,
        x="Dataset",
        y="Cardinality Q-Error",
        palette=color_map,
        ax=axes[1],
        legend=True,
    )
    axes[1].set_ylabel("Cardinality Q-Error (log$_{10}$ scale)")
    axes[1].set_title("Cardinality Q-Error Boxplot")
    axes[1].set_xticklabels([])
    axes[1].set_xlabel("")

    # # Add shared legend below both plots
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[ds]) for ds in color_map]
    fig.legend(
        handles=handles,
        labels=list(color_map.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        frameon=False,
    )

    # create tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as filename:
        plt.savefig(filename.name, bbox_inches="tight", dpi=300)

        # log to wandb if available
        if wandb_run is not None:
            if is_per_epoch:
                wandb_run.log(
                    {"runtime_cardinality_boxplot": wandb.Image(filename.name)},
                    step=epoch,
                )
            else:
                wandb_run.log(
                    {"runtime_cardinality_boxplot": wandb.Image(filename.name)}
                )

        plt.close()  # Close the plot to free memory
