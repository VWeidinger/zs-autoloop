from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from wandb.sdk.wandb_run import Run

from training.training.metrics import SpearmanRank, SelectedRuntime, SurpassedPlans, QError, MaxOverestimation, \
    MaxUnderestimation, PickRate, OptimalRuntime
import time


def evaluate_join_order(predictions, is_per_epoch: bool = False, epoch: int = None, run: Run = None):
    """ This method evaluates a given workload of queries, how well the join order was predicted."""
    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {
        "spearman_corr": SpearmanRank(),
        "selected_runtime": SelectedRuntime(),
        "surpassed_plans": SurpassedPlans(),
        "q_error": QError(),
        "max_overestimation": MaxOverestimation(),
        "max_underestimation": MaxUnderestimation()}

    aggregated_results = {key: [] for key in metrics}

    prefix = 'join_order_eval'
    if is_per_epoch:
        prefix += '-per_epoch'

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"]
        labels = group_df["label"]

        # print(f"Found {len(group_df)} predictions for SQL query in {sql}")

        if len(group_df) == 1 or pd.isna(SpearmanRank().evaluate_metric(preds=preds, labels=labels)):
            print(f'Join order evaluation: Skipping SQL query in {sql} as Spearman correlation is invalid/NaN')
            continue

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

        search_sql = 'SELECT COUNT(*) FROM title,movie_info,movie_info_idx,movie_companies WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.id=movie_companies.movie_id AND movie_info_idx.info_type_id=101 AND movie_info.info_type_id=3 AND title.production_year>2005 AND title.production_year<2008 AND movie_companies.company_type_id=2;'
        if search_sql == sql:
            # sort both lists by labels
            sorted_labels, sorted_preds = zip(*sorted(zip(labels, preds)))
            # assert len(sorted_labels) == len(sorted_preds) and  len(sorted_labels) == 120, "Missing labels for example query"
            if run:
                # create a plot
                data = {prefix: wandb.plot.line_series(
                    xs=[i for i in range(len(sorted_labels))],
                    ys=[sorted_labels, sorted_preds],
                    keys=['Real Runtime', 'Predictions'],
                    title='Runtime Enumeration Plot JOB-Light Query 33',
                    xname='Join Enumeration',
                )}
                if is_per_epoch:
                    wandb.log(data, step=epoch)
                else:
                    wandb.log(data)

    avg_spearman_corr = np.mean(aggregated_results['spearman_corr'])
    total_selected_runtime = np.sum(aggregated_results['selected_runtime'])
    avg_surpassed_plans = np.mean(aggregated_results['surpassed_plans'])
    avg_q_error = np.mean(aggregated_results['q_error'])
    avg_max_overestimation = np.mean(aggregated_results['max_overestimation'])
    avg_max_underestimation = np.mean(aggregated_results['max_underestimation'])

    print("Join order evaluation results:")
    print(f"Average Spearman correlation: {avg_spearman_corr}")
    print(f"Total Selected runtime: {total_selected_runtime}")
    print(f"Average Ratio Surpassed plans: {avg_surpassed_plans}")
    print(f"Average QError over plan alternatives: {avg_q_error}")
    print(f"Average Max Overestimation: {avg_max_overestimation}")
    print(f"Average Max Underestimation: {avg_max_underestimation}")

    if run:
        if is_per_epoch:
            wandb.log({
                f'{prefix}/avg_spearman_corr': avg_spearman_corr,
                f'{prefix}/total_selected_runtime': total_selected_runtime,
                f'{prefix}/avg_surpassed_plans': avg_surpassed_plans,
                f'{prefix}/avg_q_error': avg_q_error,
                f'{prefix}/avg_max_overestimation': avg_max_overestimation,
                f'{prefix}/avg_max_underestimation': avg_max_underestimation
            }, step=epoch)
        else:
            wandb.run.summary[f'{prefix}/avg_spearman_corr'] = avg_spearman_corr
            wandb.run.summary[f'{prefix}/total_selected_runtime'] = total_selected_runtime
            wandb.run.summary[f'{prefix}/avg_surpassed_plans'] = avg_surpassed_plans
            wandb.run.summary[f'{prefix}/avg_q_error'] = avg_q_error
            wandb.run.summary[f'{prefix}/avg_max_overestimation'] = avg_max_overestimation
            wandb.run.summary[f'{prefix}/avg_max_underestimation'] = avg_max_underestimation

    return


def evaluate_physical_plan_selection(predictions: dict, with_idxs: bool = False, is_per_epoch: bool = False,
                                     epoch: int = None, run: Run = None):
    """ This method evaluates a given workload of queries, how well physical operators were selected."""

    log_prefix = f'physop_eval'
    if with_idxs:
        log_prefix += '_with_idxs'

    if is_per_epoch:
        log_prefix += '-per_epoch'

    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {"selected_runtime": SelectedRuntime(),
               "pick_rate": PickRate(),
               'optimal_runtime': OptimalRuntime()}
    aggregated_results = {key: [] for key in metrics}

    optimal_join_op_ctr = {
        'Hash Join': 0,
        'Merge Join': 0,
        'Index Nested Loop': 0,
        'Nested Loop': 0,

    }
    selected_join_op_ctr = {
        'Hash Join': 0,
        'Merge Join': 0,
        'Index Nested Loop': 0,
        'Nested Loop': 0,
    }

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"].astype(float)
        labels = group_df["label"].astype(float)

        if len(group_df) != 3:
            print(f"Physical operator selection requires "
                  f"three different query plans per SQL statement. Got: {len(group_df)}")
            continue

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

        # optimal plan
        optimal_row_id = labels.idxmin()
        optimal_join_op = group_df.loc[optimal_row_id, 'join_op']
        optimal_join_op_ctr[optimal_join_op] += 1

        # selected plan
        selected_row_id = preds.idxmin()
        selected_join_op = group_df.loc[selected_row_id, 'join_op']
        selected_join_op_ctr[selected_join_op] += 1

    total_selected_runtime = np.sum(aggregated_results['selected_runtime'])
    total_optimal_runtime = np.sum(aggregated_results['optimal_runtime'])
    avg_pick_rate = np.mean(aggregated_results['pick_rate'])

    print("Physical operator evaluation results:")
    print(f"Total Selected runtime: {total_selected_runtime}")
    print(f"Total Optimal runtime: {total_optimal_runtime}")
    print(f"Average Pick Rate: {avg_pick_rate}")

    if run:
        if is_per_epoch:
            wandb.log({
                f'{log_prefix}/total_selected_runtime': total_selected_runtime,
                f'{log_prefix}/total_optimal_runtime': total_optimal_runtime,
                f'{log_prefix}/avg_pick_rate': avg_pick_rate
            }, step=epoch)
        else:
            wandb.run.summary[f'{log_prefix}/total_selected_runtime'] = total_selected_runtime
            wandb.run.summary[f'{log_prefix}/total_optimal_runtime'] = total_optimal_runtime
            wandb.run.summary[f'{log_prefix}/avg_pick_rate'] = avg_pick_rate

    total = sum(optimal_join_op_ctr.values())
    optimal_join_op_ctr = {k: v / total for k, v in optimal_join_op_ctr.items()}
    assert total == sum(selected_join_op_ctr.values())
    selected_join_op_ctr = {k: v / total for k, v in selected_join_op_ctr.items()}

    def autopct_format(values):
        def my_format(pct):
            return ('%1.1f%%' % pct) if pct > 2 else ''

        return my_format

    # create pie chart for optimal join operator distribution and predicted join operator distribution
    plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].pie(optimal_join_op_ctr.values(), autopct=autopct_format(optimal_join_op_ctr.values()))
    axs[0].set_title('Optimal Join Operator Distribution')

    axs[1].pie(selected_join_op_ctr.values(), autopct=autopct_format(selected_join_op_ctr.values()))
    axs[1].set_title('Predicted Join Operator Distribution')

    # create legend
    fig.legend(optimal_join_op_ctr.keys(), loc='lower center')
    plt.show()

    # log to wandb
    if run:
        if is_per_epoch:
            wandb.log({f'{log_prefix}/join_op_distribution': wandb.Image(plt)}, step=epoch)
        else:
            wandb.log({f'{log_prefix}/join_op_distribution': wandb.Image(plt)})

    return


def evaluate_access_path_selection(predictions: dict, is_per_epoch: bool = False, epoch: int = None, run: Run = None):
    """ This method evaluates a given workload of queries, how well access paths were selected.
    """

    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {"selected_runtime": SelectedRuntime(), 'optimal_runtime': OptimalRuntime()}
    aggregated_results = {key: [] for key in metrics}

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"]
        labels = group_df["label"]

        if len(group_df) != 2:
            print(
                f'Access path selection requires exactly two different query plans per SQL statement. Got: {len(group_df)}')
            continue

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

    total_selected_runtime = np.sum(aggregated_results['selected_runtime'])
    total_optimal_runtime = np.sum(aggregated_results['optimal_runtime'])

    print("Access path evaluation results:")
    print(f"Total Selected runtime: {total_selected_runtime}")
    print(f"Total Optimal runtime: {total_optimal_runtime}")

    prefix = 'accesspath_eval'
    if is_per_epoch:
        prefix += '-per_epoch'

    if run:
        if is_per_epoch:
            wandb.log({
                f'{prefix}/total_selected_runtime': total_selected_runtime,
                f'{prefix}/total_optimal_runtime': total_optimal_runtime
            }, step=epoch)
        else:
            wandb.run.summary[f'{prefix}/total_selected_runtime'] = total_selected_runtime
            wandb.run.summary[f'{prefix}/total_optimal_runtime'] = total_optimal_runtime

    def extract_table_col(sql):
        table_name = sql.split("FROM")[1].split("WHERE")[0].strip()
        col_name = sql.split('WHERE')[1].strip().split(' ')[0]
        return table_name, col_name

    table_names, col_names = zip(*[extract_table_col(sql) for sql in prediction_dataframe['sql']])

    prediction_dataframe['table_name'] = table_names
    prediction_dataframe['col_name'] = col_names

    optimal_scan_op_list_dict = defaultdict(list)  # sel -> list
    predicted_scan_op_list_dict = defaultdict(list)

    for (table_name, col_name), group_df in prediction_dataframe.groupby(['table_name', 'col_name']):
        sel_runtimes_idx_dict = dict()
        sel_runtimes_seq_dict = dict()

        # iterate over group_df
        idx_labels_dict = {}
        idx_preds_dict = {}
        seq_labels_dict = {}
        seq_preds_dict = {}
        for idx, row in group_df.iterrows():
            if 'IndexScan' in row['hint']:
                if 'Index Scan' != row['scan_op']:
                    # not necessary - discarding plans where hint was not obeyed in dataset creation
                    raise Exception(f'Hint {row["hint"]} and scan_op {row["scan_op"]} do not match')
                    continue
                sel_runtimes_idx_dict[float(row['selectivity'])] = {'label': float(row['label']),
                                                                    'prediction': float(row['prediction'])}
                idx_labels_dict[row['sql']] = float(row['label'])
                idx_preds_dict[row['sql']] = float(row['prediction'])
            elif 'SeqScan' in row['hint']:
                if 'Seq Scan' != row['scan_op']:
                    # not necessary - discarding plans where hint was not obeyed in dataset creation
                    raise Exception(f'Hint {row["hint"]} and scan_op {row["scan_op"]} do not match')
                    continue
                sel_runtimes_seq_dict[float(row['selectivity'])] = {'label': float(row['label']),
                                                                    'prediction': float(row['prediction'])}
                seq_labels_dict[row['sql']] = float(row['label'])
                seq_preds_dict[row['sql']] = float(row['prediction'])
            else:
                raise Exception(f"Unknown scan op in hint: {row['hint']}")

        sel_winners = dict()
        for sel in sel_runtimes_seq_dict.keys():

            if sel not in sel_runtimes_idx_dict or sel_runtimes_seq_dict[sel]['prediction'] < \
                    sel_runtimes_idx_dict[sel]['prediction']:
                sel_winners[sel] = sel_runtimes_seq_dict[sel]['label']
                predicted_scan_op_list_dict[sel].append('SeqScan')
            else:
                sel_winners[sel] = sel_runtimes_idx_dict[sel]['label']
                predicted_scan_op_list_dict[sel].append('IndexScan')

            if sel_runtimes_seq_dict[sel]['label'] < sel_runtimes_idx_dict[sel]['label']:
                optimal_scan_op_list_dict[sel].append('SeqScan')
            else:
                optimal_scan_op_list_dict[sel].append('IndexScan')

        x_sel = sorted(list(sel_winners.keys()), key=lambda x: float(x))
        y_seq = [sel_runtimes_seq_dict[sel]['label'] for sel in x_sel]
        x_idx_sel = [sel for sel in x_sel if sel in sel_runtimes_idx_dict]
        y_idx = [sel_runtimes_idx_dict[sel]['label'] for sel in x_sel if sel in sel_runtimes_idx_dict]
        y_winner = [sel_winners[sel] for sel in x_sel]

        plt.close()
        plt.plot(x_sel, y_seq, label='SeqScan')
        plt.plot(x_idx_sel, y_idx, label='IndexScan')

        # add x for each winner
        plt.scatter(x_sel, y_winner, color='red', marker='x')
        plt.xlabel('Selectivity')
        plt.ylabel('Runtime (s)')
        plt.legend()
        plt.title(f"{table_name}.{col_name}")
        plt.show()

        if run:
            # log to wandb
            if is_per_epoch:
                wandb.log({f'{prefix}/col_plots/{table_name}.{col_name}': wandb.Image(plt)}, step=epoch)
            else:
                wandb.log({f'{prefix}/col_plots/{table_name}.{col_name}': wandb.Image(plt)})

        # selection list
        pred_selection = []
        for sql in seq_labels_dict.keys():
            if sql not in idx_preds_dict:
                idx_preds_dict[sql] = np.inf
            pred_selection.append(idx_preds_dict[sql] < seq_preds_dict[sql])

        label_selection = []
        for sql in seq_labels_dict.keys():
            if sql not in idx_labels_dict:
                idx_labels_dict[sql] = np.inf
            label_selection.append(idx_labels_dict[sql] < seq_labels_dict[sql])

        # calculate accuracy
        accuracy = np.mean([p == l for p, l in zip(pred_selection, label_selection)])
        balanced_accuracy = balanced_accuracy_score(label_selection, pred_selection)

        print(f'{table_name}.{col_name} - Accuracy: {accuracy:.2f} - Balanced Accuracy: {balanced_accuracy:.2f}')
        if run:
            if is_per_epoch:
                wandb.log({
                    f'{prefix}/acc/{table_name}.{col_name}': accuracy,
                    f'{prefix}/bal_acc/{table_name}.{col_name}': balanced_accuracy
                }, step=epoch, commit=False)
            else:
                wandb.run.summary[f'{prefix}/acc/{table_name}.{col_name}'] = accuracy
                wandb.run.summary[f'{prefix}/bal_acc/{table_name}.{col_name}'] = balanced_accuracy
        else:
            time.sleep(0.1)  # Otherwise too many requests for localhost

    # bucketize dict
    bucketized_predicted_scan_op_list_dict = defaultdict(list)
    bucketized_optimal_scan_op_list_dict = defaultdict(list)
    for sel in predicted_scan_op_list_dict.keys():
        bucketized_sel = round(sel, 1)
        bucketized_predicted_scan_op_list_dict[bucketized_sel].extend(predicted_scan_op_list_dict[sel])
        bucketized_optimal_scan_op_list_dict[bucketized_sel].extend(optimal_scan_op_list_dict[sel])

    # create ratio of seqeuential scans with increasing selectivity
    x_sel_list = sorted([float(s) for s in bucketized_predicted_scan_op_list_dict.keys()])
    y_seq_ratio_pred = []
    y_seq_ratio_labels = []

    for sel in x_sel_list:
        y_seq_ratio_pred.append(
            np.mean([1 if op == 'SeqScan' else 0 for op in bucketized_predicted_scan_op_list_dict[sel]]))
        y_seq_ratio_labels.append(
            np.mean([1 if op == 'SeqScan' else 0 for op in bucketized_optimal_scan_op_list_dict[sel]]))

    plt.close()
    plt.plot(x_sel_list, y_seq_ratio_labels, label='Optimal')
    plt.plot(x_sel_list, y_seq_ratio_pred, label='Predicted')
    plt.xlabel('Selectivity')
    plt.ylabel('Ratio SeqScan')
    plt.legend()
    plt.title('SeqScan Ratio')
    plt.show()

    if run:
        if is_per_epoch:
            wandb.log({f'{prefix}/seq_scan_ratio': wandb.Image(plt)}, step=epoch)
        else:
            print(f'Logging seq_scan_ratio to wandb summary',flush=True)
            wandb.run.summary[f'{prefix}/overall/seq_scan_ratio'] = wandb.Image(plt)
