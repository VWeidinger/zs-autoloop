import collections
import hashlib
import json
import os

import torch
from ldb_models.training.training.outputs import LossOutput, ModelOutput


def q_error(preds: torch.Tensor, labels: torch.Tensor):
    """Calculate the Q-Error between predictions and labels."""
    preds = torch.clamp(preds, min=1e-6)  # Prevent division by zero
    labels = torch.clamp(labels, min=1e-6)  # Prevent division by zero
    q_errors = torch.max(preds / labels, labels / preds)
    return q_errors.detach().cpu().numpy()


def cascade_debug_print(top_k, top_k_indices, plans, y, r_hat, **kwargs):
    print(f"Cascade Ranking - Filtered Top-{top_k}: ")
    rows = []
    ordered = r_hat.cpu().argsort().tolist()

    for pos, idx in enumerate(top_k_indices):
        filter_pred = float(plans[idx])
        filter_rank = int(torch.argsort(plans).cpu().argsort()[idx])
        rank_model_pred = float(y[pos])
        rescaled_pred = float(r_hat[pos])
        predicted_rank_in_batch = ordered.index(pos)

        entry = {
            "rescaled": rescaled_pred,
            "filter_pred": filter_pred,
            "filter_rank": filter_rank,
            "final": rank_model_pred,
            "pred_rank": predicted_rank_in_batch,
        }

        plan_labels = kwargs.get("plan_labels", None)
        if plan_labels is not None:
            true_label = float(plan_labels[idx])
            true_rank = int(torch.argsort(plan_labels).cpu().argsort()[idx])
            entry["true_label"] = true_label
            entry["true_rank"] = true_rank

        rows.append(entry)

    for r in sorted(rows, key=lambda x: x["rescaled"]):
        line = (
            f"  rescaled={r['rescaled']:.3f}\t| "
            f"filter={r['filter_pred']:.3f} | "
            f"rank={r['filter_rank']}\t| "
            f"final={r['final']:.3} | "
            f"rank={r['pred_rank']}\t"
        )
        if "true_label" in r:
            line += f"| true={r['true_label']:.3f} | rank={r['true_rank']}"
        print(line)


def print_debug_table(
    batch_idx,
    loss_output: LossOutput,
    output: ModelOutput,
    labels: torch.Tensor,
    query_stats,
):
    """Prints a formatted table of predictions vs. labels for debugging."""
    subplan_labels, subplan_card_labels, subplan_est_card_labels = None, None, None
    if "subplan_labels" in query_stats:
        subplan_labels = query_stats["subplan_labels"]
    if "subplan_card_labels" in query_stats:
        subplan_card_labels = query_stats["subplan_card_labels"]
    if "subplan_est_card_labels" in query_stats:
        subplan_est_card_labels = query_stats["subplan_est_card_labels"]
    print(f"\n--- DEBUG: Batch {batch_idx} ---", flush=True)
    plan_loss = (
        loss_output.plan_loss.item() if loss_output.plan_loss is not None else 0.0
    )
    subplan_loss = (
        loss_output.subplan_loss.item() if loss_output.subplan_loss is not None else 0.0
    )
    card_loss = (
        loss_output.card_loss.item() if loss_output.card_loss is not None else 0.0
    )
    main_loss = loss_output.loss.item()
    print(f"Total Loss: {main_loss:.4f}", flush=True)
    print(f"Plan Loss: {plan_loss:.4f}", flush=True)
    if subplan_loss > 0:
        print(f"Subplan Op Time Loss: {subplan_loss:.4f}", flush=True)
    if card_loss > 0:
        print(f"Cardinality Loss: {card_loss:.4f}", flush=True)

    # --- Prepare data for each column ---
    num_samples = min(5, labels.shape[0])
    plan_preds = (
        output.plan_predictions[:num_samples].detach().cpu().squeeze(-1).numpy()
    )
    plan_labels = labels[:num_samples].detach().cpu().numpy()

    columns = [
        {
            "header1": "Plan Runtimes (ms)",
            "header2": "Pred    vs    Label",
            "width": 23,
            "format": "{p:>7.3f} vs {l:>7.3f}",
            "preds": plan_preds,
            "labels": plan_labels,
        }
    ]

    if output.subplan_predictions is not None:
        sp_preds = (
            output.subplan_predictions[:num_samples].detach().cpu().squeeze(-1).numpy()
        )
        sp_labels = subplan_labels[:num_samples].detach().cpu().numpy()
        columns.append(
            {
                "header1": "Subplan Op Times (ms)",
                "header2": "Pred     vs     Label",
                "width": 26,
                "format": "{p:>9.5f} vs {l:>9.5f}",
                "preds": sp_preds,
                "labels": sp_labels,
            }
        )

    if output.subplan_card_predictions is not None:
        card_preds = (
            output.subplan_card_predictions[:num_samples]
            .detach()
            .cpu()
            .squeeze(-1)
            .numpy()
        )
        card_labels = subplan_card_labels[:num_samples].detach().cpu().numpy()
        est_card_labels = subplan_est_card_labels[:num_samples].detach().cpu().numpy()
        columns.append(
            {
                "header1": "Cardinality Estimates",
                "header2": "Pred  vs  Est  vs  Act",
                "width": 30,
                "format": "{p:>7.1f} vs {l:>5.1f} vs {el:>5.1f}",
                "preds": card_preds,
                "labels": est_card_labels,
                "extra_labels": card_labels,
            }
        )

    if loss_output.relevancy_scores_preds is not None:
        num_samples_relevancy = min(5, loss_output.relevancy_scores_preds.shape[0])
        rel_preds = (
            loss_output.relevancy_scores_preds[:num_samples_relevancy]
            .squeeze(-1)
            .numpy()
        )
        # Handle cases where labels weren't computed or passed
        if loss_output.relevancy_scores_labels is not None:
            rel_labels = (
                loss_output.relevancy_scores_labels[:num_samples_relevancy]
                .squeeze(-1)
                .numpy()
            )
        else:
            rel_labels = [0.0] * num_samples_relevancy

        columns.append(
            {
                "header1": "Relevancy Scores",
                "header2": "Pred     vs     Label",
                "width": 26,
                "format": "{p:>9.4f} vs {l:>9.4f}",
                "preds": rel_preds,
                "labels": rel_labels,
            }
        )

    # --- Build and print table ---
    separator = "+-" + "-+-".join(["-" * c["width"] for c in columns]) + "-+"
    header1_str = " | ".join([c["header1"].center(c["width"]) for c in columns])
    header2_str = " | ".join([c["header2"].center(c["width"]) for c in columns])

    print("\n" + separator)
    print(f"| {header1_str} |")
    print(f"| {header2_str} |")
    print(separator)

    for i in range(num_samples):
        row_parts = []
        for col in columns:
            pred, label = col["preds"][i], col["labels"][i]
            if "extra_labels" in col:
                # If so, get the extra label and format with all three values
                extra_label = col["extra_labels"][i]
                cell_content = col["format"].format(p=pred, l=label, el=extra_label)
            else:
                # Otherwise, format as before with just p and l
                cell_content = col["format"].format(p=pred, l=label)
            row_parts.append(cell_content.center(col["width"]))
        print(f"| {' | '.join(row_parts)} |")

    print(separator, flush=True)

    if output.subplan_card_predictions is not None:
        est_mean_q_error = q_error(subplan_est_card_labels, subplan_card_labels).mean()
        pred_mean_q_error = q_error(
            output.subplan_card_predictions, subplan_card_labels
        ).mean()
        print(
            f"Mean Q-Error (Cardinality Estimates): {est_mean_q_error:.4f}", flush=True
        )
        print(
            f"Mean Q-Error (Cardinality Predictions): {pred_mean_q_error:.4f}",
            flush=True,
        )


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def find_early_stopping_metric(metrics):
    potential_metrics = [m for m in metrics if m.early_stopping_metric]
    assert len(potential_metrics) == 1
    early_stopping_metric = potential_metrics[0]
    return early_stopping_metric


def save_config(save_dict, target_dir, json_name):
    os.makedirs(target_dir, exist_ok=True)
    target_params_path = os.path.join(target_dir, json_name)
    print(f"Saving best params to {target_params_path}")

    with open(target_params_path, "w") as f:
        json.dump(save_dict, f)


def prefix_dict(dictionary, prefix):
    return {prefix + k: v for k, v in dictionary.items()}
