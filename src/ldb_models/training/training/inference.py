from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.batch_to_funcs import zs_batch_to


def inference(
    dataloader: DataLoader, model, batch_to: Callable = zs_batch_to
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = []
    preds = []

    model.eval()

    for batch in tqdm(dataloader, desc="Inference: "):
        (input_model, label, query_stats) = batch_to(
            batch, model.device, model.label_norm
        )

        with torch.no_grad():
            # Extra kwargs are consumed by loop-aware variants;
            # base zero-shot forward paths simply ignore optional kwargs they do not use.
            model_kwargs = {
                "plan_dict": query_stats.get("plan_dict", None),
                "loop_multiplier_map": query_stats.get("loop_multiplier_map", None),
            }
            if "subplan_est_card_labels" in query_stats:
                model_kwargs["subplan_est_card_labels"] = query_stats[
                    "subplan_est_card_labels"
                ]
            output = model(
                input_model,
                **model_kwargs,
            )

        if model.label_norm is not None:
            pred = model.label_norm.inverse_transform(output).to("cpu")
            label = model.label_norm.inverse_transform(label.reshape(-1, 1)).to("cpu")
        else:
            pred = output

        labels.append(label)
        preds.append(pred)

    # Concatenate all labels and predictions
    labels = torch.concatenate(labels)
    preds = torch.concatenate(preds)

    return labels, preds
