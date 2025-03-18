from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.batch_to_funcs import batch_to


def inference(dataloader: DataLoader, model, batch_to: Callable = batch_to) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = []
    preds = []

    model.eval()

    for batch in tqdm(dataloader, desc='Inference: '):
        input_model, label, sample_idxs_batch, sql, hints, selectivity, scan_op, join_op = batch_to(batch,
                                                                                                    model.device,
                                                                                                    model.label_norm)

        with torch.no_grad():
            output = model(input_model)

        if model.label_norm is not None:
            pred = model.label_norm.inverse_transform(output).to('cpu')
            label = model.label_norm.inverse_transform(label.reshape(-1, 1)).to('cpu')
        else:
            pred = output

        labels.append(label)
        preds.append(pred)

    # Concatenate all labels and predictions
    labels = torch.concatenate(labels)
    preds = torch.concatenate(preds)

    return labels, preds
