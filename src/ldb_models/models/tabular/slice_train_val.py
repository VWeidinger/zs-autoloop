import numpy as np


def slice_train_val(
    cap_training_samples, feature_vecs, labels, plans, val_ratio, shuffle=True
):
    feature_vecs = np.array(feature_vecs)
    labels = np.array(labels)
    no_plans = len(plans)
    plan_idxs = list(range(no_plans))
    if shuffle:
        np.random.shuffle(plan_idxs)
    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_idxs = plan_idxs[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        train_idxs = train_idxs[:cap_training_samples]
    X_val = None
    Y_val = None
    if val_ratio > 0:
        val_idxs = plan_idxs[split_train:]
        X_val = feature_vecs[val_idxs]
        Y_val = labels[val_idxs]
    X_train = feature_vecs[train_idxs]
    Y_train = labels[train_idxs]
    return X_train, X_val, Y_train, Y_val
