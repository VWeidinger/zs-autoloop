from typing import List
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from models.t3.benchmarked_query import BenchmarkedQuery
from models.t3.features import FeatureMapper
from models.t3.model import TreeModel, PerTupleTreeModel, FlatTreeModel


def optimize_tree_model(queries: List[BenchmarkedQuery], verbose: bool = False) -> TreeModel:
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []
    for query in queries:
        for x, y in query.get_pipeline_runtime_data(feature_mapper):
            x_vectors.append(x)
            y_values.append(y)
    x = np.vstack(x_vectors)
    y = np.array(y_values)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for _ in range(200):
        bst.update()
        if verbose:
            print(bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return TreeModel(bst)



def optimize_flat_tree_model(queries: list[BenchmarkedQuery], verbose: bool = False) -> FlatTreeModel:
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []
    for query in queries:
        current_x = []
        current_y = []
        for x, y in query.get_pipeline_runtime_data(feature_mapper):
            current_x.append(x)
            current_y.append(y)
        x_vectors.append(np.sum(current_x, axis=0))
        y_values.append(float(np.sum(current_y)))
    x = np.vstack(x_vectors)
    y = np.array(y_values)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for _ in range(200):
        bst.update()
        if verbose:
            print(bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return FlatTreeModel(bst)


def optimize_per_tuple_tree_model(queries: list[BenchmarkedQuery], verbose: bool = False) -> PerTupleTreeModel:
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []
    for query in queries:
        for x, y in query.get_per_tuple_pipeline_runtime_data(feature_mapper):
            if np.any(x != 0):
                x_vectors.append(x)
                y_values.append(y)
    x = np.vstack(x_vectors)
    y = np.array(y_values)
    # log scale improves training
    y = np.maximum(y, 1e-15)
    y = -np.log(y)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, feature_name=FeatureMapper.get_names(), params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for i in range(200):
        bst.update()
        if verbose:
            print(i + 1, bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return PerTupleTreeModel(bst)
