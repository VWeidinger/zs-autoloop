from typing import Dict

from ldb_models.training.losses import model_specific, ranking, regression, wrappers


def create_loss(loss_name: str, loss_kwargs: Dict):
    if loss_name == "MSELoss":
        loss = regression.MSELoss(**loss_kwargs)
    elif loss_name == "MSELogLoss":
        loss = regression.MSELogLoss(**loss_kwargs)
    elif loss_name == "QLoss":
        loss = regression.QLoss(**loss_kwargs)
    elif loss_name == "QLogSqLoss":
        loss = regression.QLogSqLoss(**loss_kwargs)
    elif loss_name == "LogQLoss":
        loss = regression.LogQLoss(**loss_kwargs)
    elif loss_name == "Log1pQLoss":
        loss = regression.Log1pQLoss(**loss_kwargs)
    elif loss_name == "Log1pQLossScaled":
        loss = regression.Log1pQLossScaled(**loss_kwargs)
    elif loss_name == "Log1pQLossPenalty":
        loss = regression.Log1pQLossPenalty(**loss_kwargs)
    elif loss_name == "LogHuberQLoss":
        loss = regression.LogHuberQLoss(**loss_kwargs)
    elif loss_name == "Log1pHuberQLoss":
        loss = regression.Log1pHuberQLoss(**loss_kwargs)
    elif loss_name == "LogQLossPenalty":
        loss = regression.LogQLossPenalty(**loss_kwargs)
    elif loss_name == "BiasedLog1pLoss":
        loss = regression.BiasedLog1pLoss(**loss_kwargs)
    elif loss_name == "ExpUELog1pLoss":
        loss = regression.ExpUELog1pLoss(**loss_kwargs)
    elif loss_name == "DaceLoss":
        loss = model_specific.DaceLoss(**loss_kwargs)
    elif loss_name == "QPPLoss":
        loss = model_specific.QPPLoss(**loss_kwargs)
    elif loss_name == "RankLoss":
        loss = ranking.RankLoss(**loss_kwargs)
    elif loss_name == "LambdaLoss":
        loss = ranking.LambdaLoss(**loss_kwargs)
    elif loss_name == "NDCGLoss":
        loss = ranking.NDCGLoss(**loss_kwargs)
    elif loss_name == "RankNetLoss":
        loss = ranking.RankNetLoss(**loss_kwargs)
    elif loss_name == "ListNetLoss":
        loss = ranking.ListNetLoss(**loss_kwargs)
    elif loss_name == "ListMLELoss":
        loss = ranking.ListMLELoss(**loss_kwargs)
    elif loss_name == "BCELoss":
        loss = ranking.BCELoss(**loss_kwargs)
    elif loss_name == "LearnToRankWithRegressionLoss":
        loss = wrappers.LearnToRankWithRegressionLoss(**loss_kwargs)
    else:
        raise ValueError(f"Loss {loss_name} not supported")

    print(f"Using loss {loss_name} with kwargs {loss_kwargs}")
    learn_from_timeout = loss_kwargs.pop(
        "learn_from_timeout", False
    )  # learn from timeout
    timeout_threshold = loss_kwargs.pop("timeout_threshold", None)

    if learn_from_timeout:
        print(f"Using timeout masking with threshold {timeout_threshold}")
        loss = wrappers.LearnFromTimeoutLossWrapper(timeout_threshold, loss)
    return loss
