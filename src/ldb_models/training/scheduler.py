import inspect

from torch.optim import lr_scheduler as schedulers


def init_scheduler(config, optimizer):
    scheduler = None
    if (
        hasattr(config, "scheduler_class_name")
        and config.scheduler_class_name
        and config.scheduler_class_name != "NoScheduler"
    ):
        scheduler_kwargs_dict = {
            "CosineAnnealingWarmRestarts": {"T_0": 5, "T_mult": 2, "eta_min": 1e-6},
            "ReduceLROnPlateau": {
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
                "threshold": 1e-4,
                "min_lr": 1e-6,
            },
            "CosineAnnealingLR": {"T_max": config.epochs, "eta_min": 1e-6},
            "MultiStepLR": {"milestones": [30, 70], "gamma": 0.3},
        }
        # 5+10+20+40+80 = 155  # T_max for CosineAnnealingLR, assuming 155 epochs in total

        try:
            scheduler_class = schedulers.__dict__[config.scheduler_class_name]
            scheduler_kwargs = scheduler_kwargs_dict.get(
                config.scheduler_class_name, {}
            )
            print(
                f"Using scheduler class: {config.scheduler_class_name} with kwargs: {scheduler_kwargs}",
                flush=True,
            )
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            print(
                f"Initialized scheduler: {config.scheduler_class_name} with kwargs: {scheduler_kwargs}",
                flush=True,
            )
        except KeyError:
            raise ValueError(
                f"Scheduler class '{config.scheduler_class_name}' not found in torch.optim.lr_scheduler"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error initializing scheduler '{config.scheduler_class_name}': {e}"
            ) from e
    else:
        # linear scheduler no change per epoch
        scheduler = schedulers.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    warmup_epochs = config.warmup_epochs if hasattr(config, "warmup_epochs") else 0
    target_epochs = config.epochs + warmup_epochs

    if warmup_epochs > 0:
        print(f"Using warmup scheduler for {warmup_epochs} epochs", flush=True)
        scheduler = schedulers.SequentialLR(
            optimizer,
            schedulers=[
                schedulers.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                scheduler,
            ],
            milestones=[warmup_epochs],
        )
    return scheduler, target_epochs


def step_scheduler(scheduler=None, epoch: int = 0, avg_loss: float = None):
    """
    Steps the learning rate scheduler with the appropriate arguments.
    """
    if scheduler is None:
        return
    step_args = inspect.signature(scheduler.step).parameters
    if "epoch" in step_args and "metrics" in step_args:
        scheduler.step(epoch=epoch, metrics=avg_loss)
    elif "epoch" in step_args:
        scheduler.step(epoch=epoch)
    elif "metrics" in step_args:
        scheduler.step(metrics=avg_loss)
    else:
        scheduler.step()
