import torch_xla.core.xla_model as xm

from utils.logging import log_step_run
from train_utils.augmentation.mixup import mixup


def train_epoch(
    model,
    loader,
    optimizer,
    config,
    criterion,
    scheduler=None,
):
    loss = None
    tracker = xm.RateTracker()
    model.train()
    loader_len = loader.__len__()
    loss_fn = criterion

    tracker.add(config.batch_size)

    for i, (data, target) in enumerate(loader):
        loss = None
        optimizer.zero_grad()

        if config.alpha_mixup and config.alpha_mixup > 0:
            data, loss_fn = mixup(
                data,
                target,
                criterion,
                alpha=config.alpha_mixup,
            )

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()

        xm.optimizer_step(optimizer)

        if i % config.log_steps == 0:
            log_step_run(i, loader_len, loss.item(), tracker)

        if scheduler:
            scheduler.step()

    return loss.item()
