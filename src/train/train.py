import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from train_utils.datasets.data_utils import get_data_loader
from train_utils.scheduler.get_scheduler import get_scheduler
from train_utils.optimizer.lars import LARS
from config.config import TrainConfig
from utils.logging import log_epoch_run
from utils.summary_writer import SummaryWriter

from .train_epoch import train_epoch
from .evaluate import evaluate


def train(
    model,
    config: TrainConfig,
    train_dataset,
    test_dataset,
    writer: SummaryWriter = None,
    model_returns_loss=False,
    model_config=None,
):
    train_loader, test_loader = get_data_loader(
        train_dataset,
        test_dataset,
        config.batch_size,
        test_set_batch_size=config.test_set_batch_size,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        subset=config.subset,
    )

    device = xm.xla_device()
    model = model.to(device)

    criterion = (lambda x, _: x) if model_returns_loss else CrossEntropyLoss()
    get_optimizer = LARS if model_returns_loss else Adam

    optimizer = get_optimizer(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = get_scheduler(
        config.scheduler,
        optimizer,
        config,
        max_iterations=config.epochs * len(train_loader)
    )

    best_accuracy = 0.
    best_result = None

    for epoch in range(1, config.epochs + 1):
        mp_train_loader = pl.MpDeviceLoader(train_loader, device)
        loss = train_epoch(
            model,
            mp_train_loader,
            optimizer,
            config,
            criterion,
            scheduler,
        )

        mp_test_loader = pl.MpDeviceLoader(test_loader, device)
        accuracy = 0.
        if not model_returns_loss:
            accuracy = evaluate(
                model,
                mp_test_loader,
            )

        log_epoch_run(
            epoch=epoch,
            total_epochs=config.epochs,
            lr=config.lr,
            loss=loss,
            accuracy=accuracy,
        )

        if writer:
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            writer.add_scalar("Loss/train", loss, epoch)

            if scheduler:
                writer.add_scalar(
                    "Learningrate/train",
                    scheduler.get_last_lr()[0],
                    epoch,
                )

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_result = {
                    'train_config': config.to_dict(),
                    'model_config':
                        model_config.to_dict() if model_config else None,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }

    return best_accuracy, best_result
