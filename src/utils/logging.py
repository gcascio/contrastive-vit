import os
import time
import torch_xla.core.xla_model as xm


def get_log_base_path(dataset: str, image_size=None, model=None, dir=None):
    date_time = time.strftime('%a_%d_%b_%Y_%H:%M:%S_%Z')
    sub_dir = dataset

    if image_size:
        sub_dir = f'{sub_dir}_{image_size}'

    if model:
        sub_dir = f'{sub_dir}_{model}'

    if dir:
        sub_dir = os.path.join(sub_dir, dir)

    return os.path.join('./runs', sub_dir, date_time)


def log_epoch_run(
    epoch: int,
    total_epochs: int,
    lr: float,
    loss: float,
    accuracy: float,
):
    log = [
        f"Finished training epoch {epoch}/{total_epochs}",
        f"[xla:{xm.get_ordinal()}]",
        f"lr: {lr}",
        f"Loss: {loss}",
        f"Val-Accuracy: {accuracy}",
    ]

    output = f'== {" | ".join(log)} =='
    xm.master_print(output)


def log_step_run(
    current_step: int,
    total_steps: int,
    loss: float,
    tracker: xm.RateTracker,
):
    log = [
        f"[xla:{xm.get_ordinal()}]",
        f"({current_step}/{total_steps})",
        f"loss={loss:.5f}",
        f"rate=({tracker.rate():.2f}|{tracker.global_rate():.2f})",
        f"time={time.asctime()}",
    ]

    output = " ".join(log)

    print(output, flush=True)


def log_run(train_config):
    log = [
        "Run Started",
        f"lr: {train_config.lr}",
    ]

    output = f'\n== {" | ".join(log)} ==\n'
    xm.master_print(output)
