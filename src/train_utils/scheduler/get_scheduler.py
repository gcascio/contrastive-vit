import sys
import torch_xla.core.xla_model as xm
from config.config import TrainConfig
from .CosineAnnealingLRWithWarmup import CosineAnnealingLRWithWarmup


def get_scheduler(
    scheduler_key: str,
    optimizer,
    config: TrainConfig,
    max_iterations: int = sys.maxsize,
):
    init_scheduler = ({
        'cosine_with_warmup': lambda: CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=max_iterations,
            warmup_steps=(config.warmup_steps or 10000) // xm.xrt_world_size(),
        )
    }).get(scheduler_key)

    if not init_scheduler:
        return None

    scheduler = init_scheduler()
    return scheduler
