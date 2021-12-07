from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLRWithWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, warmup_steps=None, eta_min=0,
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(CosineAnnealingLRWithWarmup, self).__init__(
            optimizer,
            T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def get_lr(self):
        learning_rates = []

        if self.warmup_steps and self.warmup_steps > self._step_count:
            scale = min(1., self._step_count / self.warmup_steps)
            learning_rates = [lr * scale for lr in self.base_lrs]
        else:
            learning_rates = super().get_lr()

        return learning_rates
