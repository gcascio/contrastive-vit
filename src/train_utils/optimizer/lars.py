import torch
from .optimizer_utils import OptimizerUtils


class LARS(torch.optim.SGD, OptimizerUtils):
    """
    Algorithm from: https://arxiv.org/abs/1708.03888
    (accessed 13.11.2021) translated to pytorch
    """

    def __init__(self, params, lr, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, lars_coefficient=0.001):
        super().__init__(params, lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)
        self.lars_coefficient = lars_coefficient

    def lars(self, weight_decays):
        with torch.no_grad():
            for i, group in enumerate(self.param_groups):
                weight_decay = weight_decays[i] or 0.
                for param in group['params']:
                    if param.grad is None:
                        continue

                    local_lr = 1.
                    param_norm = torch.norm(param.data)
                    grad_norm = torch.norm(param.grad.data)

                    if param_norm > 0 and grad_norm > 0:
                        denom = grad_norm + weight_decay * param_norm
                        local_lr = self.lars_coefficient * param_norm / denom

                    param.grad.data += weight_decay * param.data
                    param.grad.data *= local_lr

    def step(self, *args, **kwargs):
        with self.bypass_param('weight_decay') as cached_weight_decays:
            self.lars(cached_weight_decays)
            super().step(*args, **kwargs)
