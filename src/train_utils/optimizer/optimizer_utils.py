import torch
from contextlib import contextmanager


class OptimizerUtils(torch.optim.Optimizer):
    @contextmanager
    def bypass_param(self, param_key: str, intermediate_value=0):
        cache = []

        for group in self.param_groups:
            if param_key in group:
                cache.append(group[param_key])
                group[param_key] = intermediate_value
            else:
                cache.append(None)

        try:
            yield cache
        finally:
            for i, group in enumerate(self.param_groups):
                if cache[i] is not None:
                    group[param_key] = cache[i]
