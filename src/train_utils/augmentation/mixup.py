import torch
import numpy.random as random


def mixup(x, y, criterion, alpha=0.2):
    if alpha <= 0:
        x, y, None, 1

    lam = random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    def loss_fn(output, _):
        return lam * criterion(output, y_a) +\
                (1 - lam) * criterion(output, y_b)

    return mixed_x, loss_fn
