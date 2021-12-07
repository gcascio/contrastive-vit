import torch
import torch.nn.functional as F

from torch import nn


class NT_Xent(nn.Module):
    """Algorithm from: https://github.com/google-research/simclr/blob/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3/tf2/objective.py # noqa: E501
    (accessed 13.11.2021) translated to pytorch
    """
    def __init__(self, batch_size, temperature=1.0):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = torch.diag(torch.ones(batch_size)) * 1e9
        self.labels = F.one_hot(
            torch.arange(batch_size),
            num_classes=batch_size * 2,
        )

    def forward(self, z_i, z_j):

        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        logits_ii = torch.mm(z_i, z_i.t()) / self.temperature
        logits_ii = logits_ii - self.mask
        logits_jj = torch.mm(z_j, z_j.t()) / self.temperature
        logits_jj = logits_jj - self.mask
        logits_ij = torch.mm(z_i, z_j.t()) / self.temperature
        logits_ji = torch.mm(z_j, z_i.t()) / self.temperature

        logits_i = torch.cat([logits_ij, logits_ii], 1)
        logits_j = torch.cat([logits_ji, logits_jj], 1)

        loss_i = torch.sum(- self.labels * F.log_softmax(logits_i, -1))
        loss_j = torch.sum(- self.labels * F.log_softmax(logits_j, -1))

        loss = loss_i + loss_j

        loss /= 2 * self.batch_size

        return loss
