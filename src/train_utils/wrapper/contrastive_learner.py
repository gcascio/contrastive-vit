from torch import nn
from train_utils.losses.nt_xent import NT_Xent


class SimCLR(nn.Module):
    def __init__(
        self,
        model,
        model_output_dim,
        batch_size,
        project_dim=128,
        temperature=0.1,
    ):
        super().__init__()
        self.model = model

        self.projection = nn.Sequential(
            nn.Linear(model_output_dim, model_output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(model_output_dim, project_dim, bias=False)
        )

        self.loss = NT_Xent(batch_size, temperature=temperature)

    def forward(self, x):
        x_i, x_j = x

        h_i = self.model(x_i)
        h_j = self.model(x_j)

        z_i = self.projection(h_i)
        z_j = self.projection(h_j)

        return self.loss(z_i, z_j)
