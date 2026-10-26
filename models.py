import torch
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    """Generator architecture matching the supplied trained checkpoint."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM + 1, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, temp):
        x = torch.cat([z, temp], dim=1)
        x = self.fc(x).view(-1, 128, 4, 4)
        return self.net(x)
