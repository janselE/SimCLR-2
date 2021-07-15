import torch.nn as nn
import torchvision
import torch

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity

class Attn_SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder1, encoder2, projection_dim, n_features):
        super(Attn_SimCLR, self).__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder1.fc = Identity()
        self.encoder2.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

        self.attn = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=False),
        )

    def forward(self, x_i, x_j, attn=False, mask_type='sigmoid'):
        h_i = self.encoder1(x_i)
        h_j = self.encoder2(x_j)

        if attn:
            mask = self.attn(h_i)
            if mask_type == "softmax":
                mask = torch.softmax(mask, 1)
            if mask_type == "sigmoid":
                mask = torch.sigmoid(mask)
            h_i = h_i * mask
            h_j = h_j * mask

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if attn:
            return h_i, h_j, z_i, z_j, mask

        return h_i, h_j, z_i, z_j, None

