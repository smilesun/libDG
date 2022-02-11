import torch.nn as nn
from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim
import torch.nn.functional as F
import numpy as np

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class F3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1206.0580.pdf
    """
    def __init__(self, outputs, i_c, i_h, i_w):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(i_c, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.h_dim = get_flat_dim(self.features, i_c, i_h, i_w)
        self.classifier = nn.Sequential(
            FlattenLayer(self.h_dim),
            nn.Linear(self.h_dim, 1000),
            nn.Softplus(),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_net(dim_y, i_c, i_h, i_w):
    """build_net."""
    net = F3Conv3FC(dim_y, i_c, i_h, i_w)
    return net
