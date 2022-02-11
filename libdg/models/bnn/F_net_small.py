import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim
from libdg.compos.net_conv import mk_conv_bn_relu_pool

class FNetSmall(nn.Module):
    """
    """
    def __init__(self, outputs, i_c, i_h, i_w):
        super().__init__()
        self.features = mk_conv_bn_relu_pool(i_c)
        self.h_dim = get_flat_dim(self.features, i_c, i_h, i_w)
        self.classifier = nn.Sequential(
            # FlattenLayer(self.h_dim),
            LayerFlat(),
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
    net = FNetSmall(dim_y, i_c, i_h, i_w)
    return net
