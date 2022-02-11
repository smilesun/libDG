import torch.nn as nn
from libdg.models.bnn.layers_local_repar import LayerBConv2d as BBBConv2d
from libdg.models.bnn.layers_linear import BBBLinearFactorial
from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim
from libdg.models.bnn.Bayesian3Conv3FC import bn_forward


class NetBayes2ConvFC(nn.Module):
    """
    """
    def __init__(self, outputs, i_c, i_h, i_w):
        super().__init__()
        self.conv1 = BBBConv2d(i_c, 32, 5, stride=1, padding=2)
        self.soft1 = nn.BatchNorm2d(32),
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, stride=1, padding=2)
        self.soft2 = nn.BatchNorm2d(32),
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = LayerFlat()

        self.conv_layer = [self.conv1, self.soft1, self.relu1, self.pool1,
                           self.conv2, self.soft2, self.relu2, self.pool2,
                           self.flatten]

        self.hdim, _ = bn_forward(self.conv_layer, i_c=i_c, i_h=i_h, i_w=i_w)
        self.fc1 = BBBLinearFactorial(self.hdim, 1000)
        #self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)  # only for 32*32 image
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.conv3, self.soft3, self.pool3, self.flatten, self.fc1, self.soft5,
                  self.fc2, self.soft6, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        # print('logits', logits)
        return logits, kl


def build_net(dim_y, i_c, i_h, i_w):
    """build_net."""
    net = NetBayes2ConvFC(dim_y, i_c, i_h, i_w)
    return net
