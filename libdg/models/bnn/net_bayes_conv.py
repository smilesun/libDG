import torch.nn as nn
from libdg.models.bnn.layers_local_repar import LayerBConv2d as BBBConv2d
from libdg.models.bnn.layers_linear import BBBLinearFactorial
from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim



class NetBayesConv(nn.Module):
    """
    """
    def __init__(self, outputs, i_channel, i_h, i_w, conv_stride=1, max_pool_stride=2):
        super().__init__()
        conv_net = nn.Sequential(
            BBBConv2d(i_channel, 32, kernel_size=5, stride=conv_stride),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=max_pool_stride),
            BBBConv2d(32, 64, kernel_size=5, stride=conv_stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=max_pool_stride),
        )
        self.flatten = LayerFlat()
        self.hdim = get_flat_dim(conv_net, i_c=i_channel, i_h=i_h, i_w=i_w)
        self.fc1 = BBBLinearFactorial(self.hdim, 1000)
        self.soft5 = nn.Softplus()
        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()
        self.fc3 = BBBLinearFactorial(1000, outputs)
        self.conv_net = conv_net

        layers = [self.fc1, self.soft5,
                  self.fc2, self.soft6, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
                x, _kl, = layer.convprobforward(x)
                kl += _kl

        logits = x
        # print('logits', logits)
        return logits, kl


def test_B3Conv3FC():
    import torch
    img = torch.randn(20, 3, 32, 32)
    img = img.cuda()
    net = BBB3Conv3FC(7, 3, 32, 32)
    net = net.cuda()
    output, kl = net.probforward(img)
    output.shape
