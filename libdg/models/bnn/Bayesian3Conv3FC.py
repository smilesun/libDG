import torch
import torch.nn as nn
from libdg.models.bnn.layers_local_repar import LayerBConv2d as BBBConv2d
from libdg.models.bnn.layers_linear import BBBLinearFactorial
from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim


def bn_forward(list_layers, i_c, i_h, i_w, bs=5):
    """
    The forward sweep of Bayesian Neural Network involves layer wise
    KL divergence calculation, which can not be put into a sequence
    model in pytorch
    :param list_layers: python list with ordered layers
    """
    x = torch.randn(bs, i_c, i_h, i_w)
    kl = 0
    for layer in list_layers:
        if hasattr(layer, 'convprobforward') and \
                callable(layer.convprobforward):
            x, _kl, = layer.convprobforward(x)
            kl += _kl

        elif hasattr(layer, 'fcprobforward') and \
            callable(layer.fcprobforward):
            x, _kl, = layer.fcprobforward(x)
            kl += _kl
        else:
            x = layer(x)
    return x, kl


class BBB3Conv3FC(nn.Module):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, h, w):
        """__init__.

        :param outputs:
        :param inputs:
        :param h:
        :param w:
        """
        super(BBB3Conv3FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 32, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, stride=1, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, stride=1, padding=1)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = LayerFlat()
        self.conv_layers = [self.conv1, self.soft1, self.pool1,
                            self.conv2, self.soft2, self.pool2,
                            self.conv3, self.soft3, self.pool3,
                            self.flatten]  # flatten layer must be here since in runtime we do not know the batchsize!
        x, _ = bn_forward(self.conv_layers, inputs, h, w)
        self.h_dim = x.shape[1]
        #self.hdim = get_flat_dim(self.conv_net, i_c=input, i_h=h, i_w=w)
        self.fc1 = BBBLinearFactorial(self.h_dim, 1000)
        # self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)  # only for 32*32 image
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)
        len1 = len(self.conv_layers)

        self.conv_layers.extend([self.fc1, self.soft5,
                                self.fc2, self.soft6, self.fc3])
        assert len(self.conv_layers) > len1

        self.layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        """forward.

        :param x:
        """
        logits = self.probforward(x)
        return logits

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        xx = x
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                xx, _kl, = layer.convprobforward(xx)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                xx, _kl, = layer.fcprobforward(xx)
                kl += _kl
            else:
                xx = layer(xx)
        logits = xx
        # print('logits', logits)
        return logits, kl


def test_B3Conv3FC():
    """test_B3Conv3FC."""
    import torch
    img = torch.randn(20, 3, 32, 32)
    img = img.cuda()
    net = BBB3Conv3FC(7, 3, 32, 32)
    net = net.cuda()
    output, kl = net.probforward(img)
    output.shape

def build_net(dim_y, i_c, i_h, i_w):
    """build_net."""
    net = BBB3Conv3FC(dim_y, i_c, i_h, i_w)
    return net
