import torch.nn as nn
from libdg.models.bnn.layers_local_repar import LayerBConv2d as BBBConv2d
from libdg.models.bnn.layers_linear import BBBLinearFactorial
from libdg.compos.nn import LayerFlat
from libdg.compos.net_conv import get_flat_dim



class NetBayes3ConvFC(nn.Module):
    """
    """
    def __init__(self, outputs, inputs, h, w):
        super().__init__()
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
        #self.hdim = get_flat_dim(self.conv_net, i_c=input, i_h=h, i_w=w)
        #self.fc1 = BBBLinearFactorial(self.hdim, 1000)
        self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)  # only for 32*32 image
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


def test_B3Conv3FC():
    import torch
    img = torch.randn(20, 3, 32, 32)
    img = img.cuda()
    net = BBB3Conv3FC(7, 3, 32, 32)
    net = net.cuda()
    output, kl = net.probforward(img)
    output.shape
