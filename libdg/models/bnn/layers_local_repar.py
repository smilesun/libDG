import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair  # repeat argument like kernel size

from libdg.models.bnn.BBBdistributions import Normal, Normalout, distribution_selector
from libdg.models.bnn.layers_base import ConvNdBase


class LayerBConv2d(ConvNdBase):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0, dilation=1, groups=1):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups)

    def forward(self, input):
        raise NotImplementedError()

    def convprobforward(self, input):
        """
        Convolutional probabilistic forwarding method.
        :param input: data tensor
        :return: output, KL-divergence
        """

        # local reparameterization trick for convolutional layer
        output_mean = F.conv2d(input=input.float(),
                                weight=self.qw_mean,  # set at parent class
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups)

        output_std = torch.sqrt(1e-8 + F.conv2d(input=input.float().pow(2),  #\sum_i \sigma_{ij}^2*a_{mi}^2
                                                # receptive field also get powered
                                                # let m be instance index
                                                # j be output index
                                                #  \sum_i \sigma_{ij}^2*a_{mi}^2
                                                weight=torch.exp(self.log_alpha)*self.qw_mean.pow(2),
                                                # mean=exp(u+\sigma^2), var = exp(2u+\sigma^2)[exp(sigma^2)-1]=mean^2*alpha
                                                stride=self.stride,
                                                padding=self.padding,
                                                dilation=self.dilation,
                                                groups=self.groups))
        epsilon = torch.randn(output_mean.size())
        epsilon = epsilon.to(output_mean.device)
        output = output_mean + output_std * epsilon

        w_sample = self.qw.sample()

        # KL divergence
        qw_logpdf = self.qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl


def test_Conv2dB():
    img = torch.randn(20, 3, 28, 28)
    img = img.cuda()
    net = LayerBConv2d(3, 5, 3)
    net = net.cuda()
    output, kl = net.convprobforward(img)
