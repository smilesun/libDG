import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair  # repeat argument like kernel size
from libdg.models.bnn.BBBdistributions import Normal, Normalout, distribution_selector


class ConvNdBase(nn.Module):
    """
    Bayesian convolutional layer with a distribution over the weights and biases
    The base class allows N-dim convolution although we normally use only 2d convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=2, dilation=1, groups=1,
                 p_logvar_init=-3,  # FIXME: are performances sensitive to this as well?
                 p_pi=1.0,
                 q_logvar_init=1):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # initialize log variance of p and q
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # approximate posterior weights...
        # let a_{mi} be the $i$th component of
        # activation from previous layer
        # for instance $m$, $u_{i,j}$ be the neural
        # network weight connect input $i$
        # to output $j$
        # E(b_{mj})=\sum_i a_{mi}u_{i,j}
        # torch.F.linear(input, weight) requires weight to be defined
        # in the following way
        self.qw_mean = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.qw_logvar = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)

        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_channels))
        # self.qb_logvar = Parameter(torch.Tensor(out_channels))

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        # (does not have any trainable parameters so we use fixed normal or fixed mixture normal distributions)
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pw = distribution_selector(mu=(0.0, -1, +1), logvar=(-1, -8, -3), pi=[0.3, 0.5, 0.2])
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all parameters
        self.reset_parameters()

    def reset_parameters(self):
        # uniforme initialization to weight
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)


def test_BNdConv():
    net = ConvNdBase(4, 5, (3, 1), 1)
