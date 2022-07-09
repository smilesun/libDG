import torch
import torch.distributions as dist
from torch import nn
from libdg.compos.zoo_nn import FeatExtractNNBuilderChainNodeGetter
from libdg.compos.net_conv import get_flat_dim


class EncoderConnectLastFeatLayer2Z(nn.Module):
    """
    Connect the last layer of a feature extraction
    neural network to the latent representation
    """
    def __init__(self, z_dim, flag_pretrain,
                 i_c, i_h, i_w, args):
        """__init__.
        :param hidden_size:
        """
        super().__init__()

        net_builder = FeatExtractNNBuilderChainNodeGetter(args)()  # request

        self.net_feat_extract = net_builder.init_business(
            flag_pretrain=flag_pretrain, dim_y=None,
            remove_last_layer=True, args=args)

        size_last_layer_before_z = get_flat_dim(
            self.net_feat_extract, i_c, i_h, i_w)

        self.net_fc_mean = nn.Sequential(
            nn.Linear(size_last_layer_before_z, z_dim))
        self.net_fc_scale = nn.Sequential(
            nn.Linear(size_last_layer_before_z, z_dim),
            nn.Softplus())  # for scale calculation

        torch.nn.init.xavier_uniform_(self.net_fc_mean[0].weight)
        self.net_fc_mean[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.net_fc_scale[0].weight)
        self.net_fc_scale[0].bias.data.zero_()

    def forward(self, x):
        """
        :param x:
        """
        feature = self.net_feat_extract(x)
        zd_q_loc = self.net_fc_mean(feature)
        zd_q_scale = self.net_fc_scale(feature) + 1e-7
        q_zd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = q_zd.rsample()  # Reparameterization trick
        return q_zd, zd_q
