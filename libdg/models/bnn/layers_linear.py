import torch
from torch import nn
class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # Approximate posterior weights...
        self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_features))
        # self.qb_logvar = Parameter(torch.Tensor(out_features))

        # ...and output...
        self.fc_qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.fc_qw_std = Parameter(torch.Tensor(out_features, in_features))

        # ...as normal distributions
        self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        # self.qb = Normal(mu=self.qb_mean, logvar=self.qb_logvar)
        self.fc_qw = Normalout(mu=self.fc_qw_mean, std=self.fc_qw_std)

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # self.qb_mean.data.uniform_(-stdv, stdv)
        # self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.fc_qw_mean.data.uniform_(-stdv, stdv)
        self.fc_qw_std.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward(self, input):
        raise NotImplementedError()

    def fcprobforward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """

        fc_qw_mean = F.linear(input=input, weight=self.qw_mean)
        fc_qw_si = torch.sqrt(1e-8 + F.linear(input=input.pow(2), weight=torch.exp(self.log_alpha)*self.qw_mean.pow(2)))

        if cuda:
            fc_qw_mean.cuda()
            fc_qw_si.cuda()

        # sample from output
        if cuda:
            output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size())).cuda()
        else:
            output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size()))

        if cuda:
            output.cuda()

        w_sample = self.fc_qw.sample()

        # KL divergence
        qw_logpdf = self.fc_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
