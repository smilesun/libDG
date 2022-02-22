import torch
import torch.nn as nn
from torch.nn import functional as F

from libdg.models.a_model_classif import AModelClassif
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na


class ModelDeepAll(AModelClassif):
    def __init__(self, net, list_str_y, list_str_d=None):
        super().__init__(list_str_y, list_str_d)
        self.add_module("net", net)
        # self.net = net

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification
        """
        logit_y = self.net(tensor_x)
        return logit_y

    def infer_y_vpicn(self, tensor):
        with torch.no_grad():
            logit_y = self.net(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class

    def forward(self, tensor_x, tensor_y, tensor_d):
        return self.cal_loss(tensor_x, tensor_y, tensor_d)

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        logit_y = self.net(tensor_x)
        if (tensor_y.shape[-1] == 1) | (len(tensor_y.shape)==1):
            y_target = tensor_y
        else:
            _, y_target = tensor_y.max(dim=1)
        lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        return lc_y


class ModelBNN(ModelDeepAll):
    def __init__(self, net, list_str_y, list_str_d=None,
                 net_builder=None, task=None):
        super().__init__(net, list_str_y, list_str_d)
        self.net_builder = net_builder
        self.task = task
        self.num_repeats = 100

    def set_net(self, net):
        self.net = net

    def _clone_net(self):
        mweights = self.net.state_dict()
        net = self.net_builder(self.task.dim_y, self.task.isize.i_c,
                               self.task.isize.i_h,
                               self.task.isize.i_w)
        net.load_state_dict(mweights)
        return net

    def cal_loss(self, tensor_x, tensor_y, tensor_d):
        logit_y, loss_kl = self.net.probforward(tensor_x)
        if (tensor_y.shape[-1] == 1) | (len(tensor_y.shape) == 1):
            y_target = tensor_y
        else:
            _, y_target = tensor_y.max(dim=1)
        if len(loss_kl.size()) == 0:
            lc_y = F.cross_entropy(logit_y, y_target, reduction="sum")
        else:
            lc_y = F.cross_entropy(logit_y, y_target, reduction="none")
        return lc_y, loss_kl

    def infer_y_vpicn(self, tensor):
        with torch.no_grad():
            #logit_y = self.net.deterministic_forward(tensor)
            tensor = tensor.repeat(self.num_repeats, 1, 1, 1)
            logit_y, _ = self.net(tensor)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
        na_class = get_label_na(ind, self.list_str_y)
        return vec_one_hot, prob, ind, confidence, na_class
