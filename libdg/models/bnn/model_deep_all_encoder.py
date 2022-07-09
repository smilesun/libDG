import torch
import torch.nn as nn

from model.model_interface import AbstractModel
from components.encoderXYD_components import Img2LatentConvStride1BnReluPool
from components.encoderXYD_components import Img2LatentConvStride1BnReluPoolNoDroput
from components.nn_aux import ReluLinearClassifier


class ModelDeepAllImg2LatentConvStride1BnReluPoolOnlyFeat(nn.Module):
    def __init__(self, input_channel, input_h, input_w,
                 stride, out_size, flag_dropout=False):
        """__init__.
        :param input_channel:
        :param input_h:
        :param input_w:
        :param stride:
        """
        super().__init__()
        if flag_dropout:
            self.feat = Img2LatentConvStride1BnReluPool(
                input_channel, input_h, input_w, stride, out_size)
        else:
            self.feat = Img2LatentConvStride1BnReluPoolNoDroput(
                input_channel, input_h, input_w, stride, out_size)

    def forward(self, x):
        """forward.

        :param x:
        """
        hidden = self.feat(x)
        return hidden


class ModelDeepAllImg2LatentConvStride1BnReluPool(
        AbstractModel,
        ModelDeepAllImg2LatentConvStride1BnReluPoolOnlyFeat):
    """Img2LatentConvStride1BnReluPool.
    This component is for dt_hdiva (direct topic inference) as well as for
    deep_all, which is extracting the path of VAE from encoder till classifier
    note in encoder, there is extra layer of hidden to mean and scale, in this
    component, it is replaced with another hidden layer.
    """

    def __init__(self, input_channel, input_h, input_w,
                 stride, out_size, y_dim, flag_dropout=False):
        """__init__.
        :param input_channel:
        :param input_h:
        :param input_w:
        :param stride:
        """
        super().__init__(input_channel, input_h, input_w,
                 stride, out_size, flag_dropout)
        self._y_dim = y_dim

        self.classifier = ReluLinearClassifier(out_size, y_dim)

    def forward(self, x):
        """forward.

        :param x:
        """
        hidden = self.feat(x)
        out = self.classifier(hidden)
        return out

    @property
    def y_dim(self):
        return self._y_dim

    def infer_class(self, x):
        with torch.no_grad():
            out = self.forward(x)
        onehot_out = super().mk_onehot(out)
        return onehot_out


def test_ModelDeepAllImg2LatentConvStride1BnRel():
    model = ModelDeepAllImg2LatentConvStride1BnReluPool(
        input_channel=3, input_h=28, input_w=28, stride=1, out_size=64, y_dim=10)
    model.y_dim
    model.state_dict().keys()
    state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key.replace('feat.', '')] = state_dict.pop(key)
    state_dict.keys()
