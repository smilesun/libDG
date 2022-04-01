import torch.nn as nn
from torchvision import models as torchvisionmodels

from libdg.compos.nn import LayerId


class AlexNetBase(nn.Module):
    """
    AlexNet(
    (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
    (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=7, bias=True)
    )
    )
    -
    """
    def __init__(self, flag_pretrain):
        super().__init__()
        self.net_torch_alex = torchvisionmodels.alexnet(pretrained=flag_pretrain)

    def forward(self, tensor):
        """
        delegate forward operation to self.net_torch_alex
        """
        out = self.net_torch_alex(tensor)
        return out

    def show(self):
        """
        print out which layer will be optimized
        """
        for name, param in self.net_torch_alex.named_parameters():
            if param.requires_grad:
                print("layers that will be optimized: \t", name)


class Alex4DeepAll(AlexNetBase):
    """
    change the last layer output of AlexNet to the dimension of the
    """
    def __init__(self, flag_pretrain, dim_y):
        super().__init__(flag_pretrain)
        if self.net_torch_alex.classifier[6].out_features != dim_y:
            print("original alex net out dim", self.net_torch_alex.classifier[6].out_features)
            num_ftrs = self.net_torch_alex.classifier[6].in_features
            self.net_torch_alex.classifier[6] = nn.Linear(num_ftrs, dim_y)
            print("re-initialized to ", dim_y)


class AlexNetNoLastLayer(AlexNetBase):
    """
    Change the last layer of AlexNet with identity layer,
    the classifier from VAE can then have the same layer depth as deep_all model
    so it is fair for comparison
    """
    def __init__(self, flag_pretrain):
        super().__init__(flag_pretrain)
        self.net_torch_alex.classifier[6] = LayerId()

    def extract_feat(self, x):
        return self.net_torch_alex(x)


def test_AlexNetConvClassif():
    import torch
    model = AlexNetNoLastLayer(True)
    x = torch.rand(20, 3, 224, 224)
    x = torch.clamp(x, 0, 1)
    x.require_grad = False
    torch.all(x > 0)
    res = model(x)
    res.shape
