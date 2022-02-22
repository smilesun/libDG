############### Configuration file ###############
import math
import torch

start_epoch = 1
num_epochs = 100
batch_size = 32
optim_type = 'Adam'
resize = 32  # image size, for mnist
#resize = 28  # image size, for fashion-mnist

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'MNIST': (0.1307,),
    'mnist': (0.1307,),
    'fashion-mnist': (0.1307,),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.3081,),
    'MNIST': (0.3081,),
    'fashion-mnist': (0.3081,),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

def use_cuda():
    use_cuda = torch.cuda.is_available()
    #import getpass
    #if getpass.getuser() == "sunxd": use_cuda = False
    return use_cuda
    #return False
