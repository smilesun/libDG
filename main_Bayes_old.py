import os
import sys
import time
import argparse
import datetime
import math
import pickle
import numpy as np


import torchvision
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import Bayesian_config as cf

from libdg.models.model_deep_all import ModelBNN
from libdg.models.bnn.B_net_small import NetBayes2ConvFC
from libdg.models.bnn.BBBlayers import GaussianVariationalInference
from libdg.tasks.task_mnist_color import NodeTaskMNISTColor4
from libdg.arg_parser import mk_parser_main
parser = mk_parser_main()
args = parser.parse_args()
args.bs = 32
task = NodeTaskMNISTColor4()
args.te_d=0

task.init_business(args)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
#parser.add_argument('--depth', default=28, type=int, help='depth of model')
#parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
# use_cuda = torch.cuda.is_available()
use_cuda = cf.use_cuda()
if use_cuda is True:
    torch.cuda.set_device(0)
best_acc = 0
resize = 32
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

transform_train = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


if (args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 3

elif (args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    outputs = 100
    inputs = 3

elif (args.dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 1

elif (args.dataset == 'fashion-mnist'):
    print("| Preparing fashion-MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 1




trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
#trainloader = task.loader_tr
#testloader = task.loader_te



# Return network & file name
def getNetwork(args):
    net = NetBayes2ConvFC(10, 3, 32, 32)
    #net = NetBayes2ConvFC(10, 3, 28, 28)
    file_name = '3Conv3FC-'
    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)

if use_cuda:
    net.cuda()
















vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

logfile = os.path.join('diagnostics_Bayes{}_{}.txt'.format(args.net_type, args.dataset))

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(trainset) / batch_size)
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), weight_decay=args.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets, *_) in enumerate(trainloader):
        # repeat samples for
        inputs = 3
        inputs_value = inputs_value.repeat(1, 3, 1, 1)
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        x = inputs_value.repeat(args.num_samples, 1, 1, 1)
        print("\nx.shape:" + str(x.shape))
        # _, targets = targets.max(dim=1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0
        # Forward Propagation
        x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)
        #print(outputs.shape)
        loss = vi(outputs, y, kl, beta)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data, (100*correct.to(dtype=torch.float)/float(total))/args.num_samples))
        sys.stdout.flush()

    #diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': (100*correct.to(dtype=torch.float)/float(total))/args.num_samples}
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': (100*correct.to(dtype=torch.float)/float(total))/args.num_samples}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf=[]
    m = math.ceil(len(testset) / batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        x = inputs_value.repeat(args.num_samples, 1, 1, 1)
        _, targets = targets.max(dim=1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        loss = vi(outputs, y, kl, beta)

        #test_loss += loss.data[0]
        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        preds = F.softmax(outputs, dim=1)
        results = torch.topk(preds.cpu().data, k=1, dim=1)
        conf.append(results[0][0].item())
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    p_hat = np.array(conf)
    confidence_mean = np.mean(p_hat, axis=0)
    confidence_var = np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    acc =(100*correct.to(dtype=torch.float)/float(total))/args.num_samples
    #print('\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%' %(epoch, loss.data[0], acc))
    print('\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%' %(epoch, loss.data, acc))
    #test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': acc}
    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data, 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

















