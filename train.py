from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import json
import os
import torch
import sys
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.backends.cudnn as cudnn
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse
import time
import math
from networks import CNN, MLPNet
from tqdm.notebook import tqdm
import pandas as pd


parser = argparse.ArgumentParser('argument for training')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=8,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs')
# optimization
parser.add_argument('--lr', type=float, default=0.02,
                    help='learning rate')

# model dataset
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--data_path', default='data/', type=str, help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'path'], help='dataset')
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')

# other setting
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--ckpt', type=str, default='prob_ckpt_epoch_1000.pth',
                    help='path to pre-trained model')
parser.add_argument('--number', type=int, default=500,
                    help='fine number of training data in class 0')





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_model(args):
    """
    TODO
    """
    if args.dataset == 'cifar10':
        #model = CNN(n_outputs=10)
        model = torchvision.models.resnet18(pretrained=False)
        model = model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])
    elif args.dataset == 'cifar100':
        model = CNN(n_outputs=100)
    else:
        model = MLPNet()

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_dataset(args, fine_tune=False):
    """
    TODO
    """

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=train_transform,
                                noise_type=args.noise_type,
                                noise_rate=args.r,
                                fine_tune=fine_tune,
                                number= args.number
                                )

        test_dataset = CIFAR10(root='./data/',
                               download=True,
                               train=False,
                               target_transform=val_transform,
                               noise_type=args.noise_type,
                               noise_rate=args.r,
                               fine_tune=fine_tune
                               )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=train_transform,
                                noise_type=args.noise_type,
                                noise_rate=args.r
                                )

        test_dataset = CIFAR10(root='./data/',
                               download=True,
                               train=False,
                               target_transform=val_transform,
                               noise_type=args.noise_type,
                               noise_rate=args.r
                               )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)


    else:
        train_dataset = MNIST(root='./data/',
                              download=True,
                              train=True,
                              transform=transforms.ToTensor(),
                              noise_type=args.noise_type,
                              noise_rate=args.r
                              )

        test_dataset = MNIST(root='./data/',
                             download=True,
                             train=False,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.r
                             )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

    return train_loader, test_loader



def train_step(net1, loader, epoch, args, opt_A, device):
    net1.train()
    #for index, (img_s, img_t, target_s, target_t, real_target_s, real_target_t) in enumerate(loader):
    for index, (img,target) in enumerate(loader):
        img, target = img.to(device), target.to(device)
        opt_A.zero_grad()
        #print(img.shape)
        output = net1(img.to(torch.float32))
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt_A.step()
    return loss, net1

       


def test_step(net, loader, epoch, device):
    net.eval()
    acc = AverageMeter()
    for idx, (inputs, targets) in enumerate(loader):

        bsz = targets.shape[0]

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = net(inputs.to(torch.float32))
            acc1 = accuracy(output, targets, topk=(1,))


        acc.update(acc1[0].item(), bsz)

    return acc.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args):
    """
    Don't forget CUDA version
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataset(args)
    ####
    train_loader_F, _ = get_dataset(args, fine_tune = True)
    #get a 20 epoch warm up model
    modelA = get_model(args)
    #modelB = get_model(args)
    opt_A = optim.Adam(modelA.parameters(), lr=args.lr, weight_decay=5e-4)
    #opt_B = optim.SGD(modelB.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #warm_up(train_loader, modelA, modelB, args, opt_A, opt_B, device)


    results = {'train_loss1': [], 'test_acc@1': []}
    #start training
    for epoch in tqdm(range(1, args.epochs+1)):
        lr = args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in opt_A.param_groups:
            param_group['lr'] = lr
       

        #acc2 = test_step(modelB, test_loader, epoch, device)

        loss1, modelA = train_step(modelA, train_loader, epoch, args, opt_A, device)

        #### model freeze representation layers
        modelA = model_freeze(modelA)
        #### fine tune
        loss2, modelA = train_step(modelA, train_loader_F, epoch, args, opt_A, device)
        #### de-freeze model
        modelA = model_defreeze(modelA)

        acc1 = test_step(modelA, test_loader, epoch, device)
        results['train_loss1'].append(loss1.detach().numpy())
        #results['train_loss2'].append(loss2)
        results['test_acc@1'].append(acc1)
        #results['test_acc@2'].append(acc2)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('./' + '/log.csv', index_label='epoch')


def model_freeze(model):
    return model


def model_defreeze(model):
    return model



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
