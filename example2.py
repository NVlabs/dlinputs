
from __future__ import print_function

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

# import tensorboard_logger as logger
from tensorboardX import SummaryWriter

import pdb
from datetime import datetime

import calendar
import time
import numpy as np

USE_CUDA = torch.cuda.is_available()


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--epochs', type=int, default=300, metavar='N')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, default=64, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--log-interval', type=int, default=25, metavar='N')
parser.add_argument('--validate-interval', type=int, default=1, metavar='N')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--devices', type=int, default=1, metavar='N')
args = parser.parse_args("")


net = torchvision.models.resnet18()

if USE_CUDA:
    net.cuda()

    devices = []
    for i in range(args.devices):
        devices.append(i)

    if len(devices)>1:
        net = torch.nn.DataParallel(net, device_ids=devices)
        cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[args.epochs*.25, args.epochs*.5,args.epochs*.75], gamma=0.1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    # Step count for decaying learning rate
    scheduler.step()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    epoch_start_time = calendar.timegm(time.gmtime())

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        curr_mini_loss = loss.data[0]
        curr_mini_size = targets.size(0)

        _, predicted = torch.max(outputs.data, 1)
        curr_mini_correct = predicted.eq(targets.data).cpu().sum()

        train_loss += curr_mini_loss
        total += curr_mini_size
        correct += curr_mini_correct

        if batch_idx%args.log_interval == 0:
            train_loss_avg = train_loss/(batch_idx+1)
            train_acc_avg = 100.*correct/total

            msg = 'Train Loss Avg: %.3f | Train Acc Avg: %.3f%% (%d/%d)' % (train_loss_avg, train_acc_avg, correct, total)
            print(msg)

    epoch_end_time = calendar.timegm(time.gmtime())
    epoch_time_taken = epoch_end_time - epoch_start_time

    print("epoch_time_taken: " + str(epoch_time_taken))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_start_time = calendar.timegm(time.gmtime())

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        curr_mini_loss = loss.data[0]
        curr_mini_size = targets.size(0)
        
        _, predicted = torch.max(outputs.data, 1)

        curr_mini_correct = predicted.eq(targets.data).cpu().sum()

        test_loss += curr_mini_loss
        total += curr_mini_size
        correct += curr_mini_correct

    test_end_time = calendar.timegm(time.gmtime())
    test_time_taken = test_end_time - test_start_time

    print("test_time_taken: " + str(test_time_taken))

    test_loss_avg =  test_loss/(batch_idx+1)
    test_acc_avg = 100.*correct/total

    msg = 'Test Loss Avg: %.3f | Test Acc Avg: %.3f%% (%d/%d)' % (test_loss_avg, test_acc_avg, correct, total)
    print(msg)


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        if epoch==0 or (epoch+1)%args.validate_interval==0:
            test(epoch)

