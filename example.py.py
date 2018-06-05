
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

best_prec1 = 0
def main():
    model = models.resnet18()
    
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch    model = models.resnet18()
    
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
'''Train CIFAR10 with PyTorch.'''
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=64, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--epochs', type=int, default=340, metavar='N')
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=25, metavar='N')
parser.add_argument('--save-interval', type=int, default=50, metavar='N')
parser.add_argument('--validate-interval', type=int, default=1, metavar='N')
parser.add_argument('--optimizer', type=str, default='momentum_sgd')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model-import', type=str, required=True)
parser.add_argument('--lr-decay', type=float, default=0.1)
parser.add_argument('--lr-decay-rate', type=float, default=100)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--no-tensorboard', action='store_true', default=False)
parser.add_argument('--devices', type=int, default=2, metavar='N')
parser.add_argument('--cifar', type=int, default=10, metavar='N')
parser.add_argument('--l', type=int, default=100, metavar='N')
parser.add_argument('--growth-rate', type=int, default=12, metavar='N')
parser.add_argument('--cs_planes', type=int, default=10, metavar='N')
parser.add_argument('--top', type=float, default=0.0, metavar='N')
parser.add_argument('--print-model-only', action='store_true', default=False)
args = parser.parse_args()

model = __import__(args.model_import)   

USE_CUDA = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('Preparing data...')

if args.cifar==100:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, drop_last=True)

else: 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, drop_last=True)
# Model
if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + 'model_{}'.format(args.model))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

else:
    print('Building model...')
    if "memnet" in args.model_import:
        net = model.MemNet(args.cifar, 
            l=args.l, 
            growth_rate=args.growth_rate, 
            cs_planes=args.cs_planes, 
            top=args.top)
    else:
        net = model.MemNet(args.cifar)

    if args.print_model_only:
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("#Params: " + str(params))
        print(net)        
        sys.exit()

if USE_CUDA:
    net.cuda()

    devices = []
    for i in range(args.devices):
        devices.append(i)

    net = torch.nn.DataParallel(net, device_ids=devices)

    # Enable benchmark mode, where CUDA finds best algorithm for h/w
    # Works well if input size doesn't change much
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[args.epochs*.25, args.epochs*.5,args.epochs*.75], gamma=0.1)

def adjust_lr(new_lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    # # After the first 10 epochs, set LR back to 0.1
    # if epoch==10:
    #     adjust_lr(0.1, optimizer)
    #     print("Set to new learning rate " + str(optimizer.param_groups[0]['lr']))
    # Step count for decaying learning rate
    scheduler.step()
    if not args.no_tensorboard:
        tb_logger.add_scalar('misc/learning_rate', optimizer.param_groups[0]['lr'], epoch)

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
            if not args.no_tensorboard:
                tb_logger.add_text('train', msg, epoch)

            if not args.no_tensorboard:
                tb_logger.add_scalar('train/train_loss', train_loss_avg, epoch)
                tb_logger.add_scalar('train/train_accuracy', train_acc_avg, epoch)
                tb_logger.add_scalar('train/train_error', 100-train_acc_avg, epoch)

    epoch_end_time = calendar.timegm(time.gmtime())
    epoch_time_taken = epoch_end_time - epoch_start_time

    print("epoch_time_taken: " + str(epoch_time_taken))

    if not args.no_tensorboard:
        tb_logger.add_scalar('train/epoch_time_taken', epoch_time_taken, epoch)

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
    if not args.no_tensorboard:
        tb_logger.add_text('test', msg, epoch)

    if not args.no_tensorboard:
        tb_logger.add_scalar('test/test_loss', test_loss_avg, epoch)
        tb_logger.add_scalar('test/test_accuracy', test_acc_avg, epoch)
        tb_logger.add_scalar('test/test_error', 100-test_acc_avg, epoch)
        tb_logger.add_scalar('test/time_taken', test_time_taken, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if (epoch+1)%args.save_interval == 0:
        print('Saving...')
        tb_logger.add_text('misc', "Saving", epoch)

        state = {
            'acc': acc,
            'epoch': epoch,
            'net': net.module if USE_CUDA else net,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.model_import+'_{}'.format(args.model))
        if acc>best_acc:
            best_acc = acc

tb_logger = None
if not args.no_tensorboard:
    if "memnet" in args.model_import:
        log_dir = os.path.join('log',
                    '{}'.format(args.model_import), 
                     '{}'.format(args.model),
                     'cifar_{}'.format(args.cifar), 
                     'lr_{}'.format(args.lr),
                     'L_{}'.format(args.l),
                     'GR_{}'.format(args.growth_rate), 
                     'CS_{}'.format(args.cs_planes),
                     'TOP_{}'.format(args.top), 
                     datetime.now().isoformat())
    else:
        log_dir = os.path.join('log',
            '{}'.format(args.model_import), 
             '{}'.format(args.model), 
             'cifar_{}'.format(args.cifar),
             'lr_{}'.format(args.lr), 
             datetime.now().isoformat())

    tb_logger = SummaryWriter(log_dir=log_dir)

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    if epoch==0 or (epoch+1)%args.validate_interval==0:
        test(epoch)

tb_logger.close()
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
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
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
if __name__ == '__main__':
    main()


from dlinputs import tarrecords
from dlinputs import gopen
from dlinputs import paths


source = gopen.sharditerator("test/testdata/cifar10-train-@000001.tar")


source = gopen.sharditerator("gs://lpr-demo/cifar10-train-@000001.tgz")


source = gopen.sharditerator("test/testdata/imagenet-@000001.tgz")


next(source)

