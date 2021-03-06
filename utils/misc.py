'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import shutil
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p',
           'AverageMeter', 'str2bool', 'reduce_list', 'is_powerOfTwo',
           'save_checkpoint', 'print_arguments']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def str2bool(v):
    '''converter for boolean argparser'''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_double_list(li):
    '''check if list is nested'''
    if not li:
        return False
    if any([isinstance(e, list) for e in li]):
        return True
    return False

def reduce_list(li, order=1):
    '''
        reduce the extra bracket
        E.g. [1, [2,3], 4] -> [1, 2, 3, 4]
        E.g. [[1], [[2,3],[]], [4]] -> [[1], [2,3], [], [4]]
    '''
    if order == 1:
        check = lambda e: isinstance(e, list)
    elif order == 2:
        check = is_double_list
    else:
        raise KeyError('order not defined! %i' % order)
    li_ = []
    for l in li:
        if check(l):
            li_.extend(l)
        else:
            li_.append(l)
    return li_

def is_powerOfTwo (x): 
    # check if an integer is power of two
    return (x and (not(x & (x - 1))) ) 

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def print_arguments(args):
    '''print input arguments'''

    print("     -------------------------- dataset -----------------------------------------")
    print("     dataset: %s" % args.dataset)

    print("     --------------------------- training ----------------------------------")
    print("     Epochs: %i" % args.epochs)
    print("     Train batch size: %i" % args.train_batch)
    print("     Test batch size: %i" % args.test_batch)
    print("     Learning rate: %g" % args.lr)
    print("     Momentum: %g" % args.momentum)
    print("     Weight decay: %g" % args.weight_decay)
    print("     Learning rate scheduler: %s" % args.scheduler)  # 'multi-step cosine annealing schedule'
    if args.scheduler in ['step', 'cosine', 'adacosine']:
        print("     Learning rate schedule - milestones: ", args.schedule)
    if args.scheduler in ['step', 'expo', 'adapt']:
        print("     Learning rate decay factor: %g" % args.gamma)
    print("     gpu id: %s" % args.gpu_id)
    print("     num workers: %i" % args.workers)
    print("     hooker: ", args.hooker)
    print("     --------------------------- model ----------------------------------")
    print("     Model: %s" % args.arch)
    print("     depth: %i" % args.depth)
    print("     block: %s" % args.block_name)
    if args.grow:
        if not args.arch in ['resnet', 'preresnet']:
            raise KeyError("model not supported for growing yet.")
        print("     --------------------------- grow ----------------------------------")
        print("     grow mode: %s" % args.mode)
        if args.mode == 'fixed':
            print("     grow milestones: ", args.grow_epoch)
        else:
            print("     max depth: %i" % args.max_depth)
            print("     smoothing scope: %i" % args.window)
            print("     reserved epochs: %i" % args.reserve)
    if args.debug_batch_size:
        print("     -------------------------- debug ------------------------------------")
        print("     debug batches: %i" % args.debug_batch_size)
    print("     ---------------------------------------------------------------------")