'''
Adapted from Copyright (c) Wei YANG, 2017
'''

from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from pipeline import get_loaders, train, test
from utils import LipHooker, ModelArch, TolTrigger
from utils import str2bool, save_checkpoint, Logger, mkdir_p, print_arguments
from utils import scheduler as schedulers

torch.autograd.set_detect_anomaly(True)

scheduler_names = sorted(name for name in schedulers.__dict__
    if name.islower() and not name.startswith("__")
    and callable(schedulers.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 + Imagenet Training')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Optimization
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--scheduler', type=str, default='constant', choices=scheduler_names, help='scheduler type: constant, step, cosine, adacosine')
parser.add_argument('--schedule', type=int, nargs='*', default=[81, 122], help='Decrease learning rate at these epochs. Required if using step scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='lr decaying rate, required if using step scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

# Model
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet') 
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock', help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')

# Grow
parser.add_argument('--grow', type=str2bool, const=True, default=False, nargs='?', help='Time to grow up!')
parser.add_argument('--mode', type=str, choices=['adapt', 'fixed'], default='adapt', help='The growing mode: adaptive to errs, or fixed at some epochs')
parser.add_argument('--grow-epoch', type=str, nargs='*', default=['60', '110'], help='Duplicate the model at these epochs. Required if mode = fixed.')
parser.add_argument('--max-depth', type=int, default=74, help='Max model depth. Required if mode = adapt.')
parser.add_argument('--window', type=int, default=3, help='Smooth scope of truncated err estimation. Required if mode = adapt.')
parser.add_argument('--threshold', type=float, default=1.4, help='Err trigger threshold for growing.  Required if mode = adapt.')
parser.add_argument('--reserve', type=int, default=30, help='Reserved epochs for final model')
parser.add_argument('--hooker', type=str, choices=['None', 'Lip'], default='Lip', help='Hooker on model to output some info')

# others
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--debug-batch-size', type=int, default=0, help='number of training batches for quick check. default: 0 - no debug')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='7', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
print_arguments(args)

# arguments post process
if args.dataset.lower().startswith('cifar'):
    import models.cifar as models
elif 'imagenet' in args.dataset.lower():
    import models.imagenet as models
else:
    raise KeyError(args.dataset)

if args.hooker == 'None':
    args.hooker = None

if args.mode == 'fixed':
    assert all([s.isdigit() for s in args.grow_epoch]), 'integer grow epochs required for fixed grow'
    args.grow_epoch = [int(s) for s in args.grow_epoch]

if args.mode == 'adapt':
    assert args.max_depth, 'max depth required for adaptive grow'

if args.scheduler == 'adacosine':
    assert(not args.schedule), 'no need to set schedule for adaptive scheduler'

assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'], args.dataset


def main():

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    # allow parallel training
    def to_parallel(model):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            return torch.nn.DataParallel(model)
        return model

    # make output dir
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Load dataset
    trainloader, valloader, testloader, num_classes = get_loaders(dataset=args.dataset,
                                                                  download=False,
                                                                  train_batch=args.train_batch,
                                                                  test_batch=args.test_batch,
                                                                  n_workers=args.workers, 
                                                                  data_dir='./data')


    # Init model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnet') or args.arch.startswith('preresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))

    to_parallel(model).to(device)

    # Set model Lipschitz hooker
    print("==> set Lipschitz hooker ")
    hooker = LipHooker(args.arch, args.checkpoint, device=device)
    hooker.hook(model)

    # Set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Set learning rate scheduler
    print("==> creating scheduler '{}'".format(args.scheduler))
    if args.scheduler.startswith('constant'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, dpath=args.checkpoint)
    elif args.scheduler.startswith('step'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, gamma=args.gamma, dpath=args.checkpoint)
    elif args.scheduler.startswith('cosine'): 
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, epochs=args.epochs, dpath=args.checkpoint)
    elif args.scheduler.startswith('adacosine'): 
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, epochs=args.epochs, dpath=args.checkpoint)
    else:
        raise KeyError(args.scheduler)

    # Set info logger
    title = args.dataset + '-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Time Elapsed', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Acc.', 'Valid Acc.', 'Test Acc.'])

    # ---------- grow -----------
    # Set model architecture tracker for grow
    modelArch=None
    if args.grow:
        modelArch = ModelArch(args.arch, model, args.epochs, args.depth, max_depth=args.max_depth, dpath=args.checkpoint, dataset=args.dataset)

    # Set trigger for grow
    if args.grow and args.mode == 'adapt':
        trigger = TolTrigger(tolerance=args.threshold, window=args.window, reserve=args.reserve, epochs=args.epochs, modelArch=modelArch)

    # Training start
    print("==> training start - epochs: %i" % args.epochs)
    time_start = time.time()
    best_val_acc = 0  # best test accuracy
    best_epoch = 0

    for epoch in range(args.epochs):

        train_loss, train_acc = train(trainloader, model, criterion, optimizer,
                                        debug_batch_size=args.debug_batch_size, device=device)
        val_loss, val_acc = test(valloader, model, criterion, device=device)
        test_loss, test_acc = test(testloader, model, criterion)

        # print('\nEpoch: [%d | %d] LR: %f Train-Loss: %.4f Val-Loss: %.4f Train-Acc: %.4f Val-Acc: %.4f' % (epoch + 1, args.epochs, scheduler.lr_(), train_loss, val_loss, train_acc, val_acc))
        logger.append([epoch, (time.time() - time_start)/60., scheduler.lr_(),
                       train_loss, val_loss, test_loss, train_acc, val_acc, test_acc])

        # save model
        is_best = val_acc > best_val_acc
        if is_best:
            best_epoch = epoch + 1
        best_val_acc = max(val_acc, best_val_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_val_acc': best_val_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        # update grower and scheduler
        if args.grow:
            modelArch.update(epoch, is_best, model)
        errs = None
        if args.hooker:
            errs = hooker.output(epoch)
        scheduler.step_(epoch, errs)

        # grow
        if args.grow:
            if args.mode == 'fixed': # grow specified at fixed epochs
                if epoch + 1 in args.grow_epoch:
                    modelArch.grow(1) # dummy grow
                    model = models.__dict__[args.arch](num_classes=num_classes,
                                                       block_name=args.block_name,
                                                       archs=modelArch.arch)
                    to_parallel(model).to(device)
                    model.load_state_dict(modelArch.state_dict.state_dict, strict=False) # True) # False due to buffer added during training to calculate lipschitz
                    if args.scheduler == 'cosine' and not args.schedule:
                        # cases that learning rate is continued
                        optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                    else:
                        # cases that learning rate is reset, aka lr restart
                        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                    '''
                        probably have to copy the entire momentum history for each weight
                        but here just initialize the optimizer again
                    '''
                    # if multi epoch cosine or cosine_restart
                    scheduler.update(optimizer, epoch=epoch)
                    if args.hooker:
                        hooker.hook(model)
                    modelArch.record(epoch, model)

            elif args.mode == 'adapt': # grow happens automatically
                assert args.hooker, 'For adaptive training, model hooker must be provided to monitor some stats'
                trigger.feed(errs) 
                err_indices = trigger.trigger(epoch, modelArch.get_num_blocks_all_layer()) # propose candidate blocks to be growed
                if err_indices:
                    err_indices = modelArch.grow(err_indices) # try duplicate it to see if any layer exceeds upper limit
                    if err_indices:
                        model = models.__dict__[args.arch](num_classes=num_classes,
                                                           block_name=args.block_name,
                                                           archs=modelArch.arch)
                        to_parallel(model).to(device)
                        model.load_state_dict(modelArch.state_dict.state_dict, strict=False) # not strict matching due to Lipschitz buffer
                        if args.scheduler == 'cosine':
                            # cases that learning rate is continued
                            optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                        else:
                            # cases that learning rate is reset, aka lr restart
                            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                        if args.hooker:
                            hooker.hook(model)
                        trigger.update(err_indices) # reset history errors in trigger
                        scheduler.update(optimizer, epoch=epoch) # update optimizer in lr scheduler
                        modelArch.record(epoch, model) # keep the current architecture for record
            else:
                raise KeyError('Grow mode %s not supported!' % args.mode)

    # print growing stats
    if args.grow:
        print('\nGrow epochs: ', modelArch.grow_epochs[1:], end=', ')
        print('Num parameters: ', modelArch.num_parameters, end=', ')
        print('PPE: %.2f' % modelArch._get_ppe())

    # evaluate best model
    print('Best val acc: %.4f at %i' % (best_val_acc, best_epoch)) # this is the validation acc
    best_checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
    if args.grow:
        best_model = models.__dict__[args.arch](num_classes=num_classes,
                                                block_name=args.block_name,
                                                archs = modelArch.best_arch)
    else:
        best_model = models.__dict__[args.arch](num_classes=num_classes,
                                                depth=args.depth,
                                                block_name=args.block_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        best_model = torch.nn.DataParallel(best_model)
    best_model.to(device) # --
    best_model.load_state_dict(best_checkpoint['state_dict'], strict=False)

    test_loss, test_acc = test(testloader, best_model, criterion)
    if args.grow:
        print('Best arch: %s' % modelArch.__str__(best=True), end=', ')
    print('Best Test Loss:  %.4f, Best Test Acc:  %.4f' % (test_loss, test_acc))

    # evaluate final model
    test_loss, test_acc = test(testloader, model, criterion)
    if args.grow:
        print('Final arch: %s' % modelArch, end=', ')
    print('Final Test Loss:  %.4f, Final Test Acc:  %.4f' % (test_loss, test_acc))

    print('Wall time: %.3f mins' % ((time.time() - time_start)/60))

    # round off
    scheduler.close()
    if args.hooker:
        hooker.close()
    if args.grow:
        modelArch.close()
        if args.mode == 'adapt':
            trigger.close()
    logger.close()


if __name__ == '__main__':
    main()
