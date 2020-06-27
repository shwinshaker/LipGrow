'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
from scipy import optimize
import itertools

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import ModelHooker, Trigger, MinTrigger, ConvergeTrigger, MoveMinTrigger, MinTolTrigger, TolTrigger
from utils import LipHooker
from utils import ModelArch
from utils import str2bool, is_powerOfTwo

from utils import scheduler as schedulers
from utils import regularizer as regularizers

import warnings

torch.autograd.set_detect_anomaly(True)
# torch.autograd.set_detect_anomaly(False)

scheduler_names = sorted(name for name in schedulers.__dict__
    if name.islower() and not name.startswith("__")
    and callable(schedulers.__dict__[name]))

regularizer_names = sorted(name for name in regularizers.__dict__
    if name.islower() and not name.startswith("__")
    and callable(regularizers.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 + Imagenet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--scheduler', type=str, default='constant', choices=scheduler_names,
                    help='scheduler type: none, step, cosine, or expo, adapt, acosine, adacosine') # cosinerestart')
parser.add_argument('--schedule', type=int, nargs='*', default=[81, 122],
                    help='Decrease learning rate at these epochs. Required if scheduler if true')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--regularization', type=str, default=None, nargs='?', #choices=regularizer_names,
                    help='custom regularizer type: none, truncateError')
parser.add_argument('--r_gamma', default=1e-4, type=float, help='regularization coefficient (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet') # ,
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--debug-batch-size', type=int, default=0, help='number of training batches for quick check. default: 0 - no debug')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#growth
parser.add_argument('--grow', type=str2bool, const=True, default=False, nargs='?', help='Let us grow!')
parser.add_argument('--mode', type=str, choices=['adapt', 'fixed'], default='adapt', help='The growing mode: adaptive to errs, or fixed at some epochs')
# todo
parser.add_argument('--grow-atom', type=str, choices=['block', 'layer', 'model'], default='block', help='blockwise, layerwise or modelwise?')
parser.add_argument('--err-atom', type=str, choices=['block', 'layer', 'model'], default='block', help='Measure errs in block, layer or model level?')
parser.add_argument('--grow-operation', type=str, choices=['duplicate', 'plus'], default='duplicate', help='duplicate or plus?')
parser.add_argument('--grow-epoch', type=str, nargs='*', default=['60', '110'], help='Duplicate the model at these epochs. Required if mode = fixed.')
parser.add_argument('--max-depth', type=int, default=74, help='Max model depth. Required if mode = adapt.')
parser.add_argument('--window', type=int, default=3, help='Smooth scope of truncated err estimation. Required if mode = adapt.')
parser.add_argument('--reserve', type=int, default=20, help='Reserved epochs for final model')
parser.add_argument('--backtrack', type=int, default=30, help='History that base err tracked back to.  Required if mode = adapt.')
parser.add_argument('--threshold', type=float, default=1.1, help='Err trigger threshold for growing.  Required if mode = adapt.')
parser.add_argument('--scale', type=str2bool, const=True, default=True, nargs='?', help='Scale the residual by activations? Scale the acceleration by residuals?')
parser.add_argument('--scale-stepsize', type=str2bool, const=True, default=False, nargs='?', help='Scale the residual by stepsize?')
# trace
# parser.add_argument('--hook', type=str2bool, const=True, default=True, nargs='?', help='Hook model to output some info')
parser.add_argument('--hooker', type=str, choices=['None', 'Model', 'Lip'], default='Model', help='Hooker on model to output some info')
parser.add_argument('--trace', type=str, nargs='+', default=['norm'], help='Trace and output some intermediate products.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# switch models dir
if args.dataset.lower().startswith('cifar'):
    import models.cifar as models
elif 'imagenet' in args.dataset.lower():
    import models.imagenet as models
else:
    raise KeyError(args.dataset)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# arguments post process
if args.hooker == 'None':
    args.hooker = None

# sanity check
def get_nbpl(depth):
    if 'cifar' in args.dataset.lower():
        n = 6
    elif 'imagenet' in args.dataset.lower():
        n = 8
    else:
        raise KeyError
    assert (depth - 2) % n == 0, 'depth should be %in+2, got %i!' % (n, depth)
    return (depth - 2) // n

def solve_multi(min_step, num_grows, epochs):
    def fun(x):
        return min_step * (1 - math.pow(x, num_grows+1)) / (1 - x) - epochs
    sol = optimize.root(fun, 2, method='hybr')
    if sol.success:
        print('multiplicity: %.4f' % sol.x[0])
        return sol.x[0]
    raise RuntimeError('solver failed!')

if args.mode == 'fixed':
    assert args.grow_atom == 'model', 'require model level grow for fixed!'
    if args.grow_epoch[0].isdigit():
        assert all([s.isdigit() for s in args.grow_epoch]), 'provide several int milestones'
        args.grow_epoch = [int(s) for s in args.grow_epoch]
    else:
        assert len(args.grow_epoch) == 1, 'otherwise choose mode: even, cold, warm'
        assert args.max_depth, 'max depth has to be provided for automatically scheduled grow!'
        schedule_mode = args.grow_epoch[0]
        assert schedule_mode in ['even', 'cold', 'warm'], 'unexpected mode %s!' % args.grow_epoch[0]
        nb = get_nbpl(args.depth)
        nb_max = get_nbpl(args.max_depth)
        if args.grow_operation == 'duplicate':
            assert nb_max % nb == 0
            assert is_powerOfTwo(nb_max // nb)
            num_grows = int(math.log2(nb_max / nb))
        elif args.grow_operation == 'plus':
            num_grows = nb_max - nb
        else:
            raise KeyError(args.grow_operation)
        if schedule_mode == 'even':
            assert args.epochs > num_grows, '# epochs should be greater than # grows!'
            step = args.epochs // (num_grows + 1)
            args.grow_epoch = list(range(step, args.epochs, step))
            if (args.epochs - args.grow_epoch[-1]) < step: args.grow_epoch.pop()
        elif schedule_mode == 'cold':
            min_step = 5
            multi = solve_multi(min_step, num_grows, args.epochs)
            args.grow_epoch = list(map(int, itertools.accumulate([min_step*(multi**i) for i in range(num_grows+1)][::-1])))
            args.grow_epoch.pop()
        elif schedule_mode == 'warm':
            min_step = 5
            multi = solve_multi(min_step, num_grows, args.epochs)
            args.grow_epoch = list(map(int, itertools.accumulate([min_step*(multi**i) for i in range(num_grows+1)])))
            args.grow_epoch.pop()
        else:
            raise KeyError(scedule_mode)

    if args.grow_operation == 'duplicate':
        assert get_nbpl(args.depth) * 2**(len(args.grow_epoch)) == get_nbpl(args.max_depth), '# grows untally with depth by %s! depth: %i, max-depth: %i, # grows: %i' % (args.grow_operation, args.depth, args.max_depth, len(args.grow_epoch))
    elif args.grow_operation == 'plus':
        assert get_nbpl(args.depth) + len(args.grow_epoch) == get_nbpl(args.max_depth), '# grows untally with depth by %s! depth: %i, max-depth: %i, # grows: %i' % (args.grow_operation, args.depth, args.max_depth, len(args.grow_epoch))
    else:
        raise KeyError(args.grow_operation)

if args.mode == 'adapt':
    assert args.max_depth, 'require max depth for adaptive mode'

# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet']
# assert args.backtrack > args.window, 'backtrack should at least greater than window size.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = "cpu"
# ---------------------
# use_cuda = False
# ---------------------

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_val_acc = 0  # best test accuracy

def main():
    time_start = time.time()

    global best_val_acc
    best_epoch = 0

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # 32x32 -> 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'imagenet':
        dataloader = datasets.ImageNet
        num_classes = 1000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224), # 256 -> 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset == 'tiny-imagenet':
        # custom dataloader
        num_classes = 200
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                # transforms.RandomCrop(64, padding=4), # 32x32 -> 32x32
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])}

    else:
        raise KeyError(args.dataset)

    if args.dataset.startswith('cifar'):
        # test set size: 10,000
        testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        # training set size: 50,000 - 10,000 = 40,000
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    elif args.dataset == 'imagenet':
        # dataset size: 1000 classes * 50,000 per class
        testset = dataloader(root='./data', split='val', download=True, transform=transform_test)
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers) # , pin_memory=True)

        trainset = dataloader(root='./data', split='train', download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers) # , pin_memory=True)
    elif args.dataset == 'tiny-imagenet':
        # dataset size:
        #   train:  200 classes * 500 per class
        #   val: 200 classes * 25 per class
        #   test: 200 classes * 25 per class (original test is not labeled, split val)
        testset = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'val'), transform=data_transforms['test'])
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        trainset = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=data_transforms['train'])
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        print(type(trainloader))
        print(len(trainloader))
    else:
        raise KeyError(args.dataset)


    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('midnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('preresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('accnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('transresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # if use_cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model) # .cuda()
    # model.cuda()
    model.to(device) # --
    cudnn.benchmark = True

    hooker = None
    if args.hooker:
        if args.hooker == 'Model':
            # model hooker
            hooker = ModelHooker(args.arch, args.checkpoint, atom=args.err_atom, scale=args.scale,
                                 scale_stepsize=args.scale_stepsize, device=device,
                                 trace=args.trace)
        elif args.hooker == 'Lip':
            hooker = LipHooker(args.arch, args.checkpoint, resume=args.resume, device=device)
        else:
            raise KeyError(args.hooker)
        hooker.hook(model)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # learning rate scheduler
    print("==> creating scheduler '{}'".format(args.scheduler))
    if args.scheduler.startswith('constant'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, dpath=args.checkpoint)
    elif args.scheduler.startswith('step'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, gamma=args.gamma, dpath=args.checkpoint)
    elif args.scheduler.startswith('cosine') or args.scheduler == 'adacosine':
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, epochs=args.epochs, dpath=args.checkpoint)
    elif args.scheduler.startswith('expo'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, gamma=args.gamma, dpath=args.checkpoint)
    elif args.scheduler.startswith('adapt'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, gamma=args.gamma, dpath=args.checkpoint)
    elif args.scheduler.startswith('acosine'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, epochs=args.epochs, dpath=args.checkpoint)
    else:
        raise KeyError(args.scheduler)

    # custom regularization
    regularizer = None
    if args.regularization:
        print("==> creating regularizer '{}'".format(args.regularization))
        if args.regularization.startswith('truncate_error'):
            regularizer= regularizers.__dict__[args.regularization](hooker=hooker, r_gamma=args.r_gamma)
        else:
            raise KeyError(args.regularization)

    # print information
    print("     ----------------------------- %s ----------------------------------" % args.arch)
    print("     depth: %i" % args.depth)
    print(model)
    print("     ----------------------------------------------------------------------")
    print("     dataset: %s" % args.dataset)

    print("     --------------------------- hypers ----------------------------------")
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
    if args.regularization:
        print("     Regularization: %s" % args.regularization)
        print("     Regularization coefficient: %g" % args.r_gamma)
    print("     gpu id: %s" % args.gpu_id)
    print("     num workers: %i" % args.workers)
    print("     hooker: ", args.hooker)
    print("     trace: ", args.trace)
    print("     --------------------------- model ----------------------------------")
    print("     Model: %s" % args.arch)
    print("     depth: %i" % args.depth)
    print("     block: %s" % args.block_name)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))
    if args.grow:
        if not args.arch in ['resnet', 'transresnet', 'preresnet']:
            raise KeyError("model not supported for growing yet.")
        print("     --------------------------- growth ----------------------------------")
        print("     grow mode: %s" % args.mode)
        print("     grow atom: %s" % args.grow_atom)
        print("     grow operation: %s" % args.grow_operation)
        print("     stepsize scaled residual: %s" % args.scale_stepsize)
        if args.mode == 'fixed':
            print("     grow milestones: ", args.grow_epoch)
        else:
            print("     max depth: %i" % args.max_depth)
            print("     scaled down err: %s" % args.scale)
            print("     err atom: %s" % args.err_atom)
            print("     err threshold: %g" % args.threshold)
            print("     smoothing scope: %i" % args.window)
            print("     reserved epochs: %i" % args.reserve)
            print("     err back track history (deprecated): %i" % args.backtrack)
    if args.debug_batch_size:
        print("     -------------------------- debug ------------------------------------")
        print("     debug batches: %i" % args.debug_batch_size)
    print("     ---------------------------------------------------------------------")

    # Resume
    title = args.dataset + '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        if args.regularization:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Acc.', 'Valid Acc.', 'Test Acc.', 'Regular Loss'])
        else:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Acc.', 'Valid Acc.', 'Test Acc.'])

    # ---------- grow -----------
    # model architecture tracker
    modelArch=None
    if args.grow:
        modelArch = ModelArch(args.arch, model, args.epochs, args.depth, max_depth=args.max_depth, dpath=args.checkpoint, operation=args.grow_operation, atom=args.grow_atom, dataset=args.dataset)
    # timer
    timeLogger = Logger(os.path.join(args.checkpoint, 'timer.txt'), title=title)
    timeLogger.set_names(['epoch', 'training-time(min)', 'end-time(min)'])

    # trigger
    if args.grow and args.mode == 'adapt':
        # trigger = Trigger(window=args.window, backtrack=args.backtrack, thresh=args.threshold, smooth='median') # test
        # trigger = MinTrigger(thresh=args.threshold, smooth='median', atom=args.grow_atom, err_atom=args.err_atom) # test
        # trigger = MinTrigger(window=args.window, epochs=args.epochs) # test
        # trigger = MinTolTrigger(tolerance=args.threshold, window=args.window, reserve=args.reserve, epochs=args.epochs) # test
        trigger = TolTrigger(tolerance=args.threshold, window=args.window, reserve=args.reserve, epochs=args.epochs, modelArch=modelArch) # test
        # trigger = ConvergeTrigger(smooth='median', atom=args.grow_atom, err_atom=args.err_atom, window=args.window, backtrack=args.backtrack, thresh=args.threshold) # test
        # trigger = MoveMinTrigger(smooth='min', window=args.window, epochs=args.epochs) # test

    # evaluation mode
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    epoch_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, args.)

        # count the training time only
        end = time.time()
        if not regularizer:
            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda,
                                          regularizer=regularizer)
        else:
            train_loss, regular_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda,
                                                        regularizer=regularizer)
        train_end = time.time()

        # errs = hooker.output(epoch, archs=modelArch.arch, atom=args.err_atom, scale=args.scale)
        val_loss, val_acc = test(valloader, model, criterion, epoch, use_cuda, hooker)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, hooker=None)

        print('\nEpoch: [%d | %d] LR: %f Train-Loss: %.4f Val-Loss: %.4f Train-Acc: %.4f Val-Acc: %.4f' % (epoch + 1, args.epochs, scheduler.lr_(), train_loss, val_loss, train_acc, val_acc))
        # append logger file
        if not regularizer:
            logger.append([scheduler.lr_(), train_loss, val_loss, test_loss, train_acc, val_acc, test_acc])
        else:
            logger.append([scheduler.lr_(), train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, regular_loss])

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

        # activations trace
        ## it's not clear should do this for training or for testing. Chang did for test.
        '''
            try it for training set?
                costs
        '''
        '''
            sometime wrap the following into a grower
                then `grower.step()`
        '''

        if args.grow:
            modelArch.update(epoch, is_best, model)
        # state dict is updated for this step: updated by the newly trained weights
        # model arch is not updated, it will update along with grow
        errs = None
        if args.hooker:
            errs = hooker.output(epoch)
            # errs = hooker.draw(epoch, archs=modelArch.arch)

        # learning rate scheduler
        scheduler.step_(epoch, errs)

        if args.grow:
            if args.mode == 'fixed':
                # existed method
                if epoch+1 in args.grow_epoch: # justin 12.14: changed `epoch` to `epoch+1`
                    modelArch.grow(1)
                    print('New archs: %s' % modelArch)
                    model = models.__dict__[args.arch](num_classes=num_classes,
                                                       block_name=args.block_name,
                                                       archs=modelArch.arch)
                    # if use_cuda:
                    #     model.cuda()
                    if torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model = torch.nn.DataParallel(model) # .cuda()
                    # model.cuda()
                    model.to(device) # --

                    model.load_state_dict(modelArch.state_dict.state_dict, strict=False) # True) # False due to buffer to calculate lipschitz
                    # optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)
                    # if cosine
                    if args.scheduler == 'cosine' and not args.schedule:
                        # no restarts
                        optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                    else:
                        # default setting
                        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                    '''
                        not sure if have to copy the entire momentum history for each weight
                        here just initialize the optimizer again
                    '''
                    # if multi epoch cosine or cosine_restart
                    scheduler.update(optimizer, epoch=epoch)
                    if args.hooker:
                        hooker.hook(model)
                    modelArch.record(epoch, model)

            elif args.mode == 'adapt':
                assert args.hooker

                # propose candidate blocks by truncated errs of each residual block
                trigger.feed(errs) 
                # err_indices = trigger.trigger(modelArch.get_num_blocks_all_layer()) 
                err_indices = trigger.trigger(epoch, modelArch.get_num_blocks_all_layer()) 
                if err_indices:
                    # try duplicate it to see if any layer exceeds upper limit
                    err_indices = modelArch.grow(err_indices)
                    if err_indices:

                        print('growed module indices: ', err_indices)
                        print('New archs: %s' % modelArch)

                        model = models.__dict__[args.arch](num_classes=num_classes,
                                                           block_name=args.block_name,
                                                           archs=modelArch.arch)
                        # if use_cuda:
                        #     model.cuda()
                        if torch.cuda.device_count() > 1:
                            print("Let's use", torch.cuda.device_count(), "GPUs!")
                            model = torch.nn.DataParallel(model) # .cuda()
                        # model.cuda()
                        model.to(device) # --

                        model.load_state_dict(modelArch.state_dict.state_dict, strict=False) # True) # False due to Lipschitz buffer
                        if args.scheduler == 'cosine': #  and not args.schedule:
                            # cosine for adapt, no schedule will be provided
                            optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                        else:
                            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                        if args.hooker:
                            hooker.hook(model)

                        # update history shape in trigger
                        trigger.update(err_indices)
                        scheduler.update(optimizer, epoch=epoch)
                        modelArch.record(epoch, model)
            else:
                raise KeyError('Grow mode %s not supported!' % args.mode)

        timeLogger.append([epoch, (train_end - end)/60, (time.time() - epoch_start)/60])

    scheduler.close()
    if args.hooker:
        hooker.close()
    timeLogger.close()
    # err_logger.close()
    if args.grow:
        modelArch.close()
        if args.mode == 'adapt':
            trigger.close()
    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    if args.grow:
        print('\nGrow epochs: ', modelArch.grow_epochs[1:], end=', ')
        print('Num parameters: ', modelArch.num_parameters, end=', ')
        print('PPE: %.2f' % modelArch._get_ppe())

    # best model
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

    # if use_cuda:
    #     best_model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        best_model = torch.nn.DataParallel(best_model) # .cuda()
    # model.cuda()
    best_model.to(device) # --
    best_model.load_state_dict(best_checkpoint['state_dict'], strict=False)

    test_loss, test_acc = test(testloader, best_model, criterion, -1, use_cuda, hooker=None)
    if args.grow:
        print('Best arch: %s' % modelArch.__str__(best=True), end=', ')
    print('Best Test Loss:  %.4f, Best Test Acc:  %.4f' % (test_loss, test_acc))

    # final model
    test_loss, test_acc = test(testloader, model, criterion, -1, use_cuda, hooker=None)
    if args.grow:
        print('Final arch: %s' % modelArch, end=', ')
    print('Final Test Loss:  %.4f, Final Test Acc:  %.4f' % (test_loss, test_acc))

    print('Wall time: %.3f mins' % ((time.time() - time_start)/60))


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, regularizer=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    # 
    r_losses = AverageMeter()

    if args.debug_batch_size:
        bar = Bar('Processing', max=args.debug_batch_size)
    else:
        bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.debug_batch_size:
            if batch_idx >= args.debug_batch_size:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True) # async=True)
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if regularizer:
            loss_ = regularizer.loss()
            loss += loss_
            r_losses.update(loss_.data.item(), inputs.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    if regularizer:
        return (losses.avg, r_losses.avg, top1.avg)

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, hooker=None):
    '''
        `epoch` is never used
    '''
    global best_val_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        # # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # justin: 12-27
            if hooker == 'Model':
                hooker.collect()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
