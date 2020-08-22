import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os


__all__ = ['get_loaders']

def get_loaders(dataset='cifar10', download=False,
                train_batch=128, test_batch=100, n_workers=4, 
                data_dir='./data'):

    # Data
    print('==> Preparing dataset %s' % dataset)
    if dataset == 'cifar10':
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
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
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
    elif dataset == 'imagenet':
        dataloader = datasets.ImageNet
        num_classes = 1000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
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
    elif dataset == 'tiny-imagenet':
        # custom dataloader
        num_classes = 200
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])}

    else:
        raise KeyError(dataset)

    if dataset.startswith('cifar'):
        # test set size: 10,000
        testset = dataloader(root=data_dir, train=False, download=download, transform=transform_test)
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=test_batch, shuffle=False, num_workers=n_workers)
        testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=n_workers)

        # training set size: 50,000 - 10,000 = 40,000
        trainset = dataloader(root=data_dir, train=True, download=download, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=n_workers)
    elif dataset == 'imagenet':
        # dataset size: 1000 classes * 50,000 per class
        testset = dataloader(root=data_dir, split='val', download=download, transform=transform_test)
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=test_batch, shuffle=False, num_workers=n_workers)
        testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=n_workers) # , pin_memory=True)

        trainset = dataloader(root=data_dir, split='train', download=download, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=n_workers) # , pin_memory=True)
    elif dataset == 'tiny-imagenet':
        # dataset size:
        #   train:  200 classes * 500 per class
        #   val: 200 classes * 25 per class
        #   test: 200 classes * 25 per class (original test is not labeled, split val)
        testset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'val'), transform=data_transforms['test'])
        valset = data.Subset(testset, range(len(testset)//2))
        testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
        valloader = data.DataLoader(valset, batch_size=test_batch, shuffle=False, num_workers=n_workers)
        testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=n_workers)

        trainset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200', 'train'), transform=data_transforms['train'])
        trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=n_workers)
    else:
        raise KeyError(dataset)


    return trainloader, valloader, testloader, num_classes



