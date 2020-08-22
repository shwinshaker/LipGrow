#!./env python

from __future__ import absolute_import
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR, ExponentialLR
import os
from . import Logger
import warnings

# __all__ = ['ConstantLR', 'MultiStepLR_', 'MultiStepCosineLR', 'ExponentialLR_']

class ConstantLR(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, dpath='.'):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        super(ConstantLR, self).__init__(optimizer, last_epoch)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of unique lrs!'
        self.logger.append([epoch, lrs[0]])

        self.step()

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

    def lr_(self):
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        self.optimizer = optimizer

    def close(self):
        self.logger.close()

def constant(**kwargs):
    return ConstantLR(**kwargs)


class MultiStepLR_(MultiStepLR):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, dpath='.'):
        super(MultiStepLR_, self).__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of unique lrs!'
        self.logger.append([epoch, lrs[0]])

        self.step()

    def lr_(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(set(lrs)) == 1)
        assert(lrs[0] == self.get_lr()[0]), (lrs[0], self.get_lr()[0], self.last_epoch)
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        self.optimizer = optimizer

    def close(self):
        self.logger.close()

def step(**kwargs):
    return MultiStepLR_(**kwargs)


class MultiStepCosineLR(CosineAnnealingLR):

    def __init__(self, optimizer, milestones, epochs, eta_min=0.001, dpath='.'):
        ## -----------
        # self.temp_list = milestones
        # ---------------
        self.milestones = iter(sorted(milestones + [epochs, 2*epochs])) # last `epochs` is dummy
        self.eta_min = eta_min
        self.T = 0
        self.next_T = next(self.milestones)
        super(MultiStepCosineLR, self).__init__(optimizer, self.next_T - self.T, eta_min)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        print(epoch, self.last_epoch, self.T_max, self.T, self.next_T)

        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of unique lrs!'
        self.logger.append([epoch, lrs[0]])

        if epoch == self.next_T-1:
            self.last_epoch = -1
            self.T = self.next_T
            self.next_T = next(self.milestones)
            self.T_max = self.next_T - self.T

        self.step()

    def lr_(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(set(lrs)) == 1)
        assert(lrs[0] == self.get_lr()[0]), (lrs[0], self.get_lr()[0], self.last_epoch)
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        self.optimizer = optimizer

    def close(self):
        self.logger.close()

def cosine(**kwargs):
    return MultiStepCosineLR(**kwargs)


class AdaptMultiStepCosineLR(CosineAnnealingLR):

    def __init__(self, optimizer, epochs, eta_min=0.001, dpath='.'):

        self.eta_min = eta_min
        self.T = 0
        self.next_T = epochs
        super(AdaptMultiStepCosineLR, self).__init__(optimizer, self.next_T - self.T, eta_min)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        print('[Lr Scheduler] epoch: %i - last_epoch: %i - T_max: %i - T: %i - next_T: %i' % (epoch, self.last_epoch, self.T_max, self.T, self.next_T))

        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of lrs!'
        self.logger.append([epoch, lrs[0]])
        self.step()

    def lr_(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(set(lrs)) == 1)
        assert(lrs[0] == self.get_lr()[0]), (lrs[0], self.get_lr()[0], self.last_epoch, 'Inconsistent learning rate between scheduler and optimizer!')
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        # update will only be called in grow case
        self.optimizer = optimizer
        self.last_epoch = 0
        self.T_max = self.next_T - epoch - 1

    def close(self):
        self.logger.close()

def adacosine(**kwargs):
    return AdaptMultiStepCosineLR(**kwargs)


class CosineLR(CosineAnnealingLR):

    def __init__(self, optimizer, epochs, eta_min=0.001, dpath='.'):
        warnings.warn('hardcoded T_max')
        super(CosineLR, self).__init__(optimizer, epochs//3, eta_min)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of unique lrs!'
        self.logger.append([epoch, lrs[0]])

        self.step()

    def lr_(self):
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        self.optimizer = optimizer
        self.last_epoch = 0

    def close(self):
        self.logger.close()

def acosine(**kwargs):
    return CosineLR(**kwargs)


class ExponentialLR_(ExponentialLR):

    def __init__(self, optimizer, gamma=0.1, last_epoch=-1, dpath='.'):
        super(ExponentialLR_, self).__init__(optimizer, gamma=gamma, last_epoch=last_epoch)

        self.logger = Logger(os.path.join(dpath, 'Learning_rate.txt'))
        self.logger.set_names(['epoch', 'learning_rate'])

    def step_(self, epoch, err):
        lrs = self.get_lr()
        assert(len(set(lrs)) == 1), 'unexpected number of unique lrs!'
        self.logger.append([epoch, lrs[0]])

        self.step()

    def lr_(self):
        return self.get_lr()[0]

    def update(self, optimizer, epoch=None):
        raise NotImplementedError('Make sure this init is correct, now optimizer updated from args.lr!')
        super(ExponentialLR_, self).__init__(optimizer, gamma=self.gamma, last_epoch=-1)
        # self.optimizer = optimizer
        # self.last_epoch = 0

    def close(self):
        self.logger.close()

def expo(**kwargs):
    return ExponentialLR_(**kwargs)