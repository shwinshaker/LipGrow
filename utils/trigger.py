from __future__ import absolute_import
import torch
import os
from . import Logger
from . import reduce_list

import statistics

__all__ = ['Trigger', 'MinTrigger', 'ConvergeTrigger', 'MoveMinTrigger', 'MinTolTrigger', 'TolTrigger']

class Trigger(object):
    def __init__(self, smooth='median', window=5, backtrack=10, thresh=1.1):
        """
            todo: pass in errs only, infer structures
        """

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        else:
            raise KeyError("Estimation method not allowed!")

        self.window = window
        self.backtrack = backtrack
        self.thresh = thresh

        self.history = []
        self.baseErrs = []

    def feed(self, errs):
        # assert len(errs) == len(self.history), "err: %i, history: %i" % (len(errs), len(self.history))

        # merge into history, and if exceeds capacity, dequeue
        if not self.history:
            self.history = [[[] for e in layer] for layer in errs]
        for l, layer in enumerate(errs):
            for b, err in enumerate(layer):
                self.history[l][b].append(err)

        # set base, happens when initialize, after model updated, or more than backtrack epochs elapsed
        if not self.baseErrs:
            self.baseErrs = [[None for e in layer] for layer in errs]
        for l, layer in enumerate(self.history):
            for b, err in enumerate(layer):
                if not self.baseErrs[l][b] and len(self.history[l][b]) >= self.window:
                    # should be exactly ==, relax to >=
                    self.baseErrs[l][b] = self.smooth(self.history[l][b][-self.window:])
                    print("Base error initialized for layer%i - block%i ." % (l, b))
                if len(self.history[l][b]) >= self.backtrack:
                    self.baseErrs[l][b] = self.smooth(self.history[l][b][-self.backtrack:-self.backtrack+self.window]) 

        print(self.baseErrs)
        for layer in self.history:
            for errs in layer:
                if len(errs) >= self.window:
                    print(self.smooth(errs[-self.window:]), end=' ')
        print()

    def trigger(self):
        err_indices = []
        for l, layer in enumerate(self.history):
            for b, err in enumerate(layer):
                if not self.baseErrs[l][b]:
                    # err base not yet set
                    continue
                if self.smooth(self.history[l][b][-self.window:]) / self.baseErrs[l][b] > self.thresh:
                    err_indices.append((l, b))
                    # clear history of updated layer
                    # self.history[l][b] = []
                    # clear base errors of updated layer
                    # self.baseErrs[l][b] = None

        if err_indices:
            print("Truncated error exceeds at blocks", sorted(err_indices))
        return sorted(err_indices)

        # todo - propose candidate layer based on average layer errr
    
    def _layer_average(self):
        '''
            average the err norm across layer for each epoch
            and see if the avg err norm in layer is increasing
                seems complicated and not very intuitive, let's use majority vote
        '''
        pass

    def update(self, block_indices):
        # now only update history for confirmed duplicate
        # excessive duplicate will not cut the history
        for l, b in block_indices:
            if b == len(self.history[l]):
                # there are n blocks, but only n-1 errs
                self.history[l].append([])
                self.baseErrs[l].append(None)
                continue
            self.history[l][b] = [[], []] # replace by two placeholders
            self.baseErrs[l][b] = [None, None]

        self.history = [reduce_list(layer, order=2) for layer in self.history]
        self.baseErrs = [reduce_list(layer) for layer in self.baseErrs]


    def close(self):
        pass
        # self.logger.close()

class ConvergeTrigger(object):
    def __init__(self, smooth='median', window=10, backtrack=10, thresh=0.0,
                 atom='model', err_atom='model'):

        if not atom == 'model' or not err_atom == 'model':
            raise NotImplementedError("ConvergeTrigger only support modelwise decision for now!")

        self.atom = atom
        self.err_atom = err_atom

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        else:
            raise KeyError("Estimation method not allowed!")

        self.window = window
        self.backtrack = backtrack
        self.thresh = thresh

        self.history = []

    def feed(self, err):
        assert(isinstance(err, float))

        # merge into history, and if exceeds capacity, dequeue
        self.history.append(err)

        # efficiency considerations, not a big deal
        # if len(self.history) > self.backtrack + self.window:
        #     self.history.pop(0)

    def _gradient(self, last=0):
        if last > 0:
            err0 = self.smooth(self.history[-self.backtrack-self.window-last:-self.backtrack-last])
            err = self.smooth(self.history[-self.window-last:-last]) 
        else:
            err0 = self.smooth(self.history[-self.backtrack-self.window:-self.backtrack])
            err = self.smooth(self.history[-self.window:]) 
        return (err - err0) / err0

    def trigger(self, arch=None):
        if len(self.history) < self.window + self.backtrack + 1:
            return 0

        print(self._gradient(), self._gradient(last=1))

        if self._gradient() <= self.thresh and self._gradient(last=1) > self.thresh:
            return 1
        return 0

    def update(self, err_index):
        self.history = []

    def close(self):
        pass


class MoveMinTrigger:
    def __init__(self, smooth='min', window=10, epochs=164):

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        elif smooth == 'min':
            self.smooth = min
        else:
            raise KeyError("Estimation method not allowed!")

        self.window = window
        self.epochs = epochs
        self.history = []

    def feed(self, err):
        assert(isinstance(err, float))

        print('err: %.4f' % err)
        # merge into history, and if exceeds capacity, dequeue
        self.history.append(err)

        # efficiency considerations, not a big deal
        # if len(self.history) > self.backtrack + self.window:
        #     self.history.pop(0)

    def trigger(self, epoch, arch=None):
        if len(self.history) < self.window + 1:
            return 0

        curr = self.smooth(self.history[-self.window:]) 
        last = self.smooth(self.history[-self.window-1:-1])
        print('curr: %.4f - last: %.4f' % (curr, last))
        if curr > last:
            return 1

        # ensures the final model will get at least 10 epochs of training, if allowed by architecture
        if self.epochs - epoch == self.window + 1:
            print('Forced grow!')
            return 1
        return 0

    def update(self, err_index):
        self.history = []

    def close(self):
        pass


class MinTrigger:
    def __init__(self, window=10, epochs=164):

        self.window = window
        self.epochs = epochs
        self.min = float("inf")
        self.buffer = 0

    def feed(self, err):
        assert(isinstance(err, float))

        print('err: %.4f min-err: %.4f buffer: %i' % (err, self.min, self.buffer))
        if err < self.min:
            self.min = err
            self.buffer = 0
        else:
            self.buffer += 1

    def trigger(self, epoch, arch=None):

        if self.buffer > self.window:
            return 1

        # ensures the final model will get at least 10 epochs of training, if allowed by architecture
        if self.epochs - epoch == self.window + 1:
            print('Forced grow!')
            return 1

        return 0

    def update(self, err_index):
        self.min = float("inf")
        self.buffer = 0

    def close(self):
        pass


class TolTrigger:
    def __init__(self, tolerance=1.1, window=1, reserve=20, epochs=164, smooth=statistics.mean, modelArch=None):

        self.tolerance = tolerance 
        self.window = window
        self.reserve = reserve
        self.epochs = epochs
        self.smooth = smooth
        self.modelArch = modelArch

        self.init_err = None
        
        self.history = []
        self.smooth_history = []

    def feed(self, err):
        assert(isinstance(err, float))

        self.history.append(err)
        if len(self.history) >= self.window:
            self.smooth_history.append(self.smooth(self.history[-self.window:]))

        if self.smooth_history:
            if not self.init_err:
                self.init_err = self.smooth_history[-1]
                return

            smooth_err = self.smooth_history[-1]
            print('[Tol Trigger] err: %.4f - smooth-err: %.4f - init-smooth-err: %.4f - ratio: %.4f' % (err, smooth_err, self.init_err, smooth_err/self.init_err))
        else:
            print('[Tol Trigger] err: %.4f - len-history: %i' % (err, len(self.history)))
    
    def trigger(self, epoch, arch=None):

        if not self.smooth_history:
            return 0

        ratio = self.smooth_history[-1] / self.init_err
        if ratio > self.tolerance:
            return 1

        # ensures every model will get at least several epochs of training
        # final model will get at least 30 epochs of training
        num_grows_left = self.modelArch.get_grows_left()
        # if self.epochs - epoch == self.reserve + (num_grows_left-1) * self.window * 2 + 1:
        if self.epochs - epoch == self.reserve + (num_grows_left-1) * self.window + 1:
            print('Forced grow!')
            return 1

        return 0

    def update(self, err_index):
        # Input
        #   err_index: index of grow (e.g. block number)

        self.init_err = None

        self.history = []
        self.smooth_history = []

        self.tolerance = 1 + (self.tolerance - 1)/2
        print('tolerance updated to: %.4f' % self.tolerance)


    def close(self):
        pass


class MinTolTrigger:
    def __init__(self, tolerance=1.1, window=1, reserve=20, epochs=164, smooth=statistics.mean):

        self.tolerance = tolerance 
        self.window = window
        self.reserve = reserve
        self.epochs = epochs
        # self.min_err = float("inf")
        # self.buffer = 0
        self.smooth = smooth
        
        self.history = []
        self.smooth_history = []

    def feed(self, err):
        assert(isinstance(err, float))

        self.history.append(err)
        if len(self.history) >= self.window:
            self.smooth_history.append(self.smooth(self.history[-self.window:]))
        # print('err: %.4f - min-err: %.4f - ratio: %.4f - buffer: %i' % (err, self.min_err, err/self.min_err, self.buffer))

        if self.smooth_history:
            smooth_err = self.smooth_history[-1]
            min_smooth_err = min(self.smooth_history)
            print('err: %.4f - smooth-err: %.4f - min-smooth-err: %.4f - ratio: %.4f' % (err, smooth_err, min_smooth_err, smooth_err/min_smooth_err))
        else:
            print('err: %.4f - len-history: %i' % (err, len(self.history)))
    
        # self.ratio = err / self.min_err
        # self.min_err = min(self.min_err, err)

    def trigger(self, epoch, arch=None):

        # ensures each model will at least get 10 epochs of training
        # self.buffer += 1
        # if self.buffer <= self.reserve:
        #     return 0

        if not self.smooth_history:
            return 0

        ratio = self.smooth_history[-1] / min(self.smooth_history)
        if ratio > self.tolerance:
            return 1

        # ensures the final model will get at least 10 epochs of training, if allowed by architecture
        if self.epochs - epoch == self.reserve + 1:
            print('Forced grow!')
            return 1

        return 0

    def update(self, err_index):
        # self.min_err = float("inf")
        # self.buffer = 0
        self.history = []
        self.smooth_history = []

    def close(self):
        pass


# class MinTrigger(object):
#     
#     def __init__(self, smooth='median', thresh=1.1, atom='block', err_atom='block'):
#         """
#             atom: grow atom, atom for output indices
#             err_atom: input err atom from hooker
#             requirement: grow atom > err atom
#         """
# 
#         if smooth == 'median':
#             self.smooth = statistics.median
#         elif smooth == 'mean':
#             self.smooth = statistics.mean
#         else:
#             raise KeyError("Estimation method not allowed!")
# 
#         self.thresh = thresh
# 
#         self._set_protocol(atom, err_atom)
#         self.atom = atom
#         self.err_atom = err_atom
# 
#         self.minErrs = None
#         self.curErrs = None
# 
#     def _set_protocol(self, atom, err_atom):
#         if err_atom == 'block':
#             self.feed = self._feed_block
#         elif err_atom == 'layer':
#             self.feed = self._feed_layer
#         elif err_atom == 'model':
#             self.feed = self._feed_model
#         else:
#             raise KeyError(err_atom)
# 
#     def _feed_block(self, errs):
#         raise KeyError('Not implemented yet!')
# 
#     def _feed_layer(self, errs):
#         if self.minErrs:
#             assert len(errs) == len(self.minErrs), "err: %i, min errs: %i" % (len(errs), len(self.minErrs))
# 
#         if not self.minErrs:
#             self.minErrs = [e for e in errs]
#             self.curErrs = [e for e in errs]
#             return
# 
#         for l in range(len(self.minErrs)):
#             if not self.minErrs[l]:
#                 self.minErrs[l] = errs[l]
#             else:
#                 self.minErrs[l] = min(self.minErrs[l], errs[l])
# 
#         self.curErrs = [e for e in errs]
# 
#         print('cur err: ', self.curErrs)
#         print('min errs: ', self.minErrs)
# 
#     def _feed_model(self, errs):
#         assert isinstance(errs, float), 'model err must be a single float number!'
#         if not self.minErrs:
#             self.minErrs = errs
#             self.curErrs = errs
#             return
# 
#         self.minErrs = min(self.minErrs, errs)
#         self.curErrs = errs
# 
#         print('cur err: ', self.curErrs)
#         print('min errs: ', self.minErrs)
# 
#     def criterion(self):
#         if self.err_atom == 'block':
#             err_indices = []
#             for l, layer in enumerate(self.curErrs):
#                 for b, e in enumerate(layer):
#                     if e / self.minErrs[l][b] > self.thresh:
#                         err_indices.append((l,b))
#             return sorted(err_indices)
# 
#         if self.err_atom == 'layer':
#             err_indices = []
#             for l, (e, me) in enumerate(zip(self.curErrs, self.minErrs)):
#                 if e / me > self.thresh:
#                     err_indices.append(l)
#             return sorted(err_indices)
# 
#         if self.err_atom == 'model':
#             if self.curErrs / self.minErrs > self.thresh:
#                 return 1
# 
#         return None
# 
#     def trigger(self, arch):
#         '''
#             arch: [3,3,3] or [3,6,6], ...
#         '''
# 
#         indices = self.criterion()
# 
#         if self.atom == 'block':
#             return indices
#     
#         if self.atom == 'layer':
#             if self.err_atom == 'layer':
#                 return indices
# 
#             if self.err_atom == 'block':
#                 count = [0] * len(self.arch)
#                 for l, b in indices:
#                     count[l] += 1
#                 layer_indices = []
#                 for l, (num_err, num_all) in enumerate(zip(count, arch)):
#                     if num_err / num_all > 0.5:
#                         layer_indices.append(l)
#                 return layer_indices
# 
# 
#         if self.atom == 'model':
#             if self.err_atom == 'model':
#                 return indices
# 
#             if self.err_atom == 'layer':
#                 if len(indices) > 0:
#                     return 1
#             
#             if self.err_atom == 'block':
#                 if len(indices) / sum(arch) > 0.5:
#                     return 1
#             
#             # print("Truncated error exceeds at layers", sorted(err_indices))
# 
#         return None
# 
#     def update(self, indices):
#         if self.err_atom == 'block':
#             raise KeyError('Not implemented yet!')
# 
#         if self.err_atom == 'layer':
#             if self.atom == 'layer':
#                 for l in indices:
#                     self.minErrs[l] = None
#                 return
# 
#             if self.atom == 'model':
#                 if indices:
#                     self.minErrs = []
#                 return
# 
#         if self.err_atom == 'model':
#             if indices:
#                 self.minErrs = None
#             return
# 
#     def close(self):
#         pass
#         # self.logger.close()
