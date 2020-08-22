from __future__ import absolute_import
import torch
import os
from . import Logger
import statistics

__all__ = ['TolTrigger']

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
            print('[Tol Trigger] err: %.4f - smooth-err: %.4f - init-smooth-err: %.4f - ratio: %.4f - threshold: %.4f' % (err, smooth_err, self.init_err, smooth_err/self.init_err, self.tolerance))
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
        if num_grows_left > 0 and self.epochs - epoch == self.reserve + (num_grows_left-1) * self.window + 1:
            print('[Tol Trigger] Forced grow at epoch %i.' % epoch)
            return 1

        return 0

    def update(self, err_index):
        # Input
        #   err_index: index of grow (e.g. block number)

        self.init_err = None

        self.history = []
        self.smooth_history = []


    def close(self):
        pass
