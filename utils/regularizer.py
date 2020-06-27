#!./env python

from __future__ import absolute_import
import torch

class TruncateErrorRegularizer:

    def __init__(self, hooker, r_gamma=1e-4):
        self.hooker = hooker
        self.r_gamma = r_gamma

    def loss(self):
        loss = 0
        for layerHooker in self.hooker.layerHookers:
            _, _, errs = layerHooker._get_activations(detach=False)
            # _, errs, _ = layerHooker._get_activations(detach=False)
            loss += sum(torch.norm(err) / err.size()[0] for err in errs) / len(errs)
        loss /= len(self.hooker)
        return loss * self.r_gamma

def truncate_error(**kwargs):
    return TruncateErrorRegularizer(**kwargs)


