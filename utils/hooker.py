from __future__ import absolute_import
import torch
import os
import statistics
import pickle
from sklearn.decomposition import PCA
import copy
from collections import defaultdict
import numpy as np
from more_itertools import peekable
import warnings
from . import Logger

__all__ = ['Hooker', 'LayerHooker', 'ModelHooker']

class Record:

    def __init__(self, reduction='norm', n=None, pca=None):
        self.li = []

        if reduction == 'norm':
            self.reduce = self.__norm
            self.batch_reduce = self.__batch_norm
        elif reduction == 'pc':
            self.pca = pca
            if not self.pca:
                self.reduce = lambda x: x
            else:
                self.reduce = self.__pca_transform
            self.batch_reduce = self.__batch_pc
            self.n = n
        else:
            raise KeyError(reduction)
        self.concat = self.__concat

    def absorb(self, li):
        if not self.li:
            self.li = [self.reduce(e) for e in li]
            return

        assert(len(self.li) == len(li))
        for i in range(len(self)):
            self.li[i] = self.concat(self.li[i], self.reduce(li[i]))

    def __concat(self, tensor1, tensor2):
        assert(type(tensor1) == type(tensor2))
        if isinstance(tensor1, torch.Tensor):
            return torch.cat((tensor1, tensor2), dim=0)
        return np.concatenate((tensor1, tensor2), axis=0)

    def __norm(self, tensor):
        # the activation / feature map of one block should be 4-dim tensor
        assert(len(tensor.size()) == 4)
        # only keep the first dimension, i.e. data batch
        return torch.tensor([torch.norm(tensor[i,:]) for i in range(tensor.size()[0])])

    def __batch_norm(self):
        self.li = [torch.norm(e).item() for e in self.li]

    def __pca_transform(self, tensor):
        return self.pca.transform(self.__ravel(tensor))
    
    def __batch_pc(self):
        # fit pca only by the first feature map (the input of the first block)
        # and only for the first epoch
        if not self.pca:
            print('fit pca! This should happen only once for each layer!')
            self.pca = PCA(n_components=self.n)
            self.pca.fit(self.__ravel(self.li[0]))
            self.reduce = self.__pca_transform # we can now reduce the tensor per batch size to reduce memory cost
            self.li = [np.mean(self.__pca_transform(e), axis=0) for e in self.li]
            return

        self.li = [np.mean(e, axis=0) for e in self.li]

    def __ravel(self, tensor):
        mat = tensor.cpu().numpy()
        return mat.reshape(mat.shape[0], -1)

    def __len__(self):
        return len(self.li)


class Hooker(object):
    """
        forward (activation) / backward (gradient) tracker
    """
    def __init__(self, block):
        self.hooker = block.register_forward_hook(self.hook)
        self.input = None
        self.output = None

    def hook(self, block, input, output):
        self.input = input
        self.output = output

    def unhook(self):
        self.hooker.remove()


class LayerHooker(object):
    def __init__(self, layer, layername=None, skipfirst=True, scale_stepsize=False, device=None, save_activation=False, pca=None):

        self.hookers = [Hooker(block) for block in layer]

        if not layername:
            self.layername = ''
        else:
            self.layername = layername

        if skipfirst:
            self.start_block = 1
        else:
            self.start_block = 0

        # this is deprecated, scale stepsize when analyzing
        if scale_stepsize:
            warnings.warn("scale_stepsize is deprecated, scale when analyzing using recorded stepsize",
                          warnings.DeprecationWarning)

        self.device = device

        self.save_activation = save_activation
        self.pca = pca
        self.init_record()

    def init_record(self):
        # recorded feature maps
        self.records = dict()
        self.records['act_norm'] = Record(reduction='norm')
        self.records['res_norm'] = Record(reduction='norm')
        self.records['acc_norm'] = Record(reduction='norm')
        if self.save_activation:
            self.records['act_pc2'] = Record(reduction='pc', n=2, pca=self.pca)
    
    # def reset_record(self):
    #     if self.save_activation:
    #         assert('act_pc2' in self.records)
    #         pca = self.records['act_pc2'].pca

    #     self.records = dict()
    #     self.records['act_norm'] = Record(reduction='norm')
    #     self.records['res_norm'] = Record(reduction='norm')
    #     self.records['acc_norm'] = Record(reduction='norm')
    #     if self.save_activation:
    #         self.records['act_pc2'] = Record(reduction='pc', pca=pca)

    def collect(self):
        activations, residuals, accelerations = self._get_activations()

        # norms
        self.records['act_norm'].absorb(activations)
        self.records['res_norm'].absorb(residuals)
        self.records['acc_norm'].absorb(accelerations)

        if self.save_activation:
            # pcs
            self.records['act_pc2'].absorb(activations)

    def output(self):
        for key in self.records:
            # print(key, records[key].li[0].size())
            # each record should contain the number of validation examples
            self.records[key].batch_reduce()

        if not self.pca:
            self.pca = self.records['act_pc2'].pca

        # reset records
        records_ = copy.deepcopy(self.records)
        # self.reset_record()
        self.init_record()
        return records_

    def close(self):
        for hooker in self.hookers:
            hooker.unhook()

    def _get_activations(self, detach=True):
        """
            It's very weird that input is a tuple including `device`, but output is just a tensor..
        """
        # activations
        # if original model, the residual of the first block can't be calculated because of inconsistent dimension
        activations = []
        for hooker in self.hookers[self.start_block:]:
            # activations.append(hooker.input[0].cpu().detach()) # cpu is much slower than on gpu
            if detach:
                activations.append(hooker.input[0].detach())
            else:
                activations.append(hooker.input[0])
        # activations.append(hooker.output.cpu().detach())
        if detach:
            activations.append(hooker.output.detach())
        else:
            activations.append(hooker.output)

        # migrate to single gpu when multiple gpu distributed training
        if self.device:
            activations = [act.to(self.device) for act in activations]

        # residuals
        residuals = []
        # for b, (input, output) in enumerate(zip(activations[:-1], activations[1:])):
        for input, output in zip(activations[:-1], activations[1:]):
            res = output - input
            # if self.scale_stepsize:
            #     res /= arch[b]
            residuals.append(res)

        # truncated errors / or accelerations
        accelerations = []
        for last, now in zip(residuals[:-1], residuals[1:]):
            accelerations.append(now - last)

        return activations, residuals, accelerations

    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)


class ModelHooker(object):
    def __init__(self, model_name, dpath, resume=False, atom='block',
                 scale_stepsize=False, scale=True, device=None, trace=['norm']):

        self.dpath = dpath
        self.device = device

        self.atom = atom
        self.scale = scale

        # this is deprecated, scale stepsize when analyzing
        if scale_stepsize:
            warnings.warn("scale_stepsize is deprecated, scale when analyzing using recorded stepsize",
                          warnings.DeprecationWarning)

        self.skipfirst=True
        if model_name.startswith('transresnet'):
            self.skipfirst=False

        self.layerHookers = []

        self.trace = trace
        self.history = defaultdict(list)

        # self.save_norm = False
        # if 'norm' in trace:
        #     self.save_norm = True
        #     self.history_norm = []

        # self.save_activation = False
        # if 'pc2' in trace:
        #     self.save_activation = True
        #     self.history_activations = []

        self.logger = Logger(os.path.join(dpath, 'Avg_truncated_err.txt'))
        if not resume:
            self.logger.set_names(['epoch', 'layer1', 'layer2', 'layer3'])

    def hook(self, model):

        # inherit pc if rehook
        pcas = []
        if 'pc2' in self.trace and self.layerHookers:
            pcas = [layerHooker.records['act_pc2'].pca for layerHooker in self.layerHookers]
        pcas = peekable(iter(pcas))

        # switch model module based on dataparallel or not
        if torch.cuda.device_count() > 1:
            model_module = model.module
        else:
            model_module = model

        self.layerHookers = []
        for key in model_module._modules:
            if key.startswith('layer'):
                self.layerHookers.append(LayerHooker(model_module._modules[key],
                                                     layername=key,
                                                     skipfirst=self.skipfirst,
                                                     device=self.device,
                                                     save_activation='pc2' in self.trace,
                                                     pca=next(pcas) if pcas else None))

    def collect(self):
        for layerHooker in self.layerHookers:
            layerHooker.collect()

    def output(self, epoch):
        err_norms = []
        # norms = []
        # if self.save_activation:
        #     activations = []
        if self.trace:
            layer_agg = defaultdict(list)

        for layerHooker in self.layerHookers:
            if len(layerHooker) < 3:
                print('Cannot calculater errs for this layer!')
                return None
            record = layerHooker.output()
            # this only works for in-situ err check, won't affect output norms
            if self.scale:
                # scale acceleration by residuals
                err_norms.append([2 * acc / (res0 + res1) for acc, res0, res1 in zip(record['acc_norm'].li,
                                                                                     record['res_norm'].li[:-1],
                                                                                     record['res_norm'].li[1:])])
            else:
                err_norms.append(record['acc_norm'].li)
                # scale residual by activations
                # res_norms = [2 * res / (act0 + act1) for res, act0, act1 in zip(res_norms, act_norms[:-1], act_norms[1:])])
            
            # save some information to file
            if 'norm' in self.trace:
                layer_agg['norm'].append([record['act_norm'].li, record['res_norm'].li, record['acc_norm'].li])
            if 'pc2' in self.trace:
                layer_agg['pc2'].append(record['act_pc2'].li)

        for key in self.trace:
            self.history[key].append(layer_agg[key])

        avg_err_norms = [statistics.mean(errs) for errs in err_norms]
        self.logger.append([epoch, *avg_err_norms])

        if self.atom == 'block':
            return err_norms
        elif self.atom == 'layer':
            return avg_err_norms
        elif self.atom == 'model':
            return statistics.mean([e for errs in err_norms for e in errs])
        else:
            raise KeyError('atom %s not supported!' % self.atom)

    def close(self):
        for layerHooker in self.layerHookers:
            layerHooker.close()
        self.logger.close()

        if 'norm' in self.trace:
            with open(os.path.join(self.dpath, 'norm_history.pkl'), 'wb') as f:
                pickle.dump(self.history['norm'], f)

        if 'pc2' in self.trace:
            with open(os.path.join(self.dpath, 'activation_history.pkl'), 'wb') as f:
                pickle.dump(self.history['pc2'], f)

    def __len__(self):
        return len(self.layerHookers)

    def __iter__(self):
        return iter(self.layerHookers)



