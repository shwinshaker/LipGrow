#!./env python
from __future__ import absolute_import
import torch
import torch.nn.functional as F
import os
import time
import pickle

from . import Logger

__all__ = ['LipHooker']

def spec_norm(weight, input_dim):
    # exact solution by svd and fft
    import numpy as np
    assert len(input_dim) == 2
    assert len(weight.shape) == 4
    fft_coeff = np.fft.fft2(weight, input_dim, axes=[2, 3])
    D = np.linalg.svd(fft_coeff.T, compute_uv=False, full_matrices=False)
    return np.max(D)


class Hooker:
    """
        hook on single node, e.g. conv, bn, relu
    """

    eps = 1e-5

    def __init__(self, name, node, device=None, n_power_iterations=100):

        # name it
        class_name = node.__class__.__name__
        if 'conv' in name: assert class_name.startswith('Conv'), 'Node name inconsistent %s - %s' % (class_name, name)
        if 'bn' in name: assert class_name.startswith('BatchNorm'), 'Node name inconsistent %s - %s' % (class_name, name)
        self.name = name
        self.module = node
        self.device = device
        self.n_power_iterations = n_power_iterations

        # lip calculation function
        """
            don't have to consider downsample here,
            because downsample is not part of the residual block
        """
        if class_name.startswith('Conv'):
            self.lip = self.__conv_lip
        elif class_name.startswith('BatchNorm'):
            self.lip = self.__bn_lip
        else:
            self.lip = lambda: torch.squeeze(torch.ones(1)) # Lipschitz constant 1 for any other nodes

        # extraction protocol
        self.hooker = node.register_forward_hook(self.hook)

        # ease pycharm complain
        self.input = None
        self.output = None

    def hook(self, node, input, output):
        self.input = input
        self.output = output

    def unhook(self):
        self.hooker.remove()
        self.__remove_buffers()

    def __conv_lip(self):
        # only when needed, i.e. after the entire validation batch, do power iteration and compute spectral norm, to gain efficiency

        buffers = dict(self.module.named_buffers())
        if 'u' not in buffers:
            assert 'v' not in buffers
            assert 'sigma' not in buffers
            self.__init_buffers(self.input[0].size(), self.output.size())

        # get buffer
        v_ = self.__get_buffer('v')
        u_ = self.__get_buffer('u')
        sigma_ = self.__get_buffer('sigma')

        # get weight
        weight = self.__get_parameter('weight')
        stride = self.module.stride
        padding = self.module.padding

        # power iteration
        v, u, sigma = v_.clone().to(self.device), \
                      u_.clone().to(self.device), \
                      sigma_.clone().to(self.device)
        """
            The output of deconvolution may not be exactly same as its convolution counterpart
            dimension lost when using stride > 1
            that's why need additional output padding
            See:
                https://towardsdatascience.com/is-the-transposed-convolution-layer-and-convolution-layer-the-same-thing-8655b751c3a1
                http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        """
        transpose_dim = stride[-1] * (u.size()[-1]-1) + weight.size()[-1] - 2 * padding[-1]
        output_padding = v.size()[-1] - transpose_dim
        for _ in range(self.n_power_iterations):
            u = F.conv2d(v, weight, stride=stride, padding=padding, bias=None)
            u = self.__normalize(u)
            v = F.conv_transpose2d(u, weight, stride=stride, padding=padding, output_padding=output_padding)
            v = self.__normalize(v)
            
        sigma = torch.norm(F.conv2d(v, weight, stride=stride, padding=padding, bias=None).view(-1))
        # print('%s - specnorm_iter: %.4f' % (self.name, sigma.item()))
        # comparison with exact solution
        # print('%s - specnorm_iter: %.4f - specnorm_svd: %.4f' % (self.name, sigma.item(), spec_norm(weight.cpu().numpy(), u.size()[2:])))

        # modify buffer - because tensor are copied for every operation, needs to modify the memory
        v_.copy_(v)
        u_.copy_(u)
        sigma_.copy_(sigma)

        # output
        return sigma

    def __init_buffers(self, input_dim, output_dim):
        # input shape is of length 4, includes an additional batch
        assert len(input_dim) == 4
        assert len(output_dim) == 4
        # discard the batch dim
        v_dim = (1, *input_dim[1:]) 
        u_dim = (1, *output_dim[1:])
        # print(self.name, v_dim, u_dim) # should be (1, 16, 32, 32) and (1, 16, 32, 32) for the first one

        v = self.__normalize(torch.randn(v_dim))
        u = self.__normalize(torch.randn(u_dim))

        self.module.register_buffer('v', v)
        self.module.register_buffer('u', u)
        self.module.register_buffer('sigma', torch.ones(1))

    def __remove_buffers(self):
        pass
        # delattr(self.module, 'v')
        # delattr(self.module, 'u')
        # delattr(self.module, 'sigma')

    def __bn_lip(self):
        weight = self.__get_parameter('weight')
        var = self.__get_buffer('running_var')
        # attention: running_var is the var for evaluation, not used for training
        assert self.module.eps == 1e-5
        # this will return a python number, no need to do tensor.item() again
        lip = torch.max(torch.abs(weight) / torch.sqrt(var + self.module.eps))
        # print('%s - lip: %.4f - weight: %.4f - var: %.4f' % (self.name, lip.item(), torch.max(torch.abs(weight)).item(), torch.max(var).item()))
        return lip

    def __get_parameter(self, name):
        return dict(self.module.named_parameters())[name].detach()

    def __get_buffer(self, name):
        return dict(self.module.named_buffers())[name].detach()

    def __normalize(self, tensor):
        dim = tensor.size()
        return F.normalize(tensor.view(-1), dim=0).view(dim)
        

class BlockHooker:
    # named_children -> immediate children

    def __init__(self, name, block, device=None, n_power_iterations=100):
        assert block.__class__.__name__ in ['BasicBlock', 'Bottleneck'], block.__class__.__name__
        self.name = name
        self.device = device

        self.hookers = []
        for name, node in block.named_children():
            hooker = Hooker('.'.join([self.name, name]), node, device=device,
                            n_power_iterations=n_power_iterations)
            # print(name)
            self.hookers.append(hooker)

    def lip(self):
        # nodes should be composition, but seems no way to know that here
        # this is a major drawback if we don't register hooker when building the model
        # TODO: some log, temmporarily
        self._lips = [hooker.lip() for hooker in self.hookers]
        self._lip = torch.prod(torch.tensor(self._lips))
        self._lip_conv = torch.prod(torch.tensor([l for l, b in zip(self._lips, self.hookers) if 'conv' in b.name]))
        self._lip_bn = torch.prod(torch.tensor([l for l, b in zip(self._lips, self.hookers) if 'bn' in b.name]))
        # return torch.prod(torch.tensor([hooker.lip() for hooker in self.hookers]))
        # return torch.prod(torch.tensor(self._lips))
        return self._lip

    def remove(self):
        for hooker in self.hookers:
            hooker.unhook()

    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)


class LayerHooker:
    # named_children -> immediate children
    # no need to skip first?

    def __init__(self, name, layer, device=None):
        assert layer.__class__.__name__ == 'Sequential'
        self.name = name
        self.device = device

        self.hookers = [BlockHooker('.'.join([self.name, name]), block, device=device) for name, block in layer.named_children()]

    def lip(self):
        # return torch.mean(torch.tensor([hooker.lip() for hooker in self.hookers]))
        # return torch.max(torch.tensor([hooker.lip() for hooker in self.hookers]))
        return torch.tensor([hooker.lip() for hooker in self.hookers])

    def remove(self):
        for hooker in self.hookers:
            hooker.remove()
        
    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)


class LipHooker:
    '''
        Lipschitz hooker
    '''

    def __init__(self, model_name, dpath, device=None, **kwargs):

        self.dpath = dpath
        self.device = device # TODO: device gather for multi-gpu training

        # self.logger = None
        self.logger = Logger(os.path.join(dpath, 'Lipschitz.txt'))
        self.logger.set_names(['epoch', 'max', 'min', 'median', 'mean', 'std', 'overhead(secs)'])

        self.logger_conv = Logger(os.path.join(dpath, 'Lipschitz_conv.txt'))
        self.logger_conv.set_names(['epoch', 'max', 'min', 'median', 'mean', 'std', 'overhead(secs)'])

        self.logger_bn = Logger(os.path.join(dpath, 'Lipschitz_bn.txt'))
        self.logger_bn.set_names(['epoch', 'max', 'min', 'median', 'mean', 'std', 'overhead(secs)'])

        # self.history = []
        self.node_logger = None

    def hook(self, model):
        # switch model module based on dataparallel or not
        if torch.cuda.device_count() > 1:
            model_module = model.module
        else:
            model_module = model

        self.hookers = []
        for name, layer in model_module.named_children():
            if name.startswith('layer'):
                self.hookers.append(LayerHooker(name, layer, device=self.device))
        
        if self.node_logger:
            self.node_logger.close()
        num_blocks_per_layer = len(self.hookers[0].hookers)
        self.node_logger = Logger(os.path.join(self.dpath, 'Lipschitz_history_%i.txt' % num_blocks_per_layer))
        node_names = [hooker.name for layerHooker in self.hookers for blockHooker in layerHooker.hookers for hooker in blockHooker.hookers]
        self.node_logger.set_names(['epoch'] + node_names)
            

    # still use 'output' here to conform with original protocol
    def output(self, epoch):
        # now this can also work for 1-1-1 net
        # lip = torch.mean(torch.tensor([hooker.lip() for hooker in self.hookers]))

        start = time.time()
        lips = torch.cat([hooker.lip() for hooker in self.hookers], dim=0)
        elapse = time.time() - start

        self.logger.append([epoch, torch.max(lips), torch.min(lips), torch.median(lips), torch.mean(lips), torch.std(lips), elapse])
        
        # test: block lip - conv only
        _lip_conv = torch.tensor([blockHooker._lip_conv for layerHooker in self.hookers for blockHooker in layerHooker.hookers])
        self.logger_conv.append([epoch, torch.max(_lip_conv), torch.min(_lip_conv), torch.median(_lip_conv), torch.mean(_lip_conv), torch.std(_lip_conv), elapse])

        # test: block lip - batch norm only
        _lip_bn = torch.tensor([blockHooker._lip_bn for layerHooker in self.hookers for blockHooker in layerHooker.hookers])
        self.logger_bn.append([epoch, torch.max(_lip_bn), torch.min(_lip_bn), torch.median(_lip_bn), torch.mean(_lip_bn), torch.std(_lip_bn), elapse])

        # test: examine each node
        _lips = [_lip for layerHooker in self.hookers for blockHooker in layerHooker.hookers for _lip in blockHooker._lips]
        self.node_logger.append([epoch] + _lips)

        # self.history.append([l.item() for l in lips])
        return torch.mean(lips).item()
        # return torch.max(lips).item()
        #### return torch.mean(_lip_conv).item()

    def close(self):
        for hooker in self.hookers:
            hooker.remove()
        self.logger.close()
        self.logger_conv.close()
        self.logger_bn.close()

        if self.node_logger:
            self.node_logger.close()

        # with open(os.path.join(self.dpath, 'lipschitz_history.pkl'), 'wb') as f:
        #     pickle.dump(self.history, f)

    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)

        


