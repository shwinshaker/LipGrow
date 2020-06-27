#!./env python

class Grower:

    def __init__(self, mode, device):

        # settings
        self.mode = mode

        # adaptive indictor
        self.hooker = 
        self.trigger =

        # learning rate scheduler
        self.scheduler = 
        self.model = 
            # weights in state dict
            # include current depth and final depth in buffer
        self.optimizer =
            # include max learning rate, etc in scheduler

    def compile(self):
        assert self.mode in ['fixed', 'adapt'], 'mode %s not found!' % self.mode 
        self.grow = {'fixed': self.grow_fixed,
                     'adapt': self.grow_adapt}.get(self.mode)
        
    def grow(self):
        pass
        return model, optimizer

    def grow_fixed(self):
        return model, optimizer

    def grow_adapt(self):
        return model, optimizer

    def __model_parallel(self)
