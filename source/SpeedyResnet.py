import functools
from functools import partial
import math
import os
import copy

import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import transforms
import time

# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

hyp = {
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .5, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'conv_norm_pow': 2.6,
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    }
}

#############################################
#            Network Components             #
#############################################

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling
    
# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'], weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, norm=False, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.norm = norm

    def forward(self, x):
        if self.training and self.norm:
            # TODO: Do/should we always normalize along dimension 1 of the weight vector(s), or the height x width dims too?
            with torch.no_grad():
                F.normalize(self.weight.data, p=self.norm)
        return super().forward(x)

class Linear(nn.Linear):
    def __init__(self, *args, norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.norm = norm

    def forward(self, x):
        if self.training and self.norm:
            # TODO: Normalize on dim 1 or dim 0 for this guy?
            with torch.no_grad():
                F.normalize(self.weight.data, p=self.norm)
        return super().forward(x)
    
# can hack any changes to each residual group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, norm):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in, channels_out, norm=norm)
        self.conv2 = Conv(channels_out, channels_out, norm=norm)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        residual = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = x + residual # haiku
        return x
    
#############################################
#            Network Definition             #
#############################################

scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
    'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 80
}

class SpeedyResNet(nn.Module):
    def __init__(self, network_dict, nseq):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function
        self.nseq = nseq

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        batch = x.shape[0]
        # if not self.training:
        #     x = torch.cat((x, torch.flip(x, (-1,))))
        x = x.reshape(-1, *x.shape[2:])
        x = self.net_dict['initial_block']['project'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['residual1'](x)
        x = self.net_dict['residual2'](x)
        x = self.net_dict['residual3'](x)
        x = self.net_dict['pooling'](x)
        #x = self.net_dict['linear'](x)
        
        x = torch.flatten(x, 1)
        
        # Reduce feature to 512 -> 96
        x = self.net_dict['linear'](x)
        
        # Reshaper size to (B, Seq, feature)
        x = x.reshape(batch, self.nseq, -1)
        
        return x

def make_SpeedyResnet(nseq):
    # TODO: A way to make this cleaner??
    # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
    whiten_conv_depth = 3
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'project': Conv(whiten_conv_depth, depths['init'], kernel_size=1, norm=2.2), # The norm argument means we renormalize the weights to be length 1 for this as the power for the norm, each step
            'activation': nn.GELU(),
        }),
        'residual1': ConvGroup(depths['init'],   depths['block1'], hyp['net']['conv_norm_pow']),
        'residual2': ConvGroup(depths['block1'], depths['block2'], hyp['net']['conv_norm_pow']),
        'residual3': ConvGroup(depths['block2'], depths['block3'], hyp['net']['conv_norm_pow']),
        'pooling': FastGlobalMaxPooling(),
        'linear': Linear(depths['block3'], depths['num_classes'], bias=False, norm=5.)
    })

    net = SpeedyResNet(network_dict, nseq=nseq)
    #net = net.to('cuda:0')
    net = net.to(memory_format=torch.channels_last) # to appropriately use tensor cores/avoid thrash while training
    net.train()
    #net.half() # Convert network to half before initializing the initial whitening layer.
    return net

if __name__ == "__main__":
    model = make_SpeedyResnet(nseq=10).to('cuda:0').half()
    x = torch.randn(20, 10, 3, 64, 128).to('cuda:0').half()
    
    t1 = time.perf_counter(), time.process_time()
    y = model(x)
    t2 = time.perf_counter(), time.process_time()
    
    print('\n output shape:', y.shape)
    print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
