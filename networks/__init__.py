import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import UNet, GRURDSEUNet, DuRDN


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts, ic=1):
    if name == 'UNet':
        network = UNet(in_channels=ic, residual=False, depth=opts.UNet_depth, wf=opts.UNet_filters, norm = opts.norm)

    elif name == 'DuRDN':
        network = DuRDN(n_channels=ic, n_filters=opts.DuRDN_filters, n_denselayer=6, growth_rate=32, norm=opts.norm)

    elif name == 'GRURDSEUNet':
        network = GRURDSEUNet(n_channels=opts.n_channels, n_filters=64, n_denselayer=4, growth_rate=32)

    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)

