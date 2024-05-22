import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

def init_weights(self, init_type='xavier_uniform', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    
    self.apply(init_func)        
    # propagate to children
    for n, m in self.named_children():
        # if hasattr(m, 'init_weights'):
        # print("Initializating ", n)
        m.apply(init_weights)
        
def normalize(tensor, mins=None, maxs=None):
    # min max normalization to [-1,1]
    mins = mins if mins is not None else tensor.min(dim=0)[0]
    maxs = maxs if maxs is not None else tensor.max(dim=0)[0]
    min_x, min_y = mins
    max_x, max_y = maxs

    x_scale = 2 / (max_x - min_x)
    y_scale = 2 / (max_y - min_y)

    # Normalize tensor
    tensor_normalized = torch.empty_like(tensor).float()
    tensor_normalized[:, 0] = (tensor[:, 0] - min_x) * x_scale - 1
    tensor_normalized[:, 1] = (tensor[:, 1] - min_y) * y_scale - 1

    return tensor_normalized, mins, maxs

def unnormalize(tensor_normalized, mins, maxs):
    # mins = tensor.min(dim=0)[0]
    # maxs = tensor.max(dim=0)[0]
    min_x, min_y = mins
    max_x, max_y = maxs

    x_scale = 2 / (max_x - min_x)
    y_scale = 2 / (max_y - min_y)

    # Unnormalize tensor
    tensor_unnormalized = torch.empty_like(tensor_normalized)
    tensor_unnormalized[:, 0] = (tensor_normalized[:, 0] + 1) / x_scale + min_x
    tensor_unnormalized[:, 1] = (tensor_normalized[:, 1] + 1) / y_scale + min_y

    return tensor_unnormalized

def get_writer(model, inputs, path, ext_name):
    writer = SummaryWriter(os.path.join(path, '', 'events', ext_name))
    # color_corr = torch.Tensor(np.load(os.path.join(path,'color_corr.npy')).reshape(3, 256, 256))  # TODO generalize for non 256 H x W
    # plt.imshow(color_corr ** (1.0/2.0))
    # color_corr = color_corr ** (1.0 / 2.0)
    # grid = torchvision.utils.make_grid(color_corr)
    # writer.add_image('color correlation reference', color_corr)
    if inputs:
        writer.add_graph(model, inputs)
    # TODO add some visualization of results here. see tutorial for how they did it with classification
    return writer
